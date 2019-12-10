import torch
import torch.nn as nn
import torchvision
from . import resnet, resnext, mobilenet, hrnet
from lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
import math

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict):
        x1,x2,x3,x4 = self.encoder(feed_dict)
        pred = self.decoder(x1, x2, x3, x4)
        return pred


class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
        net_encoder = Resnet(orig_resnet)
        
        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):

        net_decoder = UPerNet(
            num_class=num_class,
            fc_dim=fc_dim,
            use_softmax=use_softmax,
            fpn_dim=512)

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return (x1,x2,x3,x4)


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x

# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        #1x1
        H = 10
        W = 15
        O_H = 1
        O_W = 1
        stride_H = math.floor(H/O_H)
        stride_W = math.floor(H/O_W)
        kernel_H = H - (O_H - 1) * stride_H
        kernel_W = W - (O_W - 1) * stride_W
        self.ppm_pooling_1 = nn.AvgPool2d(kernel_size=(kernel_H, kernel_W), stride=(stride_H, stride_W), padding=0)
        #2x2
        H = 10
        W = 15
        O_H = 2
        O_W = 2
        stride_H = math.floor(H/O_H)
        stride_W = math.floor(H/O_W)
        kernel_H = H - (O_H - 1) * stride_H
        kernel_W = W - (O_W - 1) * stride_W
        self.ppm_pooling_2 = nn.AvgPool2d(kernel_size=(kernel_H, kernel_W), stride=(stride_H, stride_W), padding=0)
        #3x3
        H = 10
        W = 15
        O_H = 3
        O_W = 3
        stride_H = math.floor(H/O_H)
        stride_W = math.floor(H/O_W)
        kernel_H = H - (O_H - 1) * stride_H
        kernel_W = W - (O_W - 1) * stride_W
        self.ppm_pooling_3 = nn.AvgPool2d(kernel_size=(kernel_H, kernel_W), stride=(stride_H, stride_W), padding=0)
        #6x6
        H = 10
        W = 15
        O_H = 6
        O_W = 6
        stride_H = math.floor(H/O_H)
        stride_W = math.floor(H/O_W)
        kernel_H = H - (O_H - 1) * stride_H
        kernel_W = W - (O_W - 1) * stride_W
        self.ppm_pooling_4 = nn.AvgPool2d(kernel_size=(kernel_H, kernel_W), stride=(stride_H, stride_W), padding=0)
        
        # self.ppm_conv = [
        #     nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),BatchNorm2d(512),nn.ReLU(inplace=True)),
        #     nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),BatchNorm2d(512),nn.ReLU(inplace=True)),
        #     nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),BatchNorm2d(512),nn.ReLU(inplace=True)),
        #     nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),BatchNorm2d(512),nn.ReLU(inplace=True))
        # ]
        self.ppm_conv_1 = nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),BatchNorm2d(512),nn.ReLU(inplace=True))
        self.ppm_conv_2 = nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),BatchNorm2d(512),nn.ReLU(inplace=True))
        self.ppm_conv_3 = nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),BatchNorm2d(512),nn.ReLU(inplace=True))
        self.ppm_conv_4 = nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),BatchNorm2d(512),nn.ReLU(inplace=True))
        
        # for scale in pool_scales:
        #     self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
        #     self.ppm_conv.append(nn.Sequential(
        #         nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
        #         BatchNorm2d(512),
        #         nn.ReLU(inplace=True)
        #     ))

        # self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        # self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    # def forward(self, conv_out, segSize=None):
    #     conv5 = conv_out[-1]

    #     input_size = conv5.size()
    #     ppm_out = [conv5]
    #     for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
    #         ppm_out.append(pool_conv(nn.functional.interpolate(
    #             pool_scale(conv5),
    #             (input_size[2], input_size[3]),
    #             mode='bilinear', align_corners=False)))
    #     ppm_out = torch.cat(ppm_out, 1)
    #     f = self.ppm_last_conv(ppm_out)

    #     fpn_feature_list = [f]
    #     for i in reversed(range(len(conv_out) - 1)):
    #         conv_x = conv_out[i]
    #         conv_x = self.fpn_in[i](conv_x) # lateral branch

    #         f = nn.functional.interpolate(
    #             f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
    #         f = conv_x + f

    #         fpn_feature_list.append(self.fpn_out[i](f))

    #     fpn_feature_list.reverse() # [P2 - P5]
    #     output_size = fpn_feature_list[0].size()[2:]
    #     fusion_list = [fpn_feature_list[0]]
    #     for i in range(1, len(fpn_feature_list)):
    #         fusion_list.append(nn.functional.interpolate(
    #             fpn_feature_list[i],
    #             output_size,
    #             mode='bilinear', align_corners=False))
    #     fusion_out = torch.cat(fusion_list, 1)
    #     x = self.conv_last(fusion_out)

    #     if self.use_softmax:  # is True during inference
    #         x = nn.functional.interpolate(
    #             x, size=segSize, mode='bilinear', align_corners=False)
    #         x = nn.functional.softmax(x, dim=1)
    #         return x

    #     x = nn.functional.log_softmax(x, dim=1)

    #     return x


    def forward(self, x1, x2, x3, x4):
        pp3 = self.ppm_conv_1(nn.functional.interpolate(self.ppm_pooling_1(x4),(10, 15),mode='bilinear', align_corners=False))
        pp2 = self.ppm_conv_2(nn.functional.interpolate(self.ppm_pooling_2(x4),(10, 15),mode='bilinear', align_corners=False))
        pp1 = self.ppm_conv_3(nn.functional.interpolate(self.ppm_pooling_3(x4),(10, 15),mode='bilinear', align_corners=False))
        pp0 = self.ppm_conv_4(nn.functional.interpolate(self.ppm_pooling_4(x4),(10, 15),mode='bilinear', align_corners=False))

        ppm_out = torch.cat([x4, pp3, pp2, pp1, pp0], 1)
        p4 = self.ppm_last_conv(ppm_out)

        conv_x3 = self.fpn_in[2](x3)
        f = nn.functional.interpolate(p4, size=(20, 30), mode='bilinear', align_corners=False)
        f = conv_x3 + f
        p3 = self.fpn_out[2](f)

        conv_x2 = self.fpn_in[1](x2)
        f = nn.functional.interpolate(f, size=(40, 60), mode='bilinear', align_corners=False)
        f = conv_x2 + f
        p2 = self.fpn_out[1](f)

        conv_x1 = self.fpn_in[0](x1)
        f = nn.functional.interpolate(f, size=(80, 120), mode='bilinear', align_corners=False)
        f = conv_x1 + f
        p1 = self.fpn_out[0](f)
        
        cat_2 = nn.functional.interpolate(p2,(80, 120),mode='bilinear', align_corners=False)
        cat_3 = nn.functional.interpolate(p3,(80, 120),mode='bilinear', align_corners=False)
        cat_4 = nn.functional.interpolate(p4,(80, 120),mode='bilinear', align_corners=False)

        fusion_out = torch.cat((p1,cat_2,cat_3,cat_4), 1)
        x = self.conv_last(fusion_out)
        x = nn.functional.interpolate(x, size=(512, 768), mode='bilinear', align_corners=False)
        x = nn.functional.softmax(x, dim=1)
        print(x.shape)
        return x
