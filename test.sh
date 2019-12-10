#!/bin/bash

# Image and model names
TEST_IMG=ADE_val_00001519.jpg
MODEL_PATH=ade20k-resnet50dilated-ppm_deepsup
RESULT_PATH=./

ENCODER=$MODEL_PATH/encoder_epoch_40.pth
DECODER=$MODEL_PATH/decoder_epoch_40.pth

# Download model weights and image
if [ ! -e $TEST_IMG ]; then
  wget -P $RESULT_PATH http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016/images/validation/$TEST_IMG
fi

# Inference
python -u test2.py \
  --imgs $TEST_IMG \
  --cfg config/ade20k-resnet101-upernet.yaml \
  DIR $MODEL_PATH \
  TEST.result ./ \
  TEST.checkpoint epoch_40.pth
