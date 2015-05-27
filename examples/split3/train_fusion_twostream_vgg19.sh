#!/usr/bin/env sh

./build/tools/caffe train -gpu $1 \
  --solver=$2 --weights=examples/split3/snapshot/vgg19_fusion_twostream.caffemodel
