#!/usr/bin/env sh

./build/tools/caffe train -gpu $1 \
  --solver=$2
