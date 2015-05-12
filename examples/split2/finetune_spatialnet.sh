#!/usr/bin/env sh

./build/tools/caffe train -gpu $1 \
	--solver=$2 \
	--weights=examples/twostream/snapshot/spatialnet_iter_450000.caffemodel
