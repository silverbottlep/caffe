#!/usr/bin/env sh

./build/tools/caffe train -gpu 0 \
	--solver=examples/twostream/spatialnet_solver.prototxt
