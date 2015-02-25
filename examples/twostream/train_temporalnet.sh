#!/usr/bin/env sh

./build/tools/caffe train -gpu 0 \
	--solver=examples/twostream/temporalnet_solver.prototxt
