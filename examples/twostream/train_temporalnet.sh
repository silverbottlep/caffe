#!/usr/bin/env sh

./build/tools/caffe train -gpu 2 \
	--solver=examples/twostream/temporalnet_solver.prototxt
