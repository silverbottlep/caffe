#!/usr/bin/env sh

./build/tools/caffe train -gpu 0 \
	--solver=examples/twostream/temporalnet_solver.prototxt

./build/tools/caffe train -gpu 0 \
	--solver=examples/twostream/temporalnet_solver2.prototxt \
	--snapshot=examples/twostream/snapshot/temporalnet_iter_50000.solverstate

./build/tools/caffe train -gpu 0 \
	--solver=examples/twostream/temporalnet_solver3.prototxt \
	--snapshot=examples/twostream/snapshot/temporalnet_iter_120000.solverstate
