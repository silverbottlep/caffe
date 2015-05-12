#!/usr/bin/env sh

#./build/tools/caffe train -gpu $1 \
#	--solver=examples/twostream/temporalnet_solver.prototxt
#
#./build/tools/caffe train -gpu $1 \
#	--solver=examples/twostream/temporalnet_solver2.prototxt \
#	--snapshot=examples/twostream/snapshot/temporalnet_iter_30000.solverstate

./build/tools/caffe train -gpu $1 \
	--solver=examples/twostream/temporalnet_solver3.prototxt \
	--snapshot=examples/twostream/snapshot/temporalnet_iter_80000.solverstate

