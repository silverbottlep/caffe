#!/usr/bin/env sh

./build/tools/caffe train -gpu $1 \
	--solver=examples/split2/temporalnet_solver.prototxt

./build/tools/caffe train -gpu $1 \
	--solver=examples/split2/temporalnet_solver2.prototxt \
	--snapshot=examples/split2/snapshot/temporalnet_iter_30000.solverstate

./build/tools/caffe train -gpu $1 \
	--solver=examples/split2/temporalnet_solver3.prototxt \
	--snapshot=examples/split2/snapshot/temporalnet_iter_80000.solverstate
