#!/usr/bin/env sh

./build/tools/caffe train -gpu 0 \
	--solver=examples/twostream/spatialnet_ft_solver.prototxt \
	--weights=examples/twostream/snapshot/spatialnet_iter_350000.caffemodel
