#!/usr/bin/env sh

./build/tools/caffe train -gpu 2 \
	--solver=examples/twostream/spatialnet_ft_consilience_solver.prototxt \
	--weights=examples/twostream/snapshot/spatialnet_iter_350000.caffemodel
