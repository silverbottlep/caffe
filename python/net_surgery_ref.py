import numpy as np
import caffe
from caffe.proto.caffe_pb2 import *
from google import protobuf


# Model definitions
ssnet_full_proto_file = (proto_root + "/" +
    "caffenet.trainval.twotower.prototxt")

# Input model parameters
caffenet_param_file = (model_root + "/" +
    "bvlc_reference_caffenet.caffemodel")
#input_model_files = [caffenet_param_file, ssnet_metric_param_file]
input_model_files = [caffenet_param_file]

# Explicit mapping correspondence
name_lookup_file = (proto_root + "/" +
    "name_lookup.txt")

# Output model parameters
ssnet_full_param_file = (model_root + "/"+
    "caffenet.trainval.twotower.caffemodel")

model_root = '/home/eunbyung/Works/src/caffe/examples/twostream/' 
spatialnet_proto_file = (model_root + 'spatialnet_ft.prototxt')
spatialnet_param_file = (model_root + 'snapshot/' + 'spatialnet_ft_iter_45000.caffemodel')
temporalnet_proto_file = (model_root + 'temporalnet_ft.prototxt')
temporalnet_param_file = ('model_root' + 'snapshot/' + 'temporalnet_iter_200000.caffemodel')

def read_proto(filename):
    with open(filename, 'rb') as f:
        proto_str=f.read()
    return proto_str

def read_prototxt(filename):
    with open(filename, 'r') as f:
        prototxt=f.read()
    return prototxt

def build_name_lookup(filename):
    lookup = dict()
    with open(filename) as f:
        for line in f:
            parts = line.split(',')
            lookup[parts[0].strip()] = parts[1].strip()
    return lookup


# Create layer-name mapping
name_lookup = build_name_lookup(name_lookup_file)

def build_blob_lookup(model_files):
    blob_lookup = dict()
    for mf in model_files:
        net = NetParameter()
        net.ParseFromString(read_proto(mf))
        print "Loading blobs from", net.name
        # Enumerate new layers
        for layer in net.layer:
            if len(layer.blobs) > 0:
                assert layer.name not in blob_lookup
                blob_lookup[layer.name] = layer.blobs
                print "-", layer.name
        # Enumerate V1 layers
        for layer in net.layers:
            if len(layer.blobs) > 0:
                assert layer.name not in blob_lookup
                blob_lookup[layer.name] = layer.blobs
                print "-", layer.name
    return blob_lookup

blob_lookup = build_blob_lookup(input_model_files)


# Create the output model, without any blobs.
fnet = NetParameter()
fnet = protobuf.text_format.Merge(
    read_prototxt(ssnet_full_proto_file), fnet)


# Set blobs for output net
request_set = set(name_lookup.keys())
for layer in fnet.layer:
    if layer.name in name_lookup:
        source_name = name_lookup[layer.name]
        assert source_name in blob_lookup
        assert len(layer.blobs) == 0
        layer.blobs.extend(blob_lookup[source_name])
        request_set.remove(layer.name)
if len(request_set) > 0:
    print "Uninitialized layers:"
    for name in request_set:
        print "-", name
else:
    print "All requested layers are initialzied."


def write_proto(net, filename):
    with open(filename, 'wb') as f:
        print "Writing", net.name, "to", filename, "..."
        f.write(net.SerializeToString())
        print "Done."


write_proto(fnet, ssnet_full_param_file)
