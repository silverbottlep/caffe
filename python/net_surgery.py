import numpy as np
import caffe
from caffe.proto.caffe_pb2 import *
from google import protobuf

model_root = '/home/eunbyung/Works/src/caffe/examples/consilience/' 
temporalnet_param_file = model_root + 'snapshot/' + 'temporalnet_iter_200000.caffemodel'

#spatialnet_param_file = model_root + 'snapshot/' + 'spatialnet_ft_iter_45000.caffemodel'
#consiliencenet_param_file = model_root + 'snapshot/' + 'consilience.caffemodel'
#consiliencenet_proto_file = model_root + 'consilience.prototxt'

spatialnet_param_file = model_root + 'snapshot/' + 'spatialnet_vgg19.caffemodel'
consiliencenet_param_file = model_root + 'snapshot/' + 'vgg19_consilience.caffemodel'
consiliencenet_proto_file = model_root + 'vgg19_consilience_spatial.prototxt'


# loading spatialnet, and build blob lookup dic
spatialnet = NetParameter()
sf = open(spatialnet_param_file,'rb')
print "Loading blobs from", spatialnet.name
spatialnet.ParseFromString(sf.read())
sf.close()

spatialnet_blob_lookup = dict()
for layer in spatialnet.layers:
    if len(layer.blobs) > 0:
        assert layer.name not in spatialnet_blob_lookup
        new_name = 'rgb_' + layer.name
        spatialnet_blob_lookup[new_name] = layer.blobs
        print "-", new_name

# loading temporalnet, and build blob lookup dic
temporalnet = NetParameter()
tf = open(temporalnet_param_file,'rb')
print "Loading blobs from", temporalnet.name
temporalnet.ParseFromString(tf.read())
tf.close()

temporalnet_blob_lookup = dict()
for layer in temporalnet.layers:
    if len(layer.blobs) > 0:
        assert layer.name not in temporalnet_blob_lookup
        new_name = 'flow_' + layer.name
        temporalnet_blob_lookup[new_name] = layer.blobs
        print "-", new_name

consiliencenet = NetParameter()
cf = open(consiliencenet_proto_file,'r')
prototxt = cf.read()
consiliencenet = protobuf.text_format.Merge(prototxt, consiliencenet)

# build consilience network
for layer in consiliencenet.layers:
    if 'rgb_conv' in layer.name:
        layer.blobs.extend(spatialnet_blob_lookup[layer.name])
        print "-", layer.name
    elif 'flow_conv' in layer.name:
        layer.blobs.extend(temporalnet_blob_lookup[layer.name])
        print "-", layer.name

out_file = open(consiliencenet_param_file, 'wb')
out_file.write(consiliencenet.SerializeToString())
