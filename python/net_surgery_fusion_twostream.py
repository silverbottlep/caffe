import numpy as np
import caffe
from caffe.proto.caffe_pb2 import *
from google import protobuf

#model_root = '/home/eunbyung/Works/src/caffe/examples/cons/' 
model_root = '/home/eunbyung/Works/src/caffe/examples/split2/' 
spatialnet_param_file = model_root + 'snapshot/' + 'spatialnet_ft_iter_45000.caffemodel'

temporalnet_param_file = model_root + 'snapshot/' + 'temporalnet_iter_180000.caffemodel'
#temporalnet_param_file = model_root + 'snapshot/' + 'temporalnet_iter_200000.caffemodel'

fusion_proto_file = model_root + 'fusion_twostream.prototxt'
fusion_param_file = model_root + 'snapshot/' + 'fusion_twostream.caffemodel'

# loading spatialnet, and build blob lookup dic
spatialnet = NetParameter()
sf = open(spatialnet_param_file,'rb')
spatialnet.ParseFromString(sf.read())
sf.close()
print "blobs loaded from", spatialnet.name

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
temporalnet.ParseFromString(tf.read())
tf.close()
print "blobs loaded from", temporalnet.name

temporalnet_blob_lookup = dict()
for layer in temporalnet.layers:
    if len(layer.blobs) > 0:
        assert layer.name not in temporalnet_blob_lookup
        new_name = 'flow_' + layer.name
        temporalnet_blob_lookup[new_name] = layer.blobs
        print "-", new_name

# loading fusion net proto description files
fusion = NetParameter()
f = open(fusion_proto_file,'r')
prototxt = f.read()
fusion = protobuf.text_format.Merge(prototxt, fusion)

# build consilience network
for layer in fusion.layers:
    if 'rgb_conv' in layer.name:
        layer.blobs.extend(spatialnet_blob_lookup[layer.name])
        print "-", layer.name
    elif 'rgb_fc6' in layer.name:
        layer.blobs.extend(spatialnet_blob_lookup[layer.name])
        print "-", layer.name
    elif 'rgb_fc7' in layer.name:
        layer.blobs.extend(spatialnet_blob_lookup[layer.name])
        print "-", layer.name
    elif 'rgb_fc8' in layer.name:
        layer.blobs.extend(spatialnet_blob_lookup['rgb_fc8_ucf101'])
        print "-", layer.name
    elif 'flow_conv' in layer.name:
        layer.blobs.extend(temporalnet_blob_lookup[layer.name])
        print "-", layer.name
    elif 'flow_fc' in layer.name:
        layer.blobs.extend(temporalnet_blob_lookup[layer.name])
        print "-", layer.name

out_file = open(fusion_param_file, 'wb')
out_file.write(fusion.SerializeToString())
out_file.close()
