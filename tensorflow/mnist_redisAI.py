import redisai as rai
import ml2rt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage import io
import json
con=rai.Client()
model = ml2rt.load_model('reference_model/1/saved_model.pb')
print(type(model))

model = keras.experimental.load_from_saved_model('reference_model/1')
print(type(model))
#con.modelset('mnist_model',rai.Backend.tf, rai.Device.cpu, input=['input'], output=['output'], data=model)

'''
con.tensorset('images', numpy_img)
con.modelrun(
'model',
input=['images'],
output=['output']
)
ret = con.tensorget('output', as_type=rai.BlobTensor).to_numpy()
classes_idx = json.load(open('data/imagenet_classes.json'))
ind = ret.argmax()
print(ind, ret.shape)
print(classes_idx[str(ind-1)])
'''
