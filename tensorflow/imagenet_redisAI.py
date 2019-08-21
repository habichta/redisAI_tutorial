import redisai as rai
import ml2rt
import numpy as np
import tensorflow as tf
from skimage import io
import json
con=rai.Client()
#model = ml2rt.load_model('reference_model/3/resnet50.pb')
filepath = 'data/guitar.jpg'
numpy_img = io.imread(filepath).astype(dtype=np.float32)
numpy_img = np.expand_dims(numpy_img, axis=0) / 255
print(numpy_img)


#con.modelset('model',rai.Backend.tf, rai.Device.cpu, input=['images'], output=['output'], data=model)


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
