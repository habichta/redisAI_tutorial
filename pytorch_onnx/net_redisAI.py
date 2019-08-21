import redisai as rai
import ml2rt
import numpy as np
import tensorflow as tf
from skimage import io
import json
con=rai.Client()
model = ml2rt.load_model('models/model1.pt')


con.modelset('model',rai.Backend.torch, rai.Device.cpu, model)


numpy_img = np.float32(np.random.rand(1, 3, 32, 32),axis=0)
print(numpy_img.shape)

con.tensorset('input', numpy_img)
con.modelrun(
    'model',
    input=['input'],
    output=['output']
)
ret = con.tensorget('output', as_type=rai.BlobTensor).to_numpy()
'''
classes_idx = json.load(open('data/imagenet_classes.json'))
ind = ret.argmax()
print(ind, ret.shape)
print(classes_idx[str(ind-1)])
'''
