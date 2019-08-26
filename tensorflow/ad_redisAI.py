import redisai as rai
import ml2rt
import numpy as np
import tensorflow as tf
import time
import json
con=rai.Client()
model = ml2rt.load_model('reference_model/4/saved_model2.pb')

con.modelset('model',rai.Backend.tf, rai.Device.cpu, input=['dense_1_input'], output=['dense_6/Relu'], data=model)

for i in range(1000):
    con.tensorset('dense_1_input', np.float32(np.ones((1, 50)), axis=0))
    con.modelrun(
        'model',
        input=['dense_1_input'],
        output=['dense_6/Relu'])
    ret = con.tensorget('dense_6/Relu', as_type=rai.BlobTensor).to_numpy()
    print(ret, ret.shape)
    time.sleep(1)
