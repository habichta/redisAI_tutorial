import redisai as rai
import ml2rt
import numpy as np

con=rai.Client()
model = ml2rt.load_model('reference_model/saved_model.pb')

con.modelset('model',rai.Backend.tf, rai.Device.cpu, input=['input'], output=['output'], data=model)
