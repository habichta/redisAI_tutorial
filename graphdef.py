import tensorflow.compat.v1 as tf
gf = tf.GraphDef()
gf.ParseFromString(open('reference_model/saved_model.pb','rb').read())
