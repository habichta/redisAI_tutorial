

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import ml2rt
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()

from tensorflow.keras import datasets, layers, models
print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
input = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name="input")
model.add(input)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
output = layers.Dense(10, activation='softmax', name="output")
model.add(output)
model.summary()
print(model.layers[0].weights)
print(model.layers[-1].weights)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)
keras.experimental.export_saved_model(model, 'reference_model/1')
#tf.saved_model.save(model, "reference_model/1")

