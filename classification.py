import tensorflow as tf
import numpy as np


x_train = []
x_test = []
Y_train = []
Y_test = []
print("lendo arq")
with open("fall_dataset", 'rb') as file:
    x_train, x_test, Y_train, Y_test = np.load(file, allow_pickle=True)


print("leu")
# x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.array(x_train)
x_test = np.array(x_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

    
inp = tf.keras.layers.Input(shape=(227,227,3))
layer = tf.keras.layers.Conv2D(3, (11,11), strides=(4,4), activation="relu")(inp)
layer = tf.keras.layers.MaxPooling2D((3 ,3), strides=(2,2))(layer)
layer = tf.keras.layers.Conv2D(96, (5,5), activation="relu")(layer)
layer = tf.keras.layers.MaxPooling2D((2,2))(layer)
layer = tf.keras.layers.Conv2D(256, (3,3), activation="relu")(layer)
layer = tf.keras.layers.Conv2D(192, (3,3), activation="relu")(layer)
# layer = tf.keras.layers.Conv2D(192, (3,3), activation="relu")(layer)
layer = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2))(layer)
layer = tf.keras.layers.Flatten()(layer)
layer = tf.keras.layers.Dense(4096, activation="relu")(layer)
layer = tf.keras.layers.Dense(1025, activation="relu")(layer)
output = tf.keras.layers.Dense(1, activation="sigmoid")(layer)

model = tf.keras.Model(inputs=inp, outputs=output)
model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])

model.fit(x_train, Y_train, batch_size= 2, epochs=20)

model.evaluate(x_test, Y_test)

# # TensorFlow e tf.keras
# import tensorflow as tf
# from tensorflow import keras

# # Librariesauxiliares
# import numpy as np
# import matplotlib.pyplot as plt

# print(tf.__version__)

# fashion_mnist = keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# train_images = train_images / 255.0

# test_images = test_images / 255.0

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=10)