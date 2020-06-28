import tensorflow as tf
import numpy as np


x_train = []
x_test = []
Y_train = []
Y_test = []
print("lendo arq")
with open("fall_dataset", 'rb') as file:
    x_train, x_test, Y_train, Y_test = np.load(file, allow_pickle=True)
print(Y_test)
print("leu")
# x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.array(x_train)
x_test = np.array(x_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

x_train = x_train / 255
x_test = x_test / 255

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

model.fit(x_train, Y_train, epochs=20)

results = model.evaluate(x_test, Y_test)

print(results)