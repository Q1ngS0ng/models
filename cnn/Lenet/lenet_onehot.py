import numpy as np
import keras
import tensorflow as tf

from keras.layers import Lambda, Input, Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Model
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28) / 255
X_test = X_test.reshape(-1, 28, 28) / 255

Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

def onehot(x, exp = 100):
    max = tf.reduce_max(x)
    x = tf.subtract(x, max)
    x = tf.exp(x * exp)
    return x

# x = tf.random.normal([1, 10], mean=0, stddev = 4)
# x =  onehot(x)
# print(x)

def Lenet():
    input = Input(shape=(28, 28, 1), name='input')
    x = Conv2D(32, 5, activation="relu")(input)
    x = MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(64, 5, activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(100, activation="relu")(x)
    x = Dense(10, activation="softmax")(x)
    output = Lambda(onehot, output_shape=(1, 10))(x)
    model = Model(inputs=[input], outputs=[output])

    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = Lenet()
    model.fit(X_train, Y_train, epochs=2)
    model.evaluate(X_test, Y_test, verbose=2)
    model.save("lenet_mnist_onehot.h5")

