from keras.datasets import mnist
from keras import Model
from keras.layers import Dense, Dropout, Input, Flatten
import numpy as np
from keras.utils import np_utils

def dataloader(dataset):
    if dataset.lower() == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_train = x_train.astype('float32')
        x_train /= 255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_test = x_test.astype('float32')
        x_test /= 255
        return x_train, y_train, x_test, y_test

def FCnet():
    input = Input(shape=(28, 28, 1), name='input')
    x = Flatten()(input)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(10, activation="softmax")(x)
    model = Model(inputs=[input], outputs=[output])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = dataloader('mnist')
    model = FCnet()

    model.fit(X_train, Y_train, epochs=10)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)
    model.save("./mnist.h5")


