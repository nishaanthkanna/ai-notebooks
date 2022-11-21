from tensorflow import keras
from keras import layers

class Lenet(keras.models):

    def __init__(self):
        super().__init__()

        self.conv1 = layers.Conv2D(filters=64, kernel_size=2, strides=2)

        self.dense = layers.Dense(units=120)