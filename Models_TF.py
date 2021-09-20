import unittest
from tensorflow import keras as keras
from tensorflow.keras import layers as layers
import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras.backend as K

def ConvBlock(numFilters, kernelSize = 3, training = True, dropoutRate = 0.4) -> keras.Sequential:
    """
    A convolution block. Composed of one convolutional layer followed by max pool
    """
    net = keras.Sequential()
    net.add(layers.Conv2D(filters=numFilters, kernel_size=kernelSize, activation='relu', padding='valid',
                          data_format='channels_last', use_bias=False))         # 1) Convolution
    # net.add(tf.keras.layers.BatchNormalization(axis=3))
    net.add(layers.MaxPool2D())                                                 # 2) Max pooling
    if training:
        net.add(layers.Dropout(rate=dropoutRate))
    return net

def Encoding(shape, embeddingSize = 64, training = True) -> keras.Model:
    """
    Network to encode an image to an embedding vector
    """
    inputs = layers.Input(shape=shape)
    x = inputs
    x = ConvBlock(64, training=training)(x)
    x = ConvBlock(64, training=training)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=embeddingSize)(x)
    return keras.Model(inputs=inputs, outputs=x, name="encoding")

def Siamese(shape, embeddingSize = 64, training = True) -> keras.Model:
    encoding = Encoding(shape, embeddingSize, training=training)
    queries = layers.Input(shape=shape)
    references = layers.Input(shape=shape)
    queryEmbeddings = encoding(queries)
    referenceEmbeddings = encoding(references)
    outputs = K.maximum(tf.norm(queryEmbeddings - referenceEmbeddings, axis=1), K.epsilon())
    return keras.Model(inputs=(queries, references), outputs=outputs, name="siamese")

class MyTestCase(unittest.TestCase):
    def test_encoding(self):
        encoding = Encoding(shape=(28, 28, 3))
        encoding.summary()

    def test_siamese(self):
        import numpy as np
        siamese = Siamese(shape=(28, 28, 3), embeddingSize = 8)
        siamese.summary()
        x = np.ones(shape=(32, 28, 28, 3), dtype=np.float32)
        y = np.zeros(shape=(32, 28, 28, 3), dtype=np.float32)
        print(siamese((x, y)))

if __name__ == '__main__':
    unittest.main()
