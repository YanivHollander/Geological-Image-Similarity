import unittest
import tensorflow as tf
from Models_TF import Siamese
from DataIO_TF import GeologicalDataset
from Globals import DATA_PATH, CHECKPOINT_PATH
import os
import tensorflow.keras.backend as K
from tensorflow import Tensor
from typing import Callable

def contrastiveLoss (y, yPred, m = 1):
    """
    Compute the contrastive loss between model predictions and labels
    :param y: Labels to indicate similarity (1) or dissimilarity (0) between images
    :param yPred: Mode predictions as to level of similarity between query and reference images
    :param m: Contrastive margin
    :return: Loss
    """
    y = tf.cast(y, yPred.dtype)
    return K.mean(y * yPred + (1. - y) * K.maximum(m - yPred, 0.))

def trainSiamese(
        path = DATA_PATH,
        batchSize = 32,
        shuffleData = True,
        learningRate = 0.01,
        numEpochs = 10,
        validationSplit = 0.05,
        checkpointPath = CHECKPOINT_PATH):
    """
    Trains a Siamese metwork
    :param path: Path to training dataset
    :param batchSize: Batch size
    :param shuffleData: Defines shuffling of the data before each training epoch
    :param learningRate: ADAM learning rate
    :param numEpochs: Nuber of Epochs
    :param validationSplit: Percentage of dataset to put aside for validation at teh end of each training epoch
    :param checkpointPath: Model checkpoint path (where to save model weights after training)
    """
    ### TPU initialization at program start
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Device:', tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except:
        strategy = tf.distribute.get_strategy()
    print('Number of replicas:', strategy.num_replicas_in_sync)

    ### Load data
    ds = GeologicalDataset(path)
    ds.initDataset()
    imageShape = ds.imageShape()

    ### Strategy scope
    with strategy.scope():
        model = Siamese(shape=imageShape, embeddingSize=32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
        model.compile(loss=contrastiveLoss, optimizer=optimizer)

    ### Model save callback
    modelCallback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpointPath, 'cp.cpkt'),
                                                       save_weights_only=True, verbose=1)

    ### Training
    return model.fit([ds.queries(), ds.references()], ds.labels(),
              epochs=numEpochs, verbose=1, batch_size=batchSize, validation_split=validationSplit,
              callbacks=[modelCallback], shuffle=shuffleData)

class MyTestCase(unittest.TestCase):
    def test_small_training(self):
        from Globals import SMALL_DATA_PATH
        from matplotlib import pyplot as plt
        history = trainSiamese(path=SMALL_DATA_PATH)
        # plt.figure()
        # plt.plot(history.history['loss'])
        # plt.show()

    def test_training(self):
        trainSiamese(path=DATA_PATH, numEpochs=100)

    def test_contrastive(self):
        import numpy as np
        y = np.zeros(shape=(32, 1))
        yPred = np.random.rand(32, 1)
        print(contrastiveLoss(y, yPred.astype('float32')))

if __name__ == '__main__':
    unittest.main()
