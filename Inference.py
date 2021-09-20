import unittest
from Globals import DATA_PATH, QUERY_DATA_PATH, CHECKPOINT_PATH, SMALL_DATA_PATH
from DataIO_TF import GeologicalDataset
from Models_TF import Siamese
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict
import heapq

class SimilarityPath:
    def __init__(self, queryPath: str, queryLabel: int, referencePaths: List[str], referenceLabels: List[int],
                 similarityValues: List[np.float32]):
        self.queryPath = queryPath
        self.queryLabel = queryLabel
        self.referencePaths = referencePaths
        self.referenceLabels = referenceLabels
        self.similarityValues = similarityValues

def findKSimilar (K: int, queryPath = QUERY_DATA_PATH, referencePath = DATA_PATH, checkpointPath = CHECKPOINT_PATH) -> \
        List[SimilarityPath]:

    ### Load reference data
    referenceDS = GeologicalDataset(referencePath)
    referenceDS.initDataset(training = False)
    imageShape = referenceDS.imageShape()

    ### Load trained model
    model = Siamese(shape=imageShape, embeddingSize=32, training=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss=contrastiveLoss, optimizer=optimizer)

    ### Load query data
    queryDS = GeologicalDataset(queryPath)
    queryDS.initDataset(training = False)

    ### Calculate similarity metric between each query and reference images
    class Similarity:
        def __init__(self, similarity2References: np.array, referenceLabels: List[int]):
            self.similarity2References = similarity2References
            self.referenceLabels = referenceLabels
        def size(self):
            return len(self.referenceLabels)

    references = referenceDS.queries()
    referenceLabels = referenceDS.queryLabels()
    similarities: List[Similarity] = []
    for query in queryDS.queries():
        queries = np.repeat(np.expand_dims(query, axis=0), references.shape[0], axis=0)
        similarities.append(Similarity(model.predict((queries, references)), referenceLabels))

    ### For each query image find the K most reference images
    minHeaps: List[List[Tuple[np.float32, int, int]]] = []
    for similarity in similarities:
        minHeaps.append([])
        minHeap = minHeaps[-1]
        for i in range(min(K, similarity.size())):
            heapq.heappush(minHeap, (similarity.similarity2References[i], i, similarity.referenceLabels[i]))
        for i in range(K, similarity.size()):
            top = heapq.heappop(minHeap)
            if similarity.similarity2References[i] > top[0]:
                heapq.heappush(minHeap, (similarity.similarity2References[i], i, similarity.referenceLabels[i]))
            else:
                heapq.heappush(minHeap, top)

    ### List of referene image file paths most similar to each query
    referencesSimilarPaths: List[SimilarityPath] = []
    referencesFilePaths = referenceDS.queryFilePaths()
    referencesLabels = referenceDS.queryLabels()
    for i in range(len(queryDS.queries())):
        queryPath = queryDS.queryFilePaths()[i]
        queryLabel = queryDS.queryLabels()[i]
        minHeap = minHeaps[i]
        referencePaths: List[str] = []
        referenceLabels: List[int] = []
        similarityValues: List[np.float32] = []
        while minHeap:
            top = heapq.heappop(minHeap)
            similarityValues.append(top[0])
            referenceIndex = top[1]
            referencePaths.append(referencesFilePaths[referenceIndex])
            referenceLabels.append(referencesLabels[referenceIndex])
        referencesSimilarPaths.append(SimilarityPath(queryPath, queryLabel, referencePaths, referenceLabels,
                                                     similarityValues))
    return referencesSimilarPaths

class MyTestCase(unittest.TestCase):
    def test_find_k_similar(self):
        referencesSimilarPaths = findKSimilar(10, referencePath = SMALL_DATA_PATH)
        pass


if __name__ == '__main__':
    unittest.main()
