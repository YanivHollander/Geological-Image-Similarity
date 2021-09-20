import unittest
import numpy as np
import matplotlib.image as mpimg
import os
from Globals import DATA_PATH, SMALL_DATA_PATH
from typing import List, Dict, Tuple
from scipy import ndimage

def loadImage(fullPath: str, display: bool = False)->np.ndarray:
    """
    Load an image and normalizes for training/display
    :param fullPath: Image full path
    :param display: Normalize for display
    :return:
    """
    img = mpimg.imread(fullPath)
    if not display:
        img = img.astype(np.float32) / 255.
    return img

class GeologicalDataset():
    """
    A class to menage the dataset of geological images
    """
    class Samples:
        """
        Holds Numpy arrays of query and reference images , and their joint labels. This data structure scales to the
        input needed for the Siamese network. The query and reference images array are of the same size. Label for each
        pair is 1 if both belong to the same class, and 0 if each belongs to a different
        """
        def __init__(self, queries: List[np.array], references: List[np.array], labels: List[np.float32]):
            self.queries = np.array(queries)
            self.references = np.array(references)
            self.labels = np.array(labels)

    def __init__(self, path: str, augmentRatio = 0.1):
        """
        Initializes GeologicalDataset
        :param path: Path to dataset of queries and reference images
        :param augmentRatio: Ratio of data to augment for training
        """
        self.__filepaths: List[str] = []    # List of full paths to images
        self.__labels: List[int] = []       # Labels for each image (rock type)
        self.__labelType: Dict[str, int] = {'andesite': 0, 'gneiss': 1, 'marble': 2, 'quartzite': 3, 'rhyolite': 4,
                                            'schist': 5}
        self.__samples = GeologicalDataset.Samples([], [], [])

        # Dataset augmentation
        self.__augmentRatio = augmentRatio

        # Extracting all image filenames and labels from data path
        pathIter = os.walk(path)
        for root, directories, files in pathIter:
            for file in files:
                filename, fileExtension = os.path.splitext(file)
                if fileExtension != '.jpg':
                    continue
                self.__filepaths.append(os.path.join(root, file))
                self.__labels.append(self.__labelType[os.path.split(root)[1]])  # Append image directory as rock type

    def initDataset(self, training = True) -> None:
        """
        Initializes the dataset by loading all query and reference images defined by the list of predefined paths. For
        training references images are also loaded. For each query, a random image is loaded as its reference. A label
        for the pair is set based on whether the two images belong to the same class or not
        :param training: Indicates whether dataset is initialized for training (for inference there is not need to
                         assign reference images)
        """
        # Loading images
        images = []
        for filepath in self.__filepaths:
            images.append(loadImage(filepath))

        # Creating pairs of queries and reference images, and the proper positive/negative labels
        queries = []
        references = []
        labels = []
        for idx, image in enumerate(images):

            # Adding to dataset
            queries.append(image)
            if training:
                idxRefernce = np.random.randint(0, len(images) - 1)
                references.append(images[idxRefernce])
                labels.append(np.float32(1.) if self.__labels[idx] is not self.__labels[idxRefernce] else
                              np.float32(0.))

        # Augment training dataset (only for training)
        if training and self.__augmentRatio > 0:
            self.__augment(queries, references, labels)

        # Initialize training sample dataset
        self.__samples = GeologicalDataset.Samples(queries, references, labels)

    def __augment(self, queries: List[np.array], references: List[np.array], labels: List[np.float32]) -> None:
        """
        Augments images to training dataset
        :param queries: Basic set of query images
        :param references: Basic set of reference images
        :param labels: Labels of basic set pairs
        """
        queriesAugment = []
        referencesAugment = []
        labelsAugment = []
        for query, reference, label in zip(queries, references, labels):
            if np.random.rand(1) >= self.__augmentRatio:    # Skip samples not augmented
                continue

            # Augmentation type - flip a coin between flipping the image and rotating it
            if np.random.rand(1) >= 0.5:    # Flip the image
                queriesAugment.append(np.flip(query, axis=0))
            else:   # Rotate the image
                queriesAugment.append(ndimage.rotate(query, 45, mode='mirror', reshape=False))
            referencesAugment.append(reference)
            labelsAugment.append(label)

        # Extend basic query, reference, and lavel set by the augmented samples
        queries.extend(queriesAugment)
        references.extend(referencesAugment)
        labels.extend(labelsAugment)

    def imageShape(self) -> Tuple[int, int, int]:
        """
        Returns a sample image size
        :return: Tuple of H-W-C
        """
        shape = self.__samples.queries.shape
        return shape[1], shape[2], shape[3]

    def queryFilePaths(self) -> List[str]:
        """
        Returns file paths of dataset images
        :return: List of file paths
        """
        return self.__filepaths

    def queryLabels(self) -> List[int]:
        """
        Returns the labels of query images (used for inference)
        :return: List of integers (1-6)
        """
        return self.__labels

    def queries(self) -> np.array:
        """
        Returns query image in the dataset
        :return: Numpy array of images (batch axis = 0)
        """
        return self.__samples.queries

    def references(self) -> np.array:
        """
        Returns reference image in the dataset
        :return: Numpy array of images (batch axis = 0)
        """
        return self.__samples.references

    def labels(self) -> np.array:
        """
        Returns labels of query-reference pairs (1 for a pair that belong to the same class; 0 otherwise)
        :return: Numpy array of labels (batch axis = 0)
        """
        return self.__samples.labels

class MyTestCase(unittest.TestCase):
    def test_load_image(self):
        loadImage(os.path.join(DATA_PATH, 'andesite/0A5NL.jpg'))

    def test_display_image(self):
        from matplotlib import pyplot as plt
        img = loadImage(os.path.join(DATA_PATH, 'andesite/0A5NL.jpg'), display=True)
        plt.figure()
        plt.imshow(img)
        plt.show()

    def test_dataset(self):
        ds = GeologicalDataset(SMALL_DATA_PATH, augmentRatio=0.5)
        ds.initDataset()

if __name__ == '__main__':
    unittest.main()
