from abc import ABC, abstractmethod
from scipy import spatial

import numpy as np


class Vector(ABC):
    """
    Abstract class used to represent a vector. This is useful for similarity computation
    which can be customized thanks to the similarity abstract method.

    Args:
        value: value to store inside of the Vector (a numpy array is used in our scenarios)
    """
    def __init__(self, value):
        self.__value = value

    @property
    def value(self):
        return self.__value

    @abstractmethod
    def similarity(self, other):
        raise NotImplementedError


class DenseVector(Vector):
    def similarity(self, other):
        """
        Computes cosine similarity between the stored value and another value passed as argument

        Args:
            other: value to compare with the value saved in the object attributes (a numpy array
                is used in our scenarios)
        """
        return 1 - spatial.distance.cosine(self.value, other.value)


class Similarity(ABC):
    """
    Abstract class for the various types of similarity, these types of similarity
    can be used, for example, in the CentroidVectorRecommender
    """
    def __init__(self):
        pass

    @abstractmethod
    def perform(self, v1: np.ndarray, v2: np.ndarray):
        """
        Calculates the similarity between v1 and v2

        Args:
            v1(np.ndarray): numpy array
            v2(np.ndarray): numpy array
        """
        raise NotImplementedError


class CosineSimilarity(Similarity):
    """
    Computes cosine similarity of given numpy arrays
    """
    def __init__(self):
        super().__init__()

    def perform(self, v1: np.ndarray, v2: np.ndarray):
        """
        Calculates the similarity between two numpy arrays.
        Instantiates two Vector objects and passes one of the two arrays to each one of them.
        Then the similarity is computed using the similarity method of the Vector object.

        Args:
            v1(np.ndarray): numpy array
            v2(np.ndarray): numpy array
        """
        return DenseVector(v1).similarity(DenseVector(v2))
