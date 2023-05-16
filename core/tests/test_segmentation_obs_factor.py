# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from core.factors import SoftmaxSegmentationObservationFactor


class TestSoftmaxSegmentationObservationFactor(tf.test.TestCase):
    def setUp(self):
        stripe = np.linspace(0, 14, 14) // 5 % 2
        self.img = np.repeat(stripe[None], 15, axis=0)[None, ..., None]
        self.sparse_labels = np.array([[0, 0, 0, 0, 0, 0],
                                       [2, 2, 13, 12, 6, 8],
                                       [2, 10, 1, 13, 5, 10],
                                       [0, 0, 0, 0, 1, 1]])
        self.dense_labels = self.img.copy()

    def test_sparse_segmentation(self):
        stripe = np.linspace(0, 14, 14) // 5 % 2
        img = np.repeat(stripe[None], 15, axis=0)[None, ..., None]
        plt.imshow(img[0, ..., 0])
        plt.show()
        print(img)


if __name__ == '__main__':
    TestSoftmaxSegmentationObservationFactor().run()