# coding=utf-8
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from core.utils.utils import patchify_image, kmeans

mpl.rcParams['image.cmap'] = 'gray'
imshow = partial(plt.imshow, interpolation='none')


def get_ground_truth_filters(img, n_filters, kernsize,
                             additive_factor=False, plot_filters=False,
                             add_flat_filter=False, seed=123):

    patches = patchify_image(img.astype(np.float32))
    patches_rs = tf.reshape(patches, [-1, kernsize ** 2])
    means = kmeans(patches_rs, additive_factor, k=n_filters, seed=seed)

    if add_flat_filter:
        # Add flat filter
        means = tf.tensor_scatter_nd_update(means, [[n_filters - 1]], np.zeros((1, kernsize ** 2)))

    if additive_factor:
        if plot_filters:
            for m in means:
                imshow(np.reshape(m, (3, 3)))
                plt.colorbar()
                plt.show()

        return tf.reshape(tf.transpose(means, (1, 0)), (kernsize, kernsize, 1, n_filters))

    # find orthogonal filters
    filters_gt = np.ones((n_filters, kernsize ** 2))
    for i, m in enumerate(means):
        filters_gt[i, i] -= 1. + np.sum([m[j] for j in range(len(m)) if j != i]) / m[i]
    filters_gt /= tf.linalg.norm(filters_gt, axis=-1, keepdims=True)
    filters_gt = tf.cast(filters_gt, dtype=tf.float32)

    if plot_filters:
        for m in filters_gt:
            imshow(np.reshape(m, (3, 3)))
            plt.show()

    return tf.reshape(tf.transpose(filters_gt, (1, 0)), (3, 3, 1, n_filters))
