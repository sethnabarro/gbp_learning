# coding=utf-8
import numpy as np
import tensorflow as tf

from core.variables.base import Variable


class SegmentationVariable(Variable):
    def __init__(self, mu_init, sigma_init, img_shape):
        is_dense = mu_init.shape[:-1] == img_shape[:-1]
        if not is_dense:
            mu_init = self.make_dense(mu_init, img_shape)
        super(SegmentationVariable, self).__init__(mu_init, sigma_init)
        self.fix_attr_fields()

    @staticmethod
    def make_dense(sparse, img_shape):
        scale = 1.
        n_class_dims = len(np.unique(sparse[:, -1]))
        n_seg_classes = tf.unique(sparse[:, -1]).y.shape[0]
        dense = tf.zeros(img_shape[:-1] + (n_seg_classes,))
        pixel_idx = tf.repeat(sparse[:, :-1], n_class_dims, axis=0)
        pixel_idx_all_class = tf.concat((pixel_idx, tf.repeat(tf.range(n_class_dims, dtype=pixel_idx.dtype), sparse.shape[0])[:, None]), axis=-1)
        dense = tf.tensor_scatter_nd_update(dense, pixel_idx_all_class, tf.broadcast_to([-scale], [pixel_idx_all_class.shape[0]]))
        dense = tf.tensor_scatter_nd_update(dense, sparse, tf.broadcast_to([scale], [sparse.shape[0]]))
        return dense

    @staticmethod
    def make_sparse(dense):
        n_classes = dense.shape[-1]
        sparse = None
        for cl in range(n_classes):
            b_x_y = tf.where(dense[..., cl] == 1)
            if b_x_y.shape[0] > 0:
                b_x_y_c = tf.concat([b_x_y, [[cl]] * b_x_y.shape[0]], axis=-1)
                if cl == 0:
                    sparse = b_x_y_c
                else:
                    sparse = tf.concat([sparse, b_x_y_c], axis=0)
        if sparse is None:
            sparse = tf.zeros((0, 4), tf.int64)
        return sparse
