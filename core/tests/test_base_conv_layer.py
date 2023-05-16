# coding=utf-8
import tensorflow as tf

from core.inference.base import ConvLayer


class TestConvLayer(tf.test.TestCase):
    def test_depatchify(self):
        """
        Sample random image(s), make into patches, then reconstruct.
        Check reconstruction matches original.
        """
        img_shp = (2, 50, 50, 3)
        img = tf.random.normal(img_shp)
        ksize = 3

        # Make patches
        img_patches = tf.image.extract_patches(img,
                                               sizes=(1, ksize, ksize, 1),
                                               strides=(1, 1, 1, 1),
                                               rates=(1, 1, 1, 1),
                                               padding='VALID')
        img_patches = tf.reshape(img_patches, img_patches.shape[:3].as_list() + [-1, img_shp[-1]])
        img_patches = tf.transpose(img_patches, (0, 1, 2, 4, 3))

        # reconstruct
        img_recon = ConvLayer.depatchify_static(img_patches, k_size=ksize)

        # End up duplicating pixels not near the edge - account for edge effects
        edge_eff_x = tf.minimum(tf.cast(tf.minimum(tf.range(1, img_shp[1] + 1),
                                                   tf.abs(tf.range(1, img_shp[1] + 1) - img_shp[1] - 1)),
                                        tf.float32), ksize)
        edge_eff_y = tf.minimum(tf.cast(tf.minimum(tf.range(1, img_shp[2] + 1),
                                                   tf.abs(tf.range(1, img_shp[2] + 1) - img_shp[2] - 1)),
                                        tf.float32), ksize)
        img_recon /= edge_eff_x[None, :, None, None]
        img_recon /= edge_eff_y[None, None, :, None]

        self.assertAllClose(img_recon, img)
