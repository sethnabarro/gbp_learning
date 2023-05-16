# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from unittest import TestCase

from core.edge import Edge
from core.factors import PairwiseSmoothingFactorDecomp
from core.inference.base import ConvLayer
from core.variables import WeightVariable, Variable
from core.utils.utils import get_n_strides, patchify_image


class TestPairwiseFactor(TestCase):
    def test_pairwise_factor(self):
        """Linear regression example"""
        tf.random.set_seed(8)
        np.random.seed(8)
        n_data = 10
        img_dims = (20, 20, 3)
        out_std = 0.1
        n_out_dim = 1
        mom = 0.
        ks = 3

        # Sample data
        x_obs = tf.random.normal([n_data] + list(img_dims))   # Inputs

        # Create variables
        input_vars = Variable(mu_init=x_obs, sigma_init=1.)

        # Create edges
        # Add zero mean prior to weights via initialisation of var to fac msg
        prior_sigma = 1.
        input_obs_sigma = 0.1
        pw_smooth_sigma = 0.01

        nx, ny = get_n_strides(img_in=x_obs,
                               stride=1,
                               ksize=ks)
        in_pw_edge_shp = [n_data, nx, ny, img_dims[-1], 1, ks ** 2, 2]
        print(patchify_image(x_obs, ksize_x=ks, stride=1)[..., None, :].shape)
        x_obs_patch = patchify_image(x_obs, ksize_x=ks, stride=1)[..., None, :]
        x_obs_patch = tf.concat([x_obs_patch[..., None], tf.repeat(x_obs_patch[..., int(ks ** 2 / 2):int(ks ** 2 / 2) + 1], repeats=ks ** 2, axis=-1)[..., None]], axis=-1)
        input_pw_fac_edge = Edge(var_to_fac_eta=x_obs_patch / input_obs_sigma ** 2.,
                          var_to_fac_Lambda=tf.ones_like(x_obs_patch) / input_obs_sigma ** 2.,
                          fac_to_var_eta=tf.zeros(in_pw_edge_shp),
                          fac_to_var_Lambda=tf.zeros(in_pw_edge_shp),
                                 momentum=mom)

        in_edge_shp = [n_data] + list(img_dims)
        input_fac_edge = Edge(var_to_fac_eta=tf.zeros(in_edge_shp),
                          var_to_fac_Lambda=tf.zeros(in_edge_shp),
                          fac_to_var_eta=x_obs / input_obs_sigma ** 2.,
                          fac_to_var_Lambda=tf.ones_like(x_obs) / input_obs_sigma ** 2.,
                              momentum=mom)

        # Initialise factor
        pw_factor = \
            PairwiseSmoothingFactorDecomp(sigma=pw_smooth_sigma,
                                          input_var_edges=input_pw_fac_edge,
                                          n_filters=None,
                                          init_lin_point=[x_obs])

        for i in range(20):
            if i % 5 == 0:
                print(np.mean(np.std(np.reshape(input_vars.mu, [n_data, -1, img_dims[-1]]), axis=1)), 'std')
                plt.imshow(input_vars.mu[0])
                plt.show()

            pw_factor.relinearise(i, [input_vars.mu])
            pw_factor.update_outgoing_messages(None)

            print(input_pw_fac_edge.fac_to_var_eta[0, 3, 3, 0])
            def combine_centpix_msgs(msg):
                msg_cent = tf.reduce_sum(msg[..., 1:], axis=-2)[..., None, :]
                msg = tf.where((tf.range(ks ** 2) == int(ks ** 2 / 2))[..., None], msg[..., :1] + msg_cent, msg[..., :1])
                return msg
            eta_centpix_combined = combine_centpix_msgs(input_pw_fac_edge.fac_to_var_eta)
            Lambda_centpix_combined = combine_centpix_msgs(input_pw_fac_edge.fac_to_var_Lambda)

            pw_fac_to_var_eta = ConvLayer.depatchify_static(eta_centpix_combined, k_size=ks, stride=1)
            pw_fac_to_var_Lambda = ConvLayer.depatchify_static(Lambda_centpix_combined, k_size=ks, stride=1)

            input_vars.eta = input_fac_edge.fac_to_var_eta + pw_fac_to_var_eta
            input_vars.Lambda = input_fac_edge.fac_to_var_Lambda + pw_fac_to_var_Lambda

            def add_centpix_msg(msg_patch):
                centpix_bc = tf.broadcast_to(msg_patch[..., int(ks ** 2 / 2):int(ks ** 2 / 2) + 1], msg_patch.shape)
                return tf.concat([msg_patch[..., None], centpix_bc[..., None]], axis=-1)

            # print(patchify_image(input_fac_edge.fac_to_var_eta, ksize_x=ks, stride=1).shape, in_pw_edge_shp[:-1])
            eta_patch = tf.broadcast_to(patchify_image(input_fac_edge.fac_to_var_eta, ksize_x=ks, stride=1)[..., None, :], in_pw_edge_shp[:-1])
            Lambda_patch = tf.broadcast_to(patchify_image(input_fac_edge.fac_to_var_Lambda, ksize_x=ks, stride=1)[..., None, :], in_pw_edge_shp[:-1])
            input_pw_fac_edge.var_to_fac_eta = patchify_image(input_vars.eta, ksize_x=ks, stride=1)[..., None, :, None] - add_centpix_msg(eta_patch)
            input_pw_fac_edge.var_to_fac_Lambda = patchify_image(input_vars.Lambda, ksize_x=ks, stride=1)[..., None, :, None] - add_centpix_msg(Lambda_patch)

