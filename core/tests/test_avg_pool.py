import numpy as np
import tensorflow as tf
from unittest import TestCase

from experiments.utils.graph_setup_utils import init_avg_pool_layer
from core.variables import PixelVariable
from core.edge import Edge
from core.factors import UnaryFactor
from core.utils.utils import dotdict

# with_output_layer = False


class TestAvgPoolFactor(TestCase):
    def test_avg_pool_factor(self, with_output_layer: bool = True):
        """with_output_layer adds a layer after the avg pool to check that the
        pooling layer works when receiving messages from both below and above"""
        img = np.reshape(np.linspace(0.1, 3.6, 36), (1, 6, 6, 1)).astype(np.float32)
        e = Edge(var_to_fac_eta=tf.zeros((1, 6, 6, 1)),
                 fac_to_var_eta=tf.zeros((1, 6, 6, 1)),
                 var_to_fac_Lambda=tf.zeros((1, 6, 6, 1)),
                 fac_to_var_Lambda=tf.zeros((1, 6, 6, 1)))
        pix = PixelVariable(tf.convert_to_tensor(img), sigma_init=1.)
        obs = UnaryFactor(sigma=1., obs=img, var_edges=e)

        if with_output_layer:
            out_layer_value = 2.
            img_out = tf.ones((1, 2, 2, 1)) * out_layer_value
            e_out = Edge(var_to_fac_eta=tf.zeros((1, 2, 2, 1)),
                        fac_to_var_eta=tf.zeros((1, 2, 2, 1)),
                        var_to_fac_Lambda=tf.zeros((1, 2, 2, 1)),
                        fac_to_var_Lambda=tf.zeros((1, 2, 2, 1)))
            obsout = UnaryFactor(sigma=1., obs=img_out, var_edges=e_out)

        avg_pool_layer = init_avg_pool_layer(input_vars=pix,
                                             momentum=0.,
                                             avg_pool_factor_config=dotdict(sigma=1., N_rob=None, rob_type=None, kmult=1., ksize=3),
                                             coeff_prior_factor_config=dotdict(sigma=100.))
        obs_eta, obs_Lambda = obs.get_eta_Lambda()
        avg_pool_layer.input_vars.eta = obs_eta
        avg_pool_layer.input_vars.Lambda = obs_Lambda
        # avg_pool_layer.coeff_vars.Lambda = tf.ones_like(avg_pool_layer.coeff_vars.Lambda) * 0.001   # Weak zero-mean prior

        if with_output_layer:
            obs_out_eta, obs_out_Lambda = obsout.get_eta_Lambda()

        true_mean = tf.nn.avg_pool2d(img, 3, 3, 'VALID')
        for i in range(3):
            avg_pool_layer.intra_layer_inference_iter(i)
            pool_eta, pool_Lambda = avg_pool_layer._update_input_marginals(return_eta_Lambda=True)
            coeff_eta, coeff_Lambda = avg_pool_layer._update_coeff_marginals(return_eta_Lambda=True)
            pix.eta = obs_eta + pool_eta
            pix.Lambda = obs_Lambda + pool_Lambda
            avg_pool_layer.input_vars.eta = obs_eta + pool_eta
            avg_pool_layer.input_vars.Lambda = obs_Lambda + pool_Lambda
            avg_pool_layer.coeff_vars.eta = coeff_eta
            avg_pool_layer.coeff_vars.Lambda = coeff_Lambda #+ tf.ones_like(coeff_Lambda) * 0.001
            if with_output_layer:
                avg_pool_layer.coeff_vars.eta += obs_out_eta
                avg_pool_layer.coeff_vars.Lambda += obs_out_Lambda
            avg_pool_mean = avg_pool_layer.coeff_vars.eta / avg_pool_layer.coeff_vars.Lambda
        print(f'True mean: {true_mean}, average pool mean: {avg_pool_mean}')

        if with_output_layer:
            # Inferred mean should be somewhere between what the avg
            # pool layer says and what layer above says
            elems_between = \
                tf.logical_or(tf.logical_and(true_mean < avg_pool_mean, avg_pool_mean < out_layer_value),
                              tf.logical_and(out_layer_value < avg_pool_mean, avg_pool_mean < true_mean))
            self.assertTrue(tf.reduce_all(elems_between))

        else:
            self.assertLess(tf.reduce_sum(tf.abs(avg_pool_mean - true_mean)), 1e-2)

    def test_avg_pool_factor2(self):
        self.test_avg_pool_factor(False)

