# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from unittest import TestCase

from core.edge import Edge
from core.factors import DenseFactor
from core.variables import WeightVariable, Variable


class TestDenseFactor(TestCase):
    def test_dense_factor(self):
        """Linear regression example"""
        tf.random.set_seed(8)
        np.random.seed(8)
        n_data = 10
        n_dim = 1
        out_std = 0.1
        n_out_dim = 1
        nonlin = ('leaky_relu', tf.nn.leaky_relu)

        # Sample data
        x_obs = tf.random.normal([n_data, n_dim])   # Inputs
        w = tf.random.normal([n_dim, n_out_dim])        # Weights
        y_obs = nonlin[1](tf.einsum('ab,ca->cb', w, x_obs)) + tf.random.normal([n_data, 1]) * out_std  # Outputs

        # Create variables
        weight_vars = WeightVariable(mu_init=tf.zeros([n_dim, n_out_dim]), sigma_init=1.)
        input_vars = Variable(mu_init=tf.zeros([n_data, n_dim]), sigma_init=1.)
        output_vars = Variable(mu_init=tf.zeros([n_data, n_out_dim]), sigma_init=1.)

        # Create edges
        # Add zero mean prior to weights via initialisation of var to fac msg
        prior_sigma = 1.
        input_obs_sigma = 0.01
        output_obs_sigma = 0.01  #out_std
        weight_edge = Edge(var_to_fac_eta=tf.zeros([n_data, n_out_dim, n_dim]),
                           var_to_fac_Lambda=tf.ones([n_data, n_out_dim, n_dim]) / prior_sigma ** 2.,
                           fac_to_var_eta=tf.zeros([n_data, n_out_dim, n_dim]),
                           fac_to_var_Lambda=tf.zeros([n_data, n_out_dim, n_dim]))

        input_edge = Edge(var_to_fac_eta=tf.broadcast_to(x_obs[:, None], [n_data, n_out_dim, n_dim]) / input_obs_sigma ** 2.,
                          var_to_fac_Lambda=tf.ones([n_data, n_out_dim, n_dim]) / input_obs_sigma ** 2.,
                          fac_to_var_eta=tf.zeros([n_data, n_out_dim, n_dim]),
                          fac_to_var_Lambda=tf.zeros([n_data, n_out_dim, n_dim]))
        output_edge = Edge(var_to_fac_eta=y_obs / output_obs_sigma ** 2.,
                           var_to_fac_Lambda=tf.ones([n_data, n_out_dim]) / output_obs_sigma ** 2.,
                           fac_to_var_eta=tf.zeros([n_data, n_out_dim]),
                           fac_to_var_Lambda=tf.zeros([n_data, n_out_dim]))

        # Initialise factor
        dense_factor = \
            DenseFactor(sigma=0.1,
                        input_var_edges=input_edge,
                        weight_var_edges=weight_edge,
                        output_var_edges=output_edge,
                        init_lin_point=[input_vars.mu, weight_vars.mu, output_vars.mu],
                        relin_freq=1,
                        nonlin=nonlin[0])

        for i in range(20):
            dense_factor.relinearise(i, [input_vars.mu, weight_vars.mu, output_vars.mu])
            dense_factor.update_outgoing_messages(None)
            weight_prior_eta, weight_prior_Lambda = weight_edge.var_to_fac_eta, weight_edge.var_to_fac_Lambda
            # print(tf.reduce_sum(weight_edge.fac_to_var_Lambda, axis=0), 'test')
            weight_vars.eta = tf.transpose(weight_prior_eta[0] + tf.reduce_sum(weight_edge.fac_to_var_eta, axis=0))
            weight_vars.Lambda = tf.transpose(weight_prior_Lambda[0] + tf.reduce_sum(weight_edge.fac_to_var_Lambda, axis=0))

            input_vars.eta = tf.reduce_sum(input_edge.fac_to_var_eta + input_edge.var_to_fac_eta, axis=1)
            input_vars.Lambda = tf.reduce_sum(input_edge.fac_to_var_Lambda + input_edge.var_to_fac_Lambda, axis=1)

            output_vars.eta = output_edge.fac_to_var_eta + output_edge.var_to_fac_eta
            output_vars.Lambda = output_edge.fac_to_var_Lambda + output_edge.var_to_fac_Lambda

        if n_dim == n_out_dim and n_dim == 1:
            x = np.linspace(-3., 3., 50)
            y = nonlin[1](w[0, 0] * x)
            weight_samples = weight_vars.mu[0,0] + weight_vars.sigma[0, 0] * np.random.normal(size=(100,))[:, None]
            y_pred_samples = nonlin[1](weight_samples * x[None])
            y_pred_mean = np.mean(y_pred_samples, axis=0)
            y_pred_std = np.std(y_pred_samples, axis=0)
            plt.plot(x, y, color='red', label='Ground truth')
            plt.plot(x, y_pred_mean, color='blue', label='Model prediction')
            plt.fill_between(x, y_pred_mean - y_pred_std, y_pred_mean + y_pred_std, color='blue', alpha=0.2, label='+-1std')
            plt.scatter(x_obs[:, 0], y_obs[:, 0], color='red', label='Observation')
            plt.legend()
            plt.show()
