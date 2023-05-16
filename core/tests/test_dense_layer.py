# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from unittest import TestCase

from core.edge import Edge
from core.factors import DenseFactor, UnaryFactor
from core.factors.utils import get_nonlin_fns, get_nonlin_fn_grads
from core.inference.gbp.layers import GBPDenseLayer, GBPLayerEmpty
from core.variables import CoeffVariable, WeightVariable, Variable, BiasVariable


def gbp_linear_regression(indims=(20,), n_out_dim=1, noiseless_x=True, with_bias=False):
    hasloops = (n_out_dim > 1) or np.prod(indims) > 1 or with_bias
    edge_mom = 0. if noiseless_x and not hasloops else 0.9
    n_data = 100
    out_std = 0.
    nonlin_name = 'none'
    nonlin_fn = get_nonlin_fns()[nonlin_name]

    relin_freq = 1 if noiseless_x and not hasloops else 1
    n_iters = 2 if noiseless_x and not hasloops else 60

    # Sample data
    x_obs = tf.random.normal([n_data] + list(indims))  # Inputs
    w = tf.random.normal([np.prod(indims), n_out_dim]) * 1.  # Weights
    b = tf.random.normal([n_out_dim]) if with_bias else tf.zeros([n_out_dim])  # bias
    y_obs = nonlin_fn(tf.einsum('ab,ca->cb', w, np.reshape(x_obs, (n_data, np.prod(indims)))) + b) + tf.random.normal(
        [n_data, n_out_dim]) * out_std  # Outputs

    # Create variables
    weight_vars = WeightVariable(mu_init=tf.zeros([np.prod(indims), n_out_dim]), sigma_init=1.)
    input_vars = Variable(mu_init=tf.zeros([n_data] + list(indims)), sigma_init=1.)
    output_vars = Variable(mu_init=tf.zeros([n_data, n_out_dim]), sigma_init=1.)
    bias_vars = BiasVariable(mu_init=tf.zeros([n_out_dim]), sigma_init=1.)

    # Create edges
    # Add zero mean prior to weights and obs factors to outputs and inputs
    weight_prior_sigma = 1.
    bias_prior_sigma = 1.
    input_obs_sigma = 0.01
    output_obs_sigma = 0.05  # out_std

    weight_prior_edge = Edge(fac_to_var_eta=tf.zeros_like(weight_vars.mu),
                             fac_to_var_Lambda=tf.ones_like(weight_vars.mu) / weight_prior_sigma ** 2,
                             var_to_fac_eta=tf.zeros_like(weight_vars.mu),
                             var_to_fac_Lambda=tf.zeros_like(weight_vars.mu))
    weight_prior_factor = UnaryFactor(sigma=weight_prior_sigma,
                                      obs=0.,
                                      var_edges=weight_prior_edge)

    if with_bias:
        bias_prior_edge = Edge(fac_to_var_eta=tf.zeros_like(bias_vars.mu),
                               fac_to_var_Lambda=tf.ones_like(bias_vars.mu) / bias_prior_sigma ** 2.,
                               var_to_fac_eta=tf.zeros_like(bias_vars.mu),
                               var_to_fac_Lambda=tf.zeros_like(bias_vars.mu))
        bias_prior_factor = UnaryFactor(sigma=bias_prior_sigma,
                                        obs=0.,
                                        var_edges=bias_prior_edge)
    else:
        bias_prior_factor = None

    output_obs_edge = Edge(var_to_fac_eta=tf.zeros_like(y_obs),
                           var_to_fac_Lambda=tf.zeros_like(y_obs),
                           fac_to_var_eta=y_obs / output_obs_sigma ** 2.,
                           fac_to_var_Lambda=tf.ones([n_data, n_out_dim]) / output_obs_sigma ** 2.)
    output_obs_factor = UnaryFactor(sigma=output_obs_sigma,
                                    obs=y_obs,
                                    var_edges=output_obs_edge)

    if not noiseless_x:
        input_obs_edge = Edge(fac_to_var_eta=x_obs / input_obs_sigma ** 2.,
                              fac_to_var_Lambda=tf.ones([n_data] + list(indims)) / input_obs_sigma ** 2.,
                              var_to_fac_eta=tf.zeros([n_data] + list(indims)),
                              var_to_fac_Lambda=tf.zeros([n_data] + list(indims)))
        input_obs_factor = UnaryFactor(sigma=input_obs_sigma,
                                       obs=x_obs,
                                       var_edges=input_obs_edge)

    # Initialise factor
    if noiseless_x:
        input_edge = None
    else:
        input_edge = Edge(var_to_fac_eta=tf.broadcast_to(np.reshape(x_obs, [n_data, 1, np.prod(indims)]),
                                                         [n_data, n_out_dim, np.prod(indims)]) / input_obs_sigma ** 2.,
                          var_to_fac_Lambda=tf.ones([n_data, n_out_dim, np.prod(indims)]) / input_obs_sigma ** 2.,
                          fac_to_var_eta=tf.zeros([n_data, n_out_dim, np.prod(indims)]),
                          fac_to_var_Lambda=tf.zeros([n_data, n_out_dim, np.prod(indims)]), momentum=edge_mom)
    weight_edge = Edge(var_to_fac_eta=tf.zeros([n_data, n_out_dim, np.prod(indims)]),
                       var_to_fac_Lambda=tf.ones([n_data, n_out_dim, np.prod(indims)]) / weight_prior_sigma ** 2.,
                       fac_to_var_eta=tf.zeros([n_data, n_out_dim, np.prod(indims)]),
                       fac_to_var_Lambda=tf.zeros([n_data, n_out_dim, np.prod(indims)]), momentum=edge_mom)
    output_edge = Edge(var_to_fac_eta=y_obs / output_obs_sigma ** 2.,
                       var_to_fac_Lambda=tf.ones([n_data, n_out_dim]) / output_obs_sigma ** 2.,
                       fac_to_var_eta=tf.zeros([n_data, n_out_dim]),
                       fac_to_var_Lambda=tf.zeros([n_data, n_out_dim]), momentum=edge_mom)
    if with_bias:
        bias_edge = Edge(var_to_fac_eta=tf.zeros([n_data, n_out_dim]),
                         var_to_fac_Lambda=tf.ones([n_data, n_out_dim]) / 0.1 ** 2.,
                         fac_to_var_eta=tf.zeros([n_data, n_out_dim]),
                         fac_to_var_Lambda=tf.zeros([n_data, n_out_dim]), momentum=edge_mom)
    else:
        bias_edge = None

    init_vars = [] if noiseless_x else [input_vars.mu]
    init_vars += [weight_vars.mu]
    init_vars += [bias_vars.mu] if with_bias else []
    init_vars += [output_vars.mu]
    dense_factor = \
        DenseFactor(sigma=0.1,
                    input_var_edges=input_edge,
                    noiseless_input=noiseless_x,
                    input_obs=x_obs,
                    weight_var_edges=weight_edge,
                    output_var_edges=output_edge,
                    bias_var_edges=bias_edge,
                    init_lin_point=init_vars,
                    relin_freq=relin_freq,
                    nonlin=nonlin_name)

    # Initialise layer
    dense_layer = \
        GBPDenseLayer(dense_factor=dense_factor,
                      coeff_vars=output_vars,
                      coeff_prior_factor=output_obs_factor,
                      bias_prior_factor=bias_prior_factor,
                      input_vars=None if noiseless_x else input_vars,
                      weight_vars=weight_vars,
                      bias_vars=bias_vars if with_bias else None,
                      weight_prior_factor=weight_prior_factor)

    if not noiseless_x:
        # Initialise layer before (with input observation factor)
        in_layer = GBPLayerEmpty(coeff_vars=input_vars,
                                 input_vars=None)

        def input_obs_marg_update(return_eta_Lambda=True):
            return input_obs_factor.get_eta_Lambda()

        in_layer._update_coeff_marginals = input_obs_marg_update

        # Link the two layers
        dense_layer.link_layer_before(in_layer)

    mses = []
    mses_pred = []
    mses_bias = []
    for i in range(n_iters):
        mse = tf.reduce_mean((weight_vars.mu - w) ** 2.)
        mse_pred = tf.reduce_mean((y_obs - nonlin_fn(
            tf.einsum('ab,ca->cb', weight_vars.mu, np.reshape(x_obs, (n_data, np.prod(indims)))) + bias_vars.mu)) ** 2.)
        mses.append(mse)
        mses_pred.append(mse_pred)
        if with_bias:
            mse_bias = tf.reduce_mean((bias_vars.mu - b) ** 2.)
            mses_bias.append(mse_bias)
        dense_layer.intra_layer_inference_iter(i)
    mse = tf.reduce_mean((weight_vars.mu - w) ** 2.)
    if with_bias:
        mse_bias = tf.reduce_mean((bias_vars.mu - b) ** 2.)
    mse_pred = tf.reduce_mean((y_obs - nonlin_fn(
        tf.einsum('ab,ca->cb', weight_vars.mu, np.reshape(x_obs, (n_data, np.prod(indims)))) + bias_vars.mu)) ** 2.)
    mses.append(mse)
    mses_pred.append(mse_pred)

    fig, ax = plt.subplots()
    ax.plot(mses, color='blue', label='weight error')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean squared error')
    ax.legend(loc='center right')
    ax2 = ax.twinx()
    ax2.plot(mses_pred, color='orange', label='pred error')
    ax2.legend(loc='upper right')
    plt.show()

    if with_bias:
        plt.plot(mses_bias)
        plt.xlabel('Iters')
        plt.ylabel('Bias error')
        plt.show()
    return mses, mses_bias


class TestGBPDenseLayer(TestCase):
    def test_dense_layer(self):
        """Linear regression example"""

        def _test(indims=(20,), n_out_dim=1, noiseless_x=True, with_bias=False):
            tf.random.set_seed(80)
            np.random.seed(80)

            mses, mses_bias = gbp_linear_regression(indims,
                                                    n_out_dim=n_out_dim,
                                                    with_bias=with_bias,
                                                    noiseless_x=noiseless_x)

            # Initial error should be greater than final error
            self.assertLess(mses[-1], mses[0])

            # Final error should be small
            self.assertLess(mses[-1], 0.1)

            if len(mses_bias):
                # Initial error should be greater than final error
                self.assertLess(mses_bias[-1], mses_bias[0])

                # Final error should be small
                self.assertLess(mses_bias[-1], 0.1)

        _test((1,), n_out_dim=1, noiseless_x=True)
        _test((1,), n_out_dim=1, noiseless_x=True, with_bias=True)
        _test((20,), n_out_dim=1, noiseless_x=True)
        _test((20,), n_out_dim=1, noiseless_x=False)
        _test((20,), n_out_dim=1, noiseless_x=False, with_bias=True)
        _test((4, 4, 3), n_out_dim=3, noiseless_x=True)   # For image-like input, vector output
        _test((4, 4, 3), n_out_dim=3, noiseless_x=False)
        _test((4, 4, 3), n_out_dim=3, noiseless_x=False, with_bias=True)
