# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import enable_numpy_behavior

from core.edge import Edge
from core.factors import DenseFactor, UnaryFactor
from core.factors.utils import get_nonlin_fns
from core.inference.gbp.learner import GBPLearner, filtering_on_weights, filtering_on_biases
from core.inference.gbp.layers import GBPDenseLayer
from core.variables import WeightVariable, Variable, BiasVariable

enable_numpy_behavior()


def sample_obs(ndata, indims, outdim, outstd, nonlin_fn):
    x_obs = tf.random.normal([ndata, 1])  # Inputs
    w = tf.random.normal(list(indims) + [outdim])  # Weights
    y_obs = nonlin_fn(
        tf.einsum('ab,ca->cb', w, np.reshape(x_obs, (ndata, 1)))) + tf.random.normal(
        [ndata, outdim]) * outstd  # Outputs
    return x_obs, y_obs


def get_model(x_obs, y_obs, with_bias=False, edge_mom=0.):
    n_out_dim = y_obs.shape[-1]
    indims = x_obs.shape[1:]
    n_data = x_obs.shape[0]

    # Create variables
    weight_vars = WeightVariable(mu_init=tf.zeros([np.prod(indims), n_out_dim]), sigma_init=1.)
    output_vars = Variable(mu_init=tf.zeros([n_data, n_out_dim]), sigma_init=1.)
    if with_bias:
        bias_vars = BiasVariable(mu_init=tf.zeros([n_out_dim]), sigma_init=1.)
    else:
        bias_vars = None

    # Create edges
    # Add zero mean prior to weights and obs factors to outputs and inputs
    weight_prior_sigma = 10.
    output_obs_sigma = 0.01  # out_std

    weight_prior_edge = Edge(fac_to_var_eta=tf.zeros_like(weight_vars.mu),
                             fac_to_var_Lambda=tf.ones_like(weight_vars.mu) / weight_prior_sigma ** 2,
                             var_to_fac_eta=tf.zeros_like(weight_vars.mu),
                             var_to_fac_Lambda=tf.zeros_like(weight_vars.mu))
    weight_prior_factor = UnaryFactor(sigma=weight_prior_sigma,
                                      obs=0.,
                                      var_edges=weight_prior_edge)

    output_obs_edge = Edge(var_to_fac_eta=tf.zeros_like(y_obs),
                           var_to_fac_Lambda=tf.zeros_like(y_obs),
                           fac_to_var_eta=y_obs / output_obs_sigma ** 2.,
                           fac_to_var_Lambda=tf.ones([n_data, n_out_dim]) / output_obs_sigma ** 2.)
    output_obs_factor = UnaryFactor(sigma=output_obs_sigma,
                                    obs=y_obs,
                                    var_edges=output_obs_edge)

    if with_bias:
        bias_prior_sigma = 1.
        bias_prior_edge = Edge(fac_to_var_eta=tf.zeros_like(bias_vars.mu),
                                 fac_to_var_Lambda=tf.ones_like(bias_vars.mu) / bias_prior_sigma ** 2,
                                 var_to_fac_eta=tf.zeros_like(bias_vars.mu),
                                 var_to_fac_Lambda=tf.zeros_like(bias_vars.mu))
        bias_prior_factor = UnaryFactor(sigma=bias_prior_sigma,
                                          obs=0.,
                                          var_edges=bias_prior_edge)
    else:
        bias_prior_edge = None
        bias_prior_factor = None

    # Initialise factor
    weight_edge = Edge(var_to_fac_eta=tf.zeros([n_data, n_out_dim, np.prod(indims)]),
                       var_to_fac_Lambda=tf.ones([n_data, n_out_dim, np.prod(indims)]) / weight_prior_sigma ** 2.,
                       fac_to_var_eta=tf.zeros([n_data, n_out_dim, np.prod(indims)]),
                       fac_to_var_Lambda=tf.zeros([n_data, n_out_dim, np.prod(indims)]),
                       momentum=edge_mom)
    output_edge = Edge(var_to_fac_eta=y_obs / output_obs_sigma ** 2.,
                       var_to_fac_Lambda=tf.ones([n_data, n_out_dim]) / output_obs_sigma ** 2.,
                       fac_to_var_eta=tf.zeros([n_data, n_out_dim]),
                       fac_to_var_Lambda=tf.zeros([n_data, n_out_dim]),
                       momentum=edge_mom)
    if with_bias:
        bias_edge = Edge(var_to_fac_eta=tf.zeros([n_data, n_out_dim]),
                         var_to_fac_Lambda=tf.ones([n_data, n_out_dim]) / bias_prior_sigma ** 2.,
                         fac_to_var_eta=tf.zeros([n_data, n_out_dim]),
                         fac_to_var_Lambda=tf.zeros([n_data, n_out_dim]),
                         momentum=edge_mom)
    else:
        bias_edge = None

    init_lin_point = [weight_vars.mu, output_vars.mu]
    if with_bias:
        init_lin_point.insert(1, bias_vars.mu)
    dense_factor = \
        DenseFactor(sigma=0.1,
                    noiseless_input=True,
                    input_obs=x_obs,
                    weight_var_edges=weight_edge,
                    output_var_edges=output_edge,
                    bias_var_edges=bias_edge if with_bias else None,
                    init_lin_point=init_lin_point,
                    relin_freq=1)

    # Initialise layer
    dense_layer = \
        GBPDenseLayer(dense_factor=dense_factor,
                      coeff_vars=output_vars,
                      coeff_prior_factor=output_obs_factor,
                      weight_vars=weight_vars,
                      weight_prior_factor=weight_prior_factor,
                      bias_vars=bias_vars if with_bias else None,
                      bias_prior_factor=bias_prior_factor if with_bias else None,
                      input_vars=None)

    return GBPLearner(layers=[dense_layer])


def gbp_linear_regression(x_obs, y_obs, n_iters=1, momentum=0., with_bias=False, prev_model=None):
    model = get_model(x_obs, y_obs, with_bias=with_bias, edge_mom=momentum)

    if prev_model is not None:
        filtering_on_weights(prev_model, model)
        if with_bias:
            filtering_on_biases(prev_model, model)

    for i in range(n_iters):
        model.layers[0].intra_layer_inference_iter(i)
        model.layers[0].update_marginals()

    return model


class TestFilter(tf.test.TestCase):
    def test_filtering(self):
        """Linear model example - check that in a linear Gaussian tree model,
        filtering one datapoint at a time gives same (or close) posterior to batch processing"""

        def _test(indims=(20,), use_bias=False):
            tf.random.set_seed(8)
            np.random.seed(8)

            n_data = 20
            out_std = 0.
            n_out_dim = 1
            nonlin_name = 'none'
            nonlin_fn = get_nonlin_fns()[nonlin_name]

            has_loops = np.prod(indims) > 1 or n_out_dim > 1 or use_bias

            # Sample data
            x_obs, y_obs = sample_obs(n_data, indims, n_out_dim, out_std, nonlin_fn)

            model_one_go = gbp_linear_regression(x_obs, y_obs,
                                                 with_bias=use_bias,
                                                 momentum=0.8 if has_loops else 0.,
                                                 n_iters=100 if has_loops else 1)
            model_step_by_step = None
            xys = list(zip(x_obs, y_obs))
            random.shuffle(xys)
            for itr, (x, y) in enumerate(xys):
                model_step_by_step = gbp_linear_regression(x[None], y[None],
                                                           prev_model=model_step_by_step,
                                                           momentum=0.0,
                                                           n_iters=20 if has_loops else 1,
                                                           with_bias=use_bias)
            if has_loops:
                # Only check means are close - precisions will be overconfident!
                self.assertAllClose(model_step_by_step.layers[0].weight_vars.mu,
                                    model_one_go.layers[0].weight_vars.mu,
                                    atol=1e-4)
                if use_bias:
                    self.assertAllClose(model_step_by_step.layers[0].bias_vars.mu,
                                        model_one_go.layers[0].bias_vars.mu,
                                        atol=1e-4)
            else:
                # If model is tree - check full posteriors are close
                self.assertAllClose(model_step_by_step.layers[0].weight_vars.eta,
                                    model_one_go.layers[0].weight_vars.eta,
                                    rtol=1e-4)
                self.assertAllClose(model_step_by_step.layers[0].weight_vars.Lambda,
                                    model_one_go.layers[0].weight_vars.Lambda,
                                    rtol=1e-4)
                if use_bias:
                    self.assertAllClose(model_step_by_step.layers[0].bias_vars.eta,
                                        model_one_go.layers[0].bias_vars.eta,
                                        rtol=1e-4)
                    self.assertAllClose(model_step_by_step.layers[0].bias_vars.Lambda,
                                        model_one_go.layers[0].bias_vars.Lambda,
                                        rtol=1e-4)

        # This example only works for 1D input and 1D output
        # as otherwise graph has loops
        _test((1,), use_bias=False)
        _test((1,), use_bias=True)

    def test_relinearisation(self):
        """Check that relinearisation point of weights in new model is set
        to their posterior mean in old model"""
        def _test(with_bias=False):
            # Train first model
            n_data = 20
            out_std = 0.
            indims = (1,)
            n_out_dim = 1
            nonlin_name = 'none'
            nonlin_fn = get_nonlin_fns()[nonlin_name]
            has_loops = with_bias

            # Sample data
            x_tr, y_tr = sample_obs(n_data, indims, n_out_dim, out_std, nonlin_fn)
            x_te, y_te = sample_obs(n_data + 3, indims, n_out_dim, out_std, nonlin_fn)

            trained_model = gbp_linear_regression(x_tr,
                                                  y_tr,
                                                  with_bias=with_bias,
                                                  momentum=0.8 * float(has_loops),
                                                  n_iters=20 if has_loops else 2)
            new_model = get_model(x_te, y_te, with_bias=with_bias, edge_mom=0.0)

            # Filtering
            filtering_on_weights(trained_model, new_model)
            if with_bias:
                filtering_on_biases(trained_model, new_model)

            # Check linearisation point
            new_weight_lin_point = new_model.layers[0].dense_factor.var0[0]
            self.assertEqual(new_weight_lin_point, trained_model.layers[0].weight_vars.mu)
            if with_bias:
                new_bias_lin_point = new_model.layers[0].dense_factor.var0[1]
                self.assertEqual(new_bias_lin_point, trained_model.layers[0].bias_vars.mu)

        _test()
        _test(with_bias=True)

