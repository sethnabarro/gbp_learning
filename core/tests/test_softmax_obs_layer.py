# coding=utf-8
import tensorflow as tf
from unittest import TestCase

from core.edge import Edge
from core.factors import SoftmaxClassObservationFactor
from core.inference.gbp.layers import GBPSoftmaxClassObservationLayer, GBPLayerEmpty
from core.variables import CoeffVariable


class TestGBPSoftmaxClassObservationLayer(TestCase):
    def test_softmax_class_observation_layer(self):
        n_data = 3
        n_classes = 10
        correct_classes = [i % n_classes for i in range(n_data)]
        logit_prior_sigma = 1.

        # Logit variables
        logit_vars = CoeffVariable(mu_init=tf.zeros((n_data, n_classes)), sigma_init=1.)

        # Weak zero-mean unary prior on logits
        prior_layer = GBPLayerEmpty(input_vars=None,
                               coeff_vars=logit_vars)

        def prior_coeff_marg_update(return_eta_Lambda=True):
            return tf.zeros_like(logit_vars.eta), tf.ones_like(logit_vars.Lambda) / logit_prior_sigma ** 2.
        prior_layer._update_coeff_marginals = prior_coeff_marg_update

        # One-hot observations
        y_one_hot = [[0.] * 10 for _ in range(n_data)]
        for d in range(n_data):
            y_one_hot[d][correct_classes[d]] = 1.
        y_one_hot = tf.convert_to_tensor(y_one_hot, dtype=tf.float32)

        # Edges between class observation and logit vars
        obs_edges = Edge(var_to_fac_eta=tf.zeros_like(logit_vars.eta),
                         var_to_fac_Lambda=tf.ones_like(logit_vars.Lambda) / logit_prior_sigma ** 2.,
                         fac_to_var_eta=tf.zeros(logit_vars.shape),
                         fac_to_var_Lambda=tf.zeros(logit_vars.shape),
                         momentum=0.8)   # Weird! momentum is necessary for this simple example

        # Observation factor
        sm_factor = SoftmaxClassObservationFactor(label=y_one_hot,
                                                  sigma=0.01,
                                                  logit_var_edges=obs_edges,
                                                  init_lin_point=[logit_vars.mu],
                                                  relin_freq=1)

        sm_layer = GBPSoftmaxClassObservationLayer(input_vars=logit_vars,
                                                   softmax_factor=sm_factor)

        # Linking softmax layer with prior layer will ensure prior messages
        # are included when logit marginals are computed.
        sm_layer.link_layer_before(prior_layer)

        # Do message passing
        # Non-linear so have to iterate
        for i in range(30):
            sm_layer.intra_layer_inference_iter(i)
            print(tf.nn.softmax(sm_layer.input_vars.mu))

        # Sample from logit vars posterior
        n_samples = 100
        logit_samples = \
            sm_layer.input_vars.mu[None] + \
            tf.random.normal(mean=0., stddev=1., shape=(n_samples,))[:, None, None] * sm_layer.input_vars.sigma[None]

        # Compute class probs for samples and average them
        prob_samples = tf.nn.softmax(logit_samples)
        prob_mu = tf.reduce_mean(prob_samples, axis=0)
        prob_std = tf.math.reduce_std(prob_samples, axis=0)
        print(f"Mean predicted probs +- 1std: \n\t{prob_mu.numpy()} +-\n\t{prob_std.numpy()}")

        # Check predicted probs are high for correct class
        for d in range(n_data):
            self.assertGreater(prob_mu[d, correct_classes[d]], 0.95)
            for c in range(n_classes):
                if c == correct_classes[d]:
                    continue
                # And low for incorrect class
                self.assertLess(prob_mu[d, c], 0.05)











