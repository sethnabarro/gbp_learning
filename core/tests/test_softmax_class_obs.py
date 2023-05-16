# coding=utf-8
import tensorflow as tf
from unittest import TestCase

from core.edge import Edge
from core.factors import SoftmaxClassObservationFactor
from core.variables import CoeffVariable


class TestSoftmaxClassObservationFactor(TestCase):
    def test_softmax_class_observation_factor(self):
        correct_class = 2

        # Logit variables
        logit_vars = CoeffVariable(mu_init=tf.zeros((10,)), sigma_init=1.)

        # Edges for a weak zero-mean unary prior on logits
        prior_edges = Edge(var_to_fac_eta=tf.zeros((10,)),
                           var_to_fac_Lambda=tf.zeros((10,)),
                           fac_to_var_eta=tf.zeros((10,)),
                           fac_to_var_Lambda=tf.ones((10,)) * 0.0001)

        # One-hot observation
        y_one_hot = [0.] * 10
        y_one_hot[correct_class] = 1.
        y_one_hot = tf.convert_to_tensor(y_one_hot, dtype=tf.float32)

        # Edges between class observation and logit vars
        obs_edges = Edge(var_to_fac_eta=prior_edges.fac_to_var_eta,
                         var_to_fac_Lambda=prior_edges.fac_to_var_Lambda,
                         fac_to_var_eta=tf.zeros((10,)),
                         fac_to_var_Lambda=tf.zeros((10,)),
                         momentum=0.7)   # Weird! momentum is necessary for this simple example

        # Observation factor
        sm_factor = SoftmaxClassObservationFactor(label=y_one_hot,
                                                  sigma=0.1,
                                                  logit_var_edges=obs_edges,
                                                  init_lin_point=[logit_vars.mu],
                                                  relin_freq=3)

        # Do message passing
        # Non-linear so have to iterate
        for i in range(10):
            sm_factor.update_outgoing_messages(None)
            logit_vars.eta = obs_edges.fac_to_var_eta + prior_edges.fac_to_var_eta
            logit_vars.Lambda = obs_edges.fac_to_var_Lambda + prior_edges.fac_to_var_Lambda
            sm_factor.relinearise(i, [logit_vars.mu])
            print(tf.nn.softmax(logit_vars.mu))

        probs = tf.nn.softmax(logit_vars.mu)
        self.assertGreater(probs[correct_class], 0.95)
        for c in range(10):
            if c == correct_class:
                continue
            self.assertLess(probs[c], 0.05)











