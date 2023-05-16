# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unittest

from core.edge import Edge
from core.factors import FilterNormalisationFactor
from core.utils import flatten_filters, make_tf_filter_shp, mpl_colors


def get_norm_stats(filters):
    filt_flat = flatten_filters(filters)
    norm = tf.norm(filt_flat, axis=-1)
    print(norm)
    return tf.reduce_mean(norm), tf.math.reduce_std(norm)


def gbp_normalise(init_filters, mom=0.9, niter=100, multidim=False):
    ksize = init_filters.shape[0]
    filtsize = init_filters.shape[0] * init_filters.shape[1]
    nfilt = init_filters.shape[-1]
    filters = init_filters

    # Need to include weak prior for conditioning
    prior_sigma = 10.
    prior_eta = flatten_filters(filters) / prior_sigma ** 2
    prior_Lambda = tf.ones((nfilt, filtsize)) / prior_sigma ** 2
    fac_to_var_Lambda = tf.zeros((nfilt, filtsize))
    if multidim:
        prior_Lambda = tf.linalg.diag(prior_Lambda)
        fac_to_var_Lambda = tf.zeros((nfilt, filtsize, filtsize))

    e = Edge(var_to_fac_eta=prior_eta,
             var_to_fac_Lambda=prior_Lambda,
             fac_to_var_eta=tf.zeros((nfilt, filtsize)),
             fac_to_var_Lambda=fac_to_var_Lambda,
             momentum=mom)
    dr = FilterNormalisationFactor(sigma=1., var_edges=e, init_lin_point=[filters], relin_freq=10)
    norm_mean = []
    norm_std = []
    filters_x0 = filters
    for r in range(niter):
        mu, sigma = get_norm_stats(filters)
        norm_mean.append(mu)
        norm_std.append(sigma)

        # Compute factor to var messages
        dr.update_outgoing_messages([filters])

        # Get marginals
        filters_marg_eta = dr.var_edges.fac_to_var_eta
        filters_marg_eta += prior_eta


        filters_marg_Lambda = dr.var_edges.fac_to_var_Lambda
        filters_marg_Lambda += prior_Lambda

        dr.var_edges.var_to_fac_eta = filters_marg_eta - dr.var_edges.fac_to_var_eta
        dr.var_edges.var_to_fac_Lambda = filters_marg_Lambda - dr.var_edges.fac_to_var_Lambda

        # dr.var_edges.var_to_fac_eta, dr.var_edges.var_to_fac_Lambda = \
        #     get_var_to_diversity_factor_messages(dr, filters_marg_eta, filters_marg_Lambda)
        sigmasq_diag_part = tf.linalg.diag_part(tf.linalg.inv(filters_marg_Lambda)) if multidim else 1. / filters_marg_Lambda
        mu = filters_marg_eta * sigmasq_diag_part
        filters = make_tf_filter_shp(mu, ksize=filters.shape[0])
        dr.relinearise(r, [filters])
        if r % 100 == 0:
            plt.plot(flatten_filters(filters_x0)[0])
            plt.title(str(norm_mean[-1]))
            plt.show()




    norm_mean = np.array(norm_mean)
    norm_std = np.array(norm_std)
    plt.plot(norm_mean, label='mean')
    plt.fill_between(range(len(norm_mean)),
                     norm_mean - norm_std,
                     norm_mean + norm_std,
                     alpha=0.3,
                     label='+- 1std')
    plt.axhline(1., 0, niter, linestyle='--', color='red', label='target')
    plt.legend()
    plt.ylabel('Norm')
    plt.xlabel('Iters')
    plt.show()

    for f, (filt_pre, filt_post) in enumerate(zip(flatten_filters(init_filters), mu)):
        plt.plot(filt_pre, linestyle='--', label=f'Initial Filter {f}', color=mpl_colors[f])
        plt.plot(filt_post, label=f'Final Filter {f}', color=mpl_colors[f])
    plt.xlabel('Axis')
    plt.legend()
    plt.show()

    return mu, norm_mean, norm_std


class TestNormalisationFactors(unittest.TestCase):
    def test_norm_approaches_one(self):
        nfilt = 8
        filters = np.random.normal(size=(3, 3, 1, nfilt)).astype(np.float32)
        norm_avg = gbp_normalise(filters, mom=0.9, niter=30, multidim=False)[1]

        # check cosine sim has decreased on avg
        self.assertLess(abs(norm_avg[-1] - 1.), abs(norm_avg[0] - 1.))

    def test_zero_energy_for_normalised_filters(self):
        nfilt = 8
        filters = np.zeros((3, 3, 1, nfilt)).astype(np.float32)
        filters[0, 0, ..., 0] = 1.
        filters[1, 0, ..., 1] = -1.
        filters[2, 0, ..., 2] = 1.
        filters[0, 1, ..., 3] = 1.
        dr = NormalisationFactor(sigma=1., var_edges=None, init_lin_point=filters)
        E_norm = dr.energy([filters]).numpy()
        np.testing.assert_equal(E_norm, np.zeros(nfilt))

    def test_energy_for_non_norm_filters(self):
        nfilt = 4
        sigma = 0.5
        # Create energy of factor between same filter is 1
        filters = np.zeros((3, 3, 1, nfilt)).astype(np.float32)
        filters[0, 0, ..., 0] = 1.
        filters[1, 0, ..., 1] = -2.
        filters[2, 0, ..., 2] = 3.
        filters[0, 1, ..., 3] = -4.
        dr = NormalisationFactor(sigma=sigma, var_edges=None, init_lin_point=filters)
        E_norm = dr.energy([filters]).numpy()
        target = (np.array([0., 1., 2., 3.]) / sigma) ** 2   # E = ((Norm - 1) / sigma) ^ 2
        np.testing.assert_equal(E_norm, target.astype(np.float32))


