# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unittest

from core.edge import Edge
from core.factors import DiversityFactorCosineSim, get_filter_to_filter_diversity_factor_messages
from core.utils.utils import flatten_filters, make_tf_filter_shp, mpl_colors


def sum_diversity_factor_energies(energy_arr):
    """Sum only upper triangular elems"""
    nfilt = energy_arr.shape[0]
    return tf.reduce_sum(tf.linalg.band_part(energy_arr, 0, -1) - tf.eye(nfilt) * energy_arr).numpy()


def get_cosine_sim2_stats(filters):
    filt_flat = flatten_filters(filters)
    inners = tf.einsum('ab,cb->ac', filt_flat, filt_flat)
    cossim = inners / (tf.norm(filt_flat, axis=-1)[None] * tf.norm(filt_flat, axis=-1)[:, None])
    cossim_upp_tri = cossim.numpy()[np.triu_indices(n=filt_flat.shape[0], k=1)] ** 2
    return tf.reduce_mean(cossim_upp_tri), tf.math.reduce_std(cossim_upp_tri)


def gbp_diversify(init_filters, mom=0.9, niter=100):
    filtsize = init_filters.shape[0] * init_filters.shape[1]
    nfilt = init_filters.shape[-1]
    filters = init_filters

    # Need to include weak prior for conditioning
    prior_sigma = 10.
    prior_eta = flatten_filters(filters) / prior_sigma ** 2

    v2f_eta = \
        tf.convert_to_tensor(
            [[tf.concat([prior_eta[i], prior_eta[j]], axis=-1) for i in range(nfilt)]
             for j in range(nfilt)])

    e = Edge(var_to_fac_eta=v2f_eta,
             var_to_fac_Lambda=tf.ones((nfilt, nfilt, 2 * filtsize)) / prior_sigma ** 2,
             fac_to_var_eta=tf.zeros((nfilt, nfilt, 2 * filtsize)),
             fac_to_var_Lambda=tf.zeros((nfilt, nfilt, 2 * filtsize)),
             momentum=mom)
    dr = DiversityFactorCosineSim(sigma=1., var_edges=e)
    cossim_mean = []
    cossim_std = []
    filters_x0 = filters
    for r in range(niter):
        mu, sigma = get_cosine_sim2_stats(filters)
        cossim_mean.append(mu)
        cossim_std.append(sigma)

        # Compute factor to var messages
        dr.update_outgoing_messages([filters_x0])

        # Get marginals
        filters_marg_eta = tf.reduce_sum(dr.var_edges.fac_to_var_eta, axis=0)[..., -filtsize:]
        filters_marg_eta += tf.reduce_sum(dr.var_edges.fac_to_var_eta, axis=1)[..., :filtsize]
        filters_marg_eta += prior_eta

        if dr.multidim_filters:
            dr.var_edges.fac_to_var_Lambda = \
                tf.concat([1. / tf.linalg.diag_part(tf.linalg.inv(dr.var_edges.fac_to_var_Lambda[..., :filtsize])),
                           1. / tf.linalg.diag_part(tf.linalg.inv(dr.var_edges.fac_to_var_Lambda[..., filtsize:]))], axis=-1)
        filters_marg_Lambda = tf.reduce_sum(dr.var_edges.fac_to_var_Lambda, axis=0)[..., -filtsize:]
        filters_marg_Lambda += tf.reduce_sum(dr.var_edges.fac_to_var_Lambda, axis=1)[..., :filtsize]
        filters_marg_Lambda += tf.ones_like(filters_marg_Lambda) / prior_sigma ** 2

        dr.var_edges.var_to_fac_eta, dr.var_edges.var_to_fac_Lambda = \
            get_filter_to_filter_diversity_factor_messages(dr, filters_marg_eta, filters_marg_Lambda)

        mu = filters_marg_eta / filters_marg_Lambda
        filters = make_tf_filter_shp(mu, ksize=filters.shape[0])
        if r % 20 == 0:
            filters_x0 = filters

    cossim_mean = np.array(cossim_mean)
    cossim_std = np.array(cossim_std)
    plt.plot(cossim_mean, label='mean')
    plt.fill_between(range(len(cossim_mean)),
                     cossim_mean - cossim_std,
                     cossim_mean + cossim_std,
                     alpha=0.3,
                     label='+- 1std')
    plt.legend()
    plt.ylabel('(Cosine Sim)^2')
    plt.xlabel('Iters')
    plt.show()

    for f, (filt_pre, filt_post) in enumerate(zip(flatten_filters(init_filters), mu)):
        plt.plot(filt_pre, linestyle='--', label=f'Initial Filter {f}', color=mpl_colors[f])
        plt.plot(filt_post, label=f'Final Filter {f}', color=mpl_colors[f])
    plt.xlabel('Axis')
    plt.legend()
    plt.show()

    return mu, cossim_mean, cossim_std



class TestDiversityFactors(unittest.TestCase):
    def test_cosine_sim_decreases(self):
        nfilt = 6
        filters = np.random.normal(size=(3, 3, 1, nfilt)).astype(np.float32)
        cossim_avg = gbp_diversify(filters, mom=0.9, niter=100)[1]

        # check cosine sim has decreased on avg
        self.assertLess(cossim_avg[-1], cossim_avg[0])

    def test_zero_energy_for_orthogonal_filters(self):
        nfilt = 4
        # Create filters which are orthogonal and test
        # the energy (=cosine_sim ^ 2) of factors between
        # them is zero
        filters = np.zeros((3, 3, 1, nfilt)).astype(np.float32)
        filters[0, 0, ..., 0] = 1.
        filters[1, 0, ..., 1] = 1.
        filters[2, 0, ..., 2] = 1.
        filters[0, 1, ..., 3] = 1.
        dr = DiversityFactorCosineSim(sigma=1., var_edges=None)
        E_cross = dr.energy([filters]).numpy()[np.triu_indices(n=nfilt, k=1)]
        np.testing.assert_equal(E_cross, np.zeros_like(E_cross))

    def test_one_energy_for_same_filters(self):
        nfilt = 4
        # Create energy of factor between same filter is 1
        filters = np.zeros((3, 3, 1, nfilt)).astype(np.float32)
        filters[0, 0, ..., 0] = 1.
        filters[1, 0, ..., 1] = 1.
        filters[2, 0, ..., 2] = 1.
        filters[0, 1, ..., 3] = 1.
        dr = DiversityFactorCosineSim(sigma=1., var_edges=None)
        E_cross = dr.energy([filters]).numpy()[np.diag_indices(nfilt)]
        np.testing.assert_equal(E_cross, np.ones_like(E_cross))


