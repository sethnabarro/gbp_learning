# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import unittest

from core.edge import Edge
from core.factors import UnaryFactor
from core.utils.utils import flatten_filters, make_tf_filter_shp, mpl_colors, patchify_image
from inputs.synthetic import generate_vhstripes


def get_norm_stats(filters):
    filt_flat = flatten_filters(filters)
    norm = tf.norm(filt_flat, axis=-1)
    print(norm)
    return tf.reduce_mean(norm), tf.math.reduce_std(norm)


def gbp_w_weight_norm(img, init_filters, mom=0.9, niter=100):
    ksize = init_filters.shape[0]
    filtsize = init_filters.shape[0] * init_filters.shape[1]
    nfilt = init_filters.shape[-1]
    filters = init_filters

    # Set filters to ground truth with strong prior sigma - basically keeps filters fixed
    filter_prior_sigma = 1e-3
    filter_prior_eta = flatten_filters(init_filters) / filter_prior_sigma ** 2
    filter_prior_Lambda = tf.ones((nfilt, filtsize)) / filter_prior_sigma ** 2
    fac_to_var_Lambda = tf.zeros((nfilt, filtsize))

    filter_prior_edge = Edge(var_to_fac_eta=filter_prior_eta,
                             var_to_fac_Lambda=filter_prior_Lambda,
                             fac_to_var_eta=tf.zeros((nfilt, filtsize)),
                             fac_to_var_Lambda=fac_to_var_Lambda,
                             momentum=mom)
    dr = UnaryFactor(sigma=filter_prior_sigma,
                     var_edges=filter_prior_edge,
                     obs=init_filters)

    sigma_pix = 1e-3
    pix_eta = img / sigma_pix ** 2
    pix_Lambda = tf.ones_like(img) / sigma_pix ** 2
    pixel_fac_edge = Edge(var_to_fac_eta=tf.zeros_like(img),
                          var_to_fac_Lambda=tf.zeros_like(img),
                          fac_to_var_eta=pix_eta,
                          fac_to_var_Lambda=pix_Lambda,
                          momentum=mom,
                          dropout=0.)
    pixel_obs_factor = UnaryFactor(sigma_pix,
                                   obs=img,
                                   var_edges=pixel_fac_edge)

    filtfac_msg_shp = [1, 2, 2, nfilt, ksize ** 2]
    filter_to_filtfac_edge = Edge(var_to_fac_eta=tf.broadcast_to(filter_prior_eta[None, None, None], filtfac_msg_shp),
                                  var_to_fac_Lambda=tf.broadcast_to(filter_prior_Lambda[None, None, None], filtfac_msg_shp),
                                  fac_to_var_eta=tf.zeros(filtfac_msg_shp),
                                  fac_to_var_Lambda=tf.zeros(filtfac_msg_shp),
                                  momentum=mom)

    pix_eta_patches =  patchify_image(pix_eta, ksize_x=ksize, stride=ksize)
    pix_Lambda_patches =  patchify_image(pix_Lambda, ksize_x=ksize, stride=ksize)
    pixel_to_filtfac_edge = Edge(var_to_fac_eta=tf.broadcast_to(filter_prior_eta[None, None, None], filtfac_msg_shp),
                                 var_to_fac_Lambda=tf.broadcast_to(filter_prior_Lambda[None, None, None], filtfac_msg_shp),
                                 fac_to_var_eta=pix_eta_patches,
                                 fac_to_var_Lambda=pix_Lambda_patches,
                                 momentum=mom)

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


class TestWeightNormFactor(unittest.TestCase):
    def test_l1_norm_factor(self):
        filter_init = tf.convert_to_tensor([[1., 1., 1., 0., 0., 0., 1., 1., 1.],
                                            [1., 0., 1., 1., 0., 1., 1., 0., 1.]])
        filter_init = make_tf_filter_shp(filter_init, ksize=3)
        stripes = generate_vhstripes(imgsize=(6, 3), plot_stripes=True)
        gbp_w_weight_norm(img=stripes, init_filters=filter_init, mom=0.9, niter=20)
