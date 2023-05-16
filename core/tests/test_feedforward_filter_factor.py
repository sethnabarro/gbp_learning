# coding=utf-8
import matplotlib.pyplot as plt
import tensorflow as tf

from core.edge import Edge
from core.factors import FeedforwardFilterFactor, UnaryFactor, PixelObservationFactor
from core.inference.gbp.layers import GBPConvLayer
from core.inference.gbp.learner import GBPLearner
from core.utils.utils import patchify_image, flatten_filters, create_plot_dirs
from core.utils.plot import plot_model_coeffs, plot_model_weights
from experiments.utils.exp_setup_utils import get_init_conv_weights
from inputs.synthetic import generate_vhstripes
from core.variables import FilterVariable, CoeffVariable, PixelVariable, BiasVariable


def main():
    ksize = 3
    stride = 1
    n_filts = 4

    img_obs_sigma = 0.01
    recon_sigma = 1.5
    weight_prior_sigma = 0.1
    bias_prior_sigma = 0.5
    coeff_prior_sigma = 0.1

    img = generate_vhstripes(12, 0.3, colour=True)[None]
    # img2 = generate_vhstripes(12, 0.3, frac_grad=0.7, colour=True)
    # img = tf.concat([img[None], img2[None]], axis=0)
    # img = tf.random.normal(img.shape)
    filts_gt1 = img[0, 8:11, 8:11][..., None]
    filts_gt2 = img[0, :3, :3][..., None]
    filts_gt3 = img[0, 2:5, 2:5][..., None]
    filts_gt4 = img[0, 3:6, 3:6][..., None]
    filts_gt = tf.concat([filts_gt1, filts_gt2, filts_gt3, filts_gt4], axis=-1)
    coeff_obs = tf.nn.conv2d(img, filts_gt, strides=[1, 1], padding='VALID')

    img_patches = patchify_image(img, ksize_x=ksize, stride=stride)
    filts = get_init_conv_weights(img, n_filters=n_filts, zero_centre=False)
    bias = tf.zeros((n_filts,))
    coeffs = tf.random.normal(img_patches.shape.as_list()[:-2] + [n_filts])

    edge_shp = list(img_patches.shape)
    edge_shp.insert(-1, n_filts)

    input_var_edges = Edge(var_to_fac_eta=tf.broadcast_to(img_patches[..., None, :], edge_shp) / img_obs_sigma ** 2.,
                           var_to_fac_Lambda=tf.ones(edge_shp) / img_obs_sigma ** 2.,
                           fac_to_var_eta=tf.zeros(edge_shp),
                           fac_to_var_Lambda=tf.zeros(edge_shp))

    weight_var_edges = Edge(var_to_fac_eta=tf.zeros(edge_shp),
                           var_to_fac_Lambda=tf.ones(edge_shp) / weight_prior_sigma ** 2.,
                           fac_to_var_eta=tf.zeros(edge_shp),
                           fac_to_var_Lambda=tf.zeros(edge_shp))

    coeff_edge_shp = edge_shp[:-3] + [n_filts]
    coeff_var_edges = Edge(var_to_fac_eta=coeff_obs / coeff_prior_sigma ** 2.,
                           var_to_fac_Lambda=tf.ones(coeff_edge_shp) / coeff_prior_sigma ** 2.,
                           fac_to_var_eta=tf.zeros(coeff_edge_shp),
                           fac_to_var_Lambda=tf.zeros(coeff_edge_shp))

    bias_edge_shp = edge_shp[:-3] + [n_filts]
    bias_var_edges = Edge(var_to_fac_eta=tf.zeros(bias_edge_shp),
                           var_to_fac_Lambda=tf.ones(bias_edge_shp) / bias_prior_sigma ** 2.,
                           fac_to_var_eta=tf.zeros(bias_edge_shp),
                           fac_to_var_Lambda=tf.zeros(bias_edge_shp))


    bias_prior_edges = Edge(var_to_fac_Lambda=tf.zeros_like(bias),
                            var_to_fac_eta=tf.zeros_like(bias),
                            fac_to_var_eta=tf.zeros_like(bias),
                            fac_to_var_Lambda=tf.ones_like(bias) / bias_prior_sigma ** 2.)
    bias_prior_factor = UnaryFactor(sigma=bias_prior_sigma,
                                    obs=tf.zeros_like(bias),
                                    var_edges=bias_prior_edges)

    weight_prior_edges = Edge(var_to_fac_Lambda=tf.zeros_like(flatten_filters(filts)),
                            var_to_fac_eta=tf.zeros_like(flatten_filters(filts)),
                            fac_to_var_eta=tf.zeros_like(flatten_filters(filts)),
                            fac_to_var_Lambda=tf.ones_like(flatten_filters(filts)) / weight_prior_sigma ** 2.)
    weight_prior_factor = UnaryFactor(sigma=weight_prior_sigma,
                                    obs=tf.zeros_like(flatten_filters(filts)),
                                    var_edges=weight_prior_edges)

    coeff_prior_edges = Edge(var_to_fac_Lambda=tf.zeros_like(coeffs),
                            var_to_fac_eta=tf.zeros_like(coeffs),
                            fac_to_var_eta=coeff_obs / coeff_prior_sigma ** 2.,
                            fac_to_var_Lambda=tf.ones_like(coeffs) / coeff_prior_sigma ** 2.)
    coeff_prior_factor = UnaryFactor(sigma=coeff_prior_sigma,
                                    obs=tf.zeros_like(coeffs),
                                    var_edges=coeff_prior_edges)

    img_obs_edges = Edge(var_to_fac_eta=tf.zeros_like(img),
                         var_to_fac_Lambda=tf.zeros_like(img),
                         fac_to_var_eta=img / img_obs_sigma ** 2.,
                         fac_to_var_Lambda=tf.ones_like(img) / img_obs_sigma ** 2.)
    img_obs_factor = PixelObservationFactor(sigma=img_obs_sigma,
                                            obs=img,
                                            var_edges=img_obs_edges,
                                            init_lin_point=[img],
                                            N_rob=0.2)

    filtfac = FeedforwardFilterFactor(sigma=recon_sigma,
                                      n_filters=n_filts,
                                      input_var_edges=input_var_edges,
                                      coeff_var_edges=coeff_var_edges,
                                      filter_var_edges=weight_var_edges,
                                      bias_var_edges=bias_var_edges,
                                      nonlin='none',
                                      init_lin_point=[filts, img, coeffs, bias],
                                      nonlin_yscale=1.,
                                      nonlin_xscale=1.
                                      )
    convlayer = GBPConvLayer(filter_vars=FilterVariable(filts, 1.),
                 input_vars=PixelVariable(img, 1.),
                 coeff_vars=CoeffVariable(coeffs, 1.),
                 bias_vars=BiasVariable(bias, 1.),
                 img_in=img,
                 filter_factor=filtfac,
                 filter_prior_factor=weight_prior_factor,
                 coeff_prior_factor=coeff_prior_factor,
                 bias_prior_factor=bias_prior_factor,
                 pixel_obs_factor=img_obs_factor,
                 fixed_params=False)

    # eta, J = filtfac._get_eta_J_filter_coeffs(None)
    plotdir = '../test_ff/'
    create_plot_dirs(plotdir, ['coeffs', 'filters', 'filters_gt', 'recon'])
    gbp = GBPLearner([convlayer])
    plot_model_coeffs(gbp, plotdir + 'coeffs/', itr=0)
    plot_model_weights(gbp, plotdir + 'filters/', itr=0)
    for j, im in enumerate(gbp.layers[0].input_vars.mu):
        plt.imshow(im, interpolation='none')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(plotdir + f'recon/{j}_start.png')
        plt.close('all')

    gbp.run_inference(20)

    print(gbp.layers[0].bias_vars.mu)
    plot_model_coeffs(gbp, plotdir + 'coeffs/', itr=20)
    plot_model_weights(gbp, plotdir + 'filters/', itr=20)
    gbp.layers[0].filter_vars.eta = filts_gt / 1. ** 2.
    gbp.layers[0].filter_vars.Lambda = tf.ones_like(filts_gt) / 1. ** 2.
    plot_model_weights(gbp, plotdir + 'filters_gt/', itr=20)
    for j, im in enumerate(gbp.layers[0].input_vars.mu):
        plt.imshow(im, interpolation='none')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(plotdir + f'recon/{j}.png')
        plt.close('all')


if __name__ == '__main__':
    main()
