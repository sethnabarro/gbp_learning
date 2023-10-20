# coding=utf-8
"""
Initialises the graph from the config.
Creates variables, factors and edges.
Initialises the edge messages
"""
import numpy as np
import tensorflow as tf
from typing import Iterable

from core.edge import Edge
from core.factors import UnaryFactor, NonLinearUnaryFactor, PixelObservationFactor, \
    PairwiseSmoothingFactorDecomp, AdditiveFilterFactorSumTpose, AvgPoolFactor, MaxPoolFactor, \
    DenseFactor, SoftmaxClassObservationFactor, FeedforwardFilterFactor, UpsampleFactor
from core.inference.gbp.learner import GBPLearner, filtering_on_weights, filtering_on_biases, filtering_on_coeffs, copy_linearisation_points
from core.inference.gbp.layers import GBPConvLayer, GBPPoolLayer, GBPDenseLayer, \
    GBPSoftmaxClassObservationLayer, GBPUpsampleLayer, GBPInputObsLayer, GBPRegressionObsLayer
from core.variables import PixelVariable, FilterVariable, CoeffVariable, BiasVariable, WeightVariable
from core.utils.utils import get_n_strides, patchify_image, dotdict, get_init_conv_weights, get_init_dense_weights, match_edgenames_without_bs


def init_recon_edge_messages(img_in,
                             filter_factor,
                             filter_vars,
                             filter_prior_factor,
                             use_filter_coeffs,
                             stride,
                             coeff_pad=None,
                             inputvars=None,
                             pixel_obs_factor=None,
                             coeff_prior_factor=None,
                             bias_prior_factor=None,
                             use_decomp_filtfac=True,
                             use_feedforward_filtfac=False,
                             use_sum_filtfac=True,
                             use_bias=False):
    coeff_pad = coeff_pad or 0
    is_first_layer = pixel_obs_factor is not None
    is_tpose_recon = isinstance(filter_factor, AdditiveFilterFactorSumTpose)
    is_pairwise_smoother = isinstance(filter_factor, PairwiseSmoothingFactorDecomp)

    if is_first_layer:
        n_strides_x, n_strides_y = get_n_strides(img_in=inputvars.mu,
                                                 stride=stride,
                                                 ksize=filter_vars.k_size)
    else:
        n_strides_y, n_strides_x = coeff_prior_factor.var_edges.var_to_fac_eta.shape[1:3]

        # Subtract coeff padding to get edge shapes for this layer
        n_strides_y -= coeff_pad * 2
        n_strides_x -= coeff_pad * 2
        if inputvars is None:
            raise ValueError('`inputvars` must be given to initialise hidden layers')

    n_channels_in = inputvars.shape[-1]

    edges_shp = \
        [inputvars.shape[0],  # Batch size
         inputvars.shape[1] if is_tpose_recon else n_strides_y,
         inputvars.shape[2] if is_tpose_recon else n_strides_x,
         n_channels_in,
         1 if is_pairwise_smoother else filter_vars.n_filters,
         filter_vars.k_size ** 2]

    if not is_pairwise_smoother:
        # filter prior not robust so don't need to give current value to get eta lambda
        filtprior_eta, filtprior_Lambda = filter_prior_factor.get_eta_Lambda()
        filter_prior_eta_bc = tf.broadcast_to(filtprior_eta[None, None, None], edges_shp)
        filter_prior_Lambda_bc = tf.broadcast_to(filtprior_Lambda[None, None, None], edges_shp)
        filter_factor.filter_var_edges._var_to_fac_eta = filter_prior_eta_bc
        filter_factor.filter_var_edges._var_to_fac_Lambda = filter_prior_Lambda_bc
        filter_factor.filter_var_edges._fac_to_var_eta = tf.zeros(edges_shp)
        filter_factor.filter_var_edges._fac_to_var_Lambda = tf.zeros(edges_shp)

        if use_filter_coeffs:
            coeff_edge_shp = edges_shp.copy()
            if use_feedforward_filtfac:
                coeff_edge_shp = edges_shp[:-3] + [filter_vars.n_filters]
            if not use_decomp_filtfac:
                coeff_edge_shp = coeff_edge_shp[:-1] + [1]
            if is_tpose_recon:
                def coeff_bc(x, pdv=0.):
                    xypad = filter_vars.k_size // 2 * 2
                    x = tf.pad(x, [[0, 0], [xypad, xypad], [xypad, xypad], [0, 0]], constant_values=pdv)
                    xpatch = patchify_image(x, ksize_x=filter_vars.k_size, stride=stride)
                    return tf.broadcast_to(xpatch[..., None, :, :], coeff_edge_shp)
            else:
                def coeff_bc(x, pdv=None):
                    if use_feedforward_filtfac:
                        return x
                    else:
                        return tf.broadcast_to(x[..., None, :, None], coeff_edge_shp)

            coeff_pri_eta = coeff_prior_factor.var_edges.fac_to_var_eta
            coeff_pri_Lambda = coeff_prior_factor.var_edges.fac_to_var_Lambda
            coeff_edge_eta = coeff_bc(coeff_pri_eta if coeff_pad == 0 else coeff_pri_eta[:, coeff_pad:-coeff_pad, coeff_pad:-coeff_pad])
            coeff_edge_Lambda = coeff_bc(coeff_pri_Lambda if coeff_pad == 0 else coeff_pri_Lambda[:, coeff_pad:-coeff_pad, coeff_pad:-coeff_pad], pdv=1e-2)
            filter_factor.coeff_var_edges._var_to_fac_eta = coeff_edge_eta
            filter_factor.coeff_var_edges._var_to_fac_Lambda = coeff_edge_Lambda
            filter_factor.coeff_var_edges._fac_to_var_eta = tf.zeros(coeff_edge_shp)
            filter_factor.coeff_var_edges._fac_to_var_Lambda = tf.zeros(coeff_edge_shp)

        if use_bias:
            bias_edge_shp = edges_shp.copy()
            bias_edge_shp.pop(4)   # One bias for all filters
            if use_feedforward_filtfac:
                bias_edge_shp = bias_edge_shp[:3] + [filter_vars.n_filters]
                eta_expand = bias_prior_factor.var_edges.fac_to_var_eta[None, None, None]
                Lambda_expand = bias_prior_factor.var_edges.fac_to_var_Lambda[None, None, None]
            elif is_tpose_recon:
                bias_edge_shp = bias_edge_shp[:3] + [n_channels_in]
                eta_expand = bias_prior_factor.var_edges.fac_to_var_eta[None, None, None]
                Lambda_expand = bias_prior_factor.var_edges.fac_to_var_Lambda[None, None, None]
            else:
                eta_expand = bias_prior_factor.var_edges.fac_to_var_eta[None, None, None, :, None]
                Lambda_expand = bias_prior_factor.var_edges.fac_to_var_Lambda[None, None, None, :, None]
            filter_factor.bias_var_edges._var_to_fac_eta = \
                tf.broadcast_to(eta_expand, bias_edge_shp)
            filter_factor.bias_var_edges._var_to_fac_Lambda = \
                tf.broadcast_to(Lambda_expand, bias_edge_shp)
            filter_factor.bias_var_edges._fac_to_var_eta = tf.zeros(bias_edge_shp)
            filter_factor.bias_var_edges._fac_to_var_Lambda = tf.zeros(bias_edge_shp)

    input_eta = img_in / pixel_obs_factor.sigma ** 2 if is_first_layer else inputvars.eta
    input_Lambda = tf.ones_like(input_eta) / pixel_obs_factor.sigma ** 2 if is_first_layer else inputvars.Lambda

    if is_first_layer and pixel_obs_factor.mask is not None:

        # Make sure first message updates are well-conditioned
        # Set messages from pix obs factors to pixel vars to ensure non-zero prec
        # for any vars in mask
        pixel_obs_factor.update_outgoing_messages([])
        input_eta = pixel_obs_factor.var_edges.fac_to_var_eta
        input_Lambda = pixel_obs_factor.var_edges.fac_to_var_Lambda

    eta_patches = patchify_image(input_eta,
                                 ksize_x=filter_vars.k_size,
                                 stride=stride)
    Lambda_patches = patchify_image(input_Lambda,
                                    ksize_x=filter_vars.k_size,
                                    stride=stride)

    def _repmat_central_pixel(x, n_repeats=filter_vars.k_size ** 2):
        return tf.repeat(x[..., None, int(filter_vars.k_size ** 2 / 2)],
                         n_repeats, axis=-1)

    if use_decomp_filtfac:
        if filter_factor.relative_to_centre:
            eta_patches = tf.stack([eta_patches, _repmat_central_pixel(eta_patches)], axis=-1)
            Lambda_patches = tf.stack([Lambda_patches, _repmat_central_pixel(Lambda_patches)], axis=-1)
        if use_feedforward_filtfac:
            pass
        else:
            eta_patches = eta_patches[..., None]
            Lambda_patches = Lambda_patches[..., None]

    pixel_edges_shp = edges_shp.copy()

    if use_decomp_filtfac and not is_tpose_recon:
        pixel_edges_shp += (2,) if filter_factor.relative_to_centre else (1,)
    bc = tf.broadcast_to
    if use_feedforward_filtfac:
        pixel_edges_shp = pixel_edges_shp[:-1]
        eta_patches = eta_patches[..., None, :]
        Lambda_patches = Lambda_patches[..., None, :]
    elif is_tpose_recon:
        pixel_edges_shp.pop(5)
        pixel_edges_shp.pop(4)
    elif use_sum_filtfac:
        pixel_edges_shp.pop(4)
        bc = tf.reshape
    else:
        eta_patches = eta_patches[..., None, :, :]
        Lambda_patches = Lambda_patches[..., None, :, :]
    pixel_to_factor_eta = bc(input_eta if is_tpose_recon else eta_patches, pixel_edges_shp)
    pixel_to_factor_Lambda = bc(input_Lambda if is_tpose_recon else Lambda_patches, pixel_edges_shp)
    filter_factor.input_var_edges._var_to_fac_eta = pixel_to_factor_eta
    filter_factor.input_var_edges._var_to_fac_Lambda = pixel_to_factor_Lambda
    filter_factor.input_var_edges._fac_to_var_eta = tf.zeros(pixel_edges_shp)
    filter_factor.input_var_edges._fac_to_var_Lambda = tf.zeros(pixel_edges_shp)


def init_pool_layer(input_vars: [CoeffVariable, PixelVariable],
                    pool_factor_config: dotdict,
                    coeff_prior_factor_config: dotdict,
                    momentum: float,
                    pool_type: [str, None] = None,
                    input_init: [np.array, tf.Tensor, None] = None,
                    layer_prev: [None, GBPPoolLayer] = None) -> GBPPoolLayer:
    is_reinit = layer_prev is not None
    pool_type = pool_type or 'avg'
    if pool_type == 'max':
        assert input_init is not None, \
            "Need to provide initial inputs for linearisation point if `pool_type == 'max'`."

    # What is output shape?
    ks = pool_factor_config.ksize
    inshp = input_vars.shape
    assert (inshp[1] % ks == 0) and (inshp[2] % ks == 0), \
        f"Input must divide by kernel size exactly. Ksize = {ks}, input_shp = {inshp}"
    outshp = [inshp[0], int(inshp[1] / ks), int(inshp[2] / ks), inshp[3]]

    # Create edges to input vars
    inedge_shp = outshp + [ks ** 2]
    fac_to_invar_eta = tf.zeros(inedge_shp)
    fac_to_invar_Lambda = tf.zeros(inedge_shp)
    invar_to_fac_eta = tf.zeros(inedge_shp)
    invar_to_fac_Lambda = tf.zeros(inedge_shp)
    invar_edges = Edge(var_to_fac_eta=invar_to_fac_eta,
                       var_to_fac_Lambda=invar_to_fac_Lambda,
                       fac_to_var_eta=fac_to_invar_eta,
                       fac_to_var_Lambda=fac_to_invar_Lambda,
                       momentum=momentum,
                       name='input_avgpool_edge')

    # Create edges to output vars
    fac_to_outvar_eta = tf.zeros(outshp)
    fac_to_outvar_Lambda = tf.zeros(outshp)
    outvar_to_fac_eta = tf.zeros(outshp)
    outvar_to_fac_Lambda = tf.zeros(outshp)
    outvar_edges = Edge(var_to_fac_eta=outvar_to_fac_eta,
                       var_to_fac_Lambda=outvar_to_fac_Lambda,
                       fac_to_var_eta=fac_to_outvar_eta,
                       fac_to_var_Lambda=fac_to_outvar_Lambda,
                       momentum=momentum,
                        name='output_avgpool_edge')

    outvar_prior_edges = Edge(var_to_fac_eta=outvar_to_fac_eta,
                              var_to_fac_Lambda=outvar_to_fac_Lambda,
                              fac_to_var_eta=fac_to_outvar_eta,
                              fac_to_var_Lambda=tf.ones_like(fac_to_outvar_Lambda) / coeff_prior_factor_config.sigma ** 2.,
                              momentum=momentum,
                              name='output_prior_edge')

    outvar_prior_factor = NonLinearUnaryFactor(var_edges=outvar_prior_edges,
                                               sigma=coeff_prior_factor_config.sigma,
                                               obs=tf.zeros(outshp),
                                               N_rob=coeff_prior_factor_config.N_rob,
                                               rob_type=coeff_prior_factor_config.rob_type,
                                               kmult=coeff_prior_factor_config.kmult,
                                               relin_freq=coeff_prior_factor_config.relin_freq,
                                               init_lin_point=tf.zeros(outshp))

    # Create output vars
    outvars = CoeffVariable(mu_init=tf.zeros(outshp),
                            sigma_init=tf.ones(outshp))

    factor_type = {'avg': AvgPoolFactor, 'max': MaxPoolFactor}[pool_type]
    factor_kwargs = dict(sigma=pool_factor_config.sigma,
                         input_var_edges=invar_edges,
                         output_var_edges=outvar_edges,
                         N_rob=pool_factor_config.N_rob,
                         rob_type=pool_factor_config.rob_type,
                         kmult=pool_factor_config.kmult)
    if pool_type == 'max':
        output_init = tf.nn.max_pool2d(input_init,
                                       ksize=ks,
                                       strides=[1, ks, ks, 1],
                                       padding='VALID')
        factor_kwargs['init_lin_point'] = [input_init, output_init]
        factor_kwargs['relin_freq'] = pool_factor_config.relin_freq

    pool_factor = factor_type(**factor_kwargs)
    if is_reinit:
        layer_prev.pool_factor = pool_factor
        layer_prev.input_vars = input_vars
        layer_prev.coeff_vars = outvars
        return layer_prev

    return GBPPoolLayer(pool_factor=pool_factor,
                        coeff_prior_factor=outvar_prior_factor,
                        coeff_vars=outvars,
                        input_vars=input_vars)


def init_upsample_layer(input_vars: [CoeffVariable, PixelVariable],
                        upsample_factor_config: dotdict,
                        coeff_prior_factor_config: dotdict,
                        momentum: float,
                        dropout: float = 0.,
                        coeff_init_seed: int = 643,
                        coeff_init_std: float = 1.,
                        layer_prev: [None, GBPUpsampleLayer] = None) -> GBPUpsampleLayer:
    is_reinit = layer_prev is not None

    # What is output shape?
    ks = upsample_factor_config.ksize
    inshp = input_vars.shape
    assert (inshp[1] % ks == 0) and (inshp[2] % ks == 0), \
        f"Upsample layer init: input must divide by kernel size exactly. " \
        f"Ksize = {ks}, input_shp = {inshp}"
    outshp = [inshp[0], int(inshp[1] / ks), int(inshp[2] / ks), inshp[3]]
    outedge_shp = outshp + [ks ** 2]
    edge_kwargs = dict(momentum=momentum) #, dropout=dropout)

    # Create edges to input vars
    inedge_shp = outedge_shp
    fac_to_invar_eta = tf.zeros(inedge_shp)
    fac_to_invar_Lambda = tf.zeros(inedge_shp)
    invar_to_fac_eta = tf.zeros(inedge_shp)
    invar_to_fac_Lambda = tf.zeros(inedge_shp)
    invar_edges = Edge(var_to_fac_eta=invar_to_fac_eta,
                       var_to_fac_Lambda=invar_to_fac_Lambda,
                       fac_to_var_eta=fac_to_invar_eta,
                       fac_to_var_Lambda=fac_to_invar_Lambda,
                       name='input_upsample_edge',
                       **edge_kwargs)

    # Create edges to output vars
    fac_to_outvar_eta = tf.zeros(outedge_shp)
    fac_to_outvar_Lambda = tf.zeros(outedge_shp)
    outvar_to_fac_eta = tf.zeros(outedge_shp)
    outvar_to_fac_Lambda = tf.ones(outedge_shp) / coeff_prior_factor_config.sigma ** 2.
    outvar_edges = Edge(var_to_fac_eta=outvar_to_fac_eta,
                        var_to_fac_Lambda=outvar_to_fac_Lambda,
                        fac_to_var_eta=fac_to_outvar_eta,
                        fac_to_var_Lambda=fac_to_outvar_Lambda,
                        name='output_upsample_edge',
                        **edge_kwargs)

    outvar_prior_edges = Edge(var_to_fac_eta=tf.zeros(outshp),
                              var_to_fac_Lambda=tf.zeros(outshp),
                              fac_to_var_eta=tf.zeros(outshp),
                              fac_to_var_Lambda=tf.ones(outshp) / coeff_prior_factor_config.sigma ** 2.,
                              name='output_prior_edge',
                              **edge_kwargs)

    outvar_prior_factor = NonLinearUnaryFactor(var_edges=outvar_prior_edges,
                                               sigma=coeff_prior_factor_config.sigma,
                                               obs=tf.zeros(outshp),
                                               N_rob=coeff_prior_factor_config.N_rob,
                                               rob_type=coeff_prior_factor_config.rob_type,
                                               kmult=coeff_prior_factor_config.kmult,
                                               relin_freq=coeff_prior_factor_config.relin_freq,
                                               init_lin_point=tf.zeros(outshp))

    upsample_factor = UpsampleFactor(sigma=upsample_factor_config.sigma,
                                     input_var_edges=invar_edges,
                                     output_var_edges=outvar_edges,
                                     N_rob=upsample_factor_config.N_rob,
                                     rob_type=upsample_factor_config.rob_type,
                                     kmult=upsample_factor_config.kmult)

    # Create output vars
    coeff_init = tf.random.stateless_normal(shape=outshp, seed=[coeff_init_seed - 1, coeff_init_seed]) * coeff_init_std
    outvars = CoeffVariable(mu_init=coeff_init,
                            sigma_init=tf.ones(outshp))
    invar_mu = tf.keras.layers.UpSampling2D(size=(upsample_factor.ksize, upsample_factor.ksize))(outvars.mu)
    input_vars.eta = invar_mu / input_vars.sigma ** 2.

    if is_reinit:
        layer_prev.pool_factor = upsample_factor
        layer_prev.input_vars = input_vars
        layer_prev.coeff_vars = outvars
        return layer_prev

    return GBPUpsampleLayer(input_vars=input_vars,
                            coeff_vars=outvars,
                            upsample_factor=upsample_factor,
                            coeff_prior_factor=outvar_prior_factor)


def init_conv_layer(filters_init, config, n_filts,
                    img_noisy=None, inputvars=None,
                    img_gt=None, layer_id=0,
                    coeff_init=None, pix_init=None,
                    img_mask=None, ksize=3, stride=1,
                    coeff_init_seed=999, padding=None, coeff_padding=None,
                    fac_to_var_chunksize=1, layer_prev=None) -> GBPConvLayer:
    is_first_layer = layer_id == 0
    if not is_first_layer:
        if inputvars is None:
            raise ValueError('`inputvars` should be given when initialising hidden layers.')

    assert config.factors.recon.relin_freq is not None, \
        "Need to do relinearisation in recon factors as message " \
        "updates depend on current variable values. " \
        "`config.factors.relin_freq` should be integer."

    padding = padding or 0
    coeff_padding = coeff_padding or 0
    is_reinit = layer_prev is not None

    if is_reinit:
        layer_prev.filter_vars = FilterVariable(filters_init, config.factors.weight_prior.sigma)
        filtervars = layer_prev.filter_vars
    else:
        filtervars = FilterVariable(filters_init, config.factors.weight_prior.sigma)

    if is_first_layer:
        if pix_init is not None:
            pix_mu_init = pix_init
        elif img_noisy is not None:
            # TODO: don't like img_gt input. Model never sees "ground_truth"
            # Instead of both img_noisy and img_gt inputs, just have `img_obs`
            pix_mu_init = img_noisy
        else:
            pix_mu_init = img_gt   # If not doing denoising

        if padding > 0:
            if img_mask is None:
                img_mask = tf.ones_like(pix_mu_init)
            npad = padding
            pads = [[0, 0], [npad, npad], [npad, npad], [0, 0]]
            pix_mu_init = tf.pad(pix_mu_init, paddings=pads, mode='REFLECT')
            img_noisy = tf.pad(img_noisy, paddings=pads, mode='REFLECT')
            if img_gt is not None:
                img_gt = tf.pad(img_gt, paddings=pads, mode='REFLECT')
            img_mask = tf.pad(img_mask, paddings=pads,)

        if is_reinit:
            layer_prev.input_vars = PixelVariable(mu_init=pix_mu_init,
                                      sigma_init=config.factors.pixel_obs.sigma)
            inputvars = layer_prev.input_vars
        else:
            inputvars = PixelVariable(mu_init=pix_mu_init,
                                      sigma_init=config.factors.pixel_obs.sigma)

    edge_args = dict(momentum=config.momentum,
                     dropout=config.dropout)
    if is_reinit:
        layer_prev.filter_factor.input_var_edges = Edge(name='input_filtfac_edge', **edge_args)
        layer_prev.filter_factor.filter_var_edges = Edge(name='filter_filtfac_edge', **edge_args)
        input_filter_fac_edge = layer_prev.filter_factor.input_var_edges
        filter_filter_fac_edge = layer_prev.filter_factor.filter_var_edges
    else:
        input_filter_fac_edge = Edge(name='input_filtfac_edge', **edge_args)
        filter_filter_fac_edge = Edge(name='filter_filtfac_edge', **edge_args)

    biasvars = None
    bias_prior_factor = None
    bias_filter_fac_edge = None
    if config.factors.recon.use_bias:
        assert config.factors.recon.sum_filters, "Bias currently only supported with sum filter recon factors"
        biasshp = [n_filts] if config.factors.recon.feedforward else [inputvars.shape[-1]]

        biasvars = BiasVariable(mu_init=tf.ones(biasshp) * config.factors.bias_prior.mean,
                                sigma_init=config.factors.bias_prior.sigma)
        bias_prior_eta = biasvars.mu / config.factors.bias_prior.sigma ** 2.
        bias_prior_Lambda = tf.ones_like(biasvars.mu) / config.factors.bias_prior.sigma ** 2.
        bias_prior_edge = Edge(var_to_fac_eta=tf.zeros(biasshp),
                               var_to_fac_Lambda=tf.zeros(biasshp),
                               fac_to_var_eta=bias_prior_eta,
                               fac_to_var_Lambda=bias_prior_Lambda,
                               name='bias_prior_edge',
                               **edge_args)
        bias_filter_fac_edge = Edge(name='bias_filtfac_edge', **edge_args)
        bias_prior_factor = UnaryFactor(sigma=config.factors.bias_prior.sigma,
                                        obs=config.factors.bias_prior.mean * tf.ones(biasshp),
                                        var_edges=bias_prior_edge)
        if is_reinit:
            layer_prev.bias_vars = BiasVariable(mu_init=tf.ones(biasshp) * config.factors.bias_prior.mean,
                                sigma_init=config.factors.bias_prior.sigma)
            biasvars = layer_prev.bias_vars
            layer_prev.bias_prior_factor = bias_prior_factor

    if config.use_filter_coeffs:
        nx, ny = get_n_strides(img_in=inputvars.mu, stride=config.factors.recon.stride, ksize=ksize)
        coeffshp = [inputvars.shape[0], ny + 2 * coeff_padding, nx + 2 * coeff_padding, n_filts]
        if coeff_init is None:
            if config.random_coeff_init:
                coeff_var_init = tf.random.stateless_normal(shape=coeffshp, seed=[coeff_init_seed - 1, coeff_init_seed]) * config.init_coeff_std
            else:
                coeff_var_init = tf.ones(coeffshp) * config.coeff_init_scale
        else:
            coeff_var_init = coeff_init

        coeff_msg_shp = coeffshp
        coeff_prior_eta = tf.ones(
            coeff_msg_shp) * config.factors.coeff_prior.mean / config.factors.coeff_prior.sigma ** 2
        coeff_prior_Lambda = tf.ones(coeff_msg_shp) / config.factors.coeff_prior.sigma ** 2

        coeffvars = CoeffVariable(coeff_var_init, 1.)
        coeff_prior_edge = Edge(var_to_fac_eta=tf.zeros(coeff_msg_shp),
                                var_to_fac_Lambda=tf.zeros(coeff_msg_shp),
                                fac_to_var_eta=coeff_prior_eta,
                                fac_to_var_Lambda=coeff_prior_Lambda,
                                name='coeff_prior_edge',
                                **edge_args)
        coeff_filter_fac_edge = Edge(name='coeff_filtfac_edge', **edge_args)
        coeff_prior_factor = NonLinearUnaryFactor(sigma=config.factors.coeff_prior.sigma,
                                                  obs=config.factors.coeff_prior.mean * tf.ones(coeff_msg_shp),
                                                  var_edges=coeff_prior_edge,
                                                  init_lin_point=[coeff_var_init],
                                                  N_rob=config.factors.coeff_prior.N_rob,
                                                  rob_type=config.factors.coeff_prior.rob_type)
        if is_reinit:
            layer_prev.coeff_vars = CoeffVariable(coeff_var_init, 1.)
            coeffvars = layer_prev.coeff_vars
            layer_prev.coeff_prior_factor = coeff_prior_factor

    else:
        coeffvars = None
        coeff_filter_fac_edge = None
        coeff_prior_factor = None

    filters_flat = filtervars.mu_flat
    filters_init = filtervars.mu
    filter_prior_mean = filters_flat if config.factors.weight_prior.random_mean else tf.zeros_like(filters_flat)
    filter_prior_sigma = config.factors.weight_prior.sigma

    filter_prior_edge = Edge(var_to_fac_eta=tf.zeros_like(filter_prior_mean),
                             var_to_fac_Lambda=tf.zeros_like(filter_prior_mean),
                             fac_to_var_eta=filter_prior_mean / filter_prior_sigma ** 2,
                             fac_to_var_Lambda=tf.ones_like(filter_prior_mean) / filter_prior_sigma ** 2,
                             name='filter_prior_edge',
                             **edge_args)

    filter_prior_factor = UnaryFactor(filter_prior_sigma,
                                      obs=filter_prior_mean,
                                      var_edges=filter_prior_edge)
    if is_reinit:
        layer_prev.filter_prior_factor = filter_prior_factor
        layer_prev.filter_prior_factor.var_edges = filter_prior_edge

    if is_first_layer:
        pixobs_to_var_eta = inputvars.mu / config.factors.pixel_obs.sigma ** 2.
        pixobs_to_var_Lambda = tf.ones_like(inputvars.mu) / config.factors.pixel_obs.sigma ** 2
        input_edge_kwargs = dict(var_to_fac_eta=tf.zeros_like(inputvars.mu),
                                 var_to_fac_Lambda=tf.zeros_like(inputvars.mu),
                                 fac_to_var_eta=pixobs_to_var_eta,
                                 fac_to_var_Lambda=pixobs_to_var_Lambda,
                                 name='input_obs_edge',
                                 **edge_args)

        input_fac_edge = Edge(**input_edge_kwargs)
        pixel_obs_factor = PixelObservationFactor(config.factors.pixel_obs.sigma,
                                                  obs=img_noisy,
                                                  init_lin_point=[img_noisy],
                                                  relin_freq=config.factors.pixel_obs.relin_freq,
                                                  var_edges=input_fac_edge,
                                                  N_rob=config.factors.pixel_obs.N_rob,
                                                  rob_type=config.factors.pixel_obs.rob_type,
                                                  kmult=config.factors.pixel_obs.kmult,
                                                  mask=img_mask,
                                                  mask_prec=config.factors.pixel_obs.mask_prec)
        if is_reinit:
            layer_prev.pixel_obs_factor = pixel_obs_factor

    else:
        pixel_obs_factor = None


    if config.factors.recon.pairwise:
        def filtfactclass(*args, **kwargs):
            return PairwiseSmoothingFactorDecomp(n_filters=None, *args, **kwargs)
    elif config.factors.recon.sum_filters:
        assert config.use_filter_coeffs, "Sum recon factors only supported when using filter coeffs"
        if config.factors.recon.tpose:  # Generate, transpose conv
            def filtfactclass(*args, **kwargs):
                return AdditiveFilterFactorSumTpose(n_filters=n_filts, *args, **kwargs)
        elif config.factors.recon.feedforward:
            def filtfactclass(*args, **kwargs):
                return FeedforwardFilterFactor(n_filters=n_filts, *args, **kwargs)
    init_lin_point = [filters_init, inputvars.mu]
    if config.use_filter_coeffs:
        init_lin_point += [coeff_var_init]
    if config.factors.recon.use_bias:
        init_lin_point += [biasvars.mu]

    filtfac_kwargs = dict(sigma=config.factors.recon.sigma,
                          init_lin_point=init_lin_point,
                          relin_freq=config.factors.recon.relin_freq,
                          input_var_edges=input_filter_fac_edge,
                          filter_var_edges=filter_filter_fac_edge,
                          coeff_var_edges=coeff_filter_fac_edge,
                          bias_var_edges=bias_filter_fac_edge,
                          N_rob=config.factors.recon.N_rob,
                          rob_type=config.factors.recon.rob_type,
                          rec_field=(ksize, ksize),
                          stride=stride,
                          relative_to_centre=config.factors.recon.relative_to_centre or False,
                          kmult=config.factors.recon.kmult,
                          nonlin=config.factors.recon.nonlin,
                          nonlin_xscale=config.factors.recon.nonlin_xscale,
                          nonlin_yscale=config.factors.recon.nonlin_yscale,
                          sum_before_nonlin=config.factors.recon.sum_before_nonlin or True,
                          fac_to_var_chunksize=fac_to_var_chunksize,
                          coeff_padding=coeff_padding)

    if is_reinit:
        layer_prev.filter_factor = filtfactclass(**filtfac_kwargs)
        filter_factor = layer_prev.filter_factor
        layer_prev.input_vars = inputvars
    else:
        filter_factor = filtfactclass(**filtfac_kwargs)

    init_recon_edge_messages(img_in=img_noisy,
                             filter_factor=filter_factor,
                             filter_vars=filtervars,
                             filter_prior_factor=filter_prior_factor,
                             pixel_obs_factor=pixel_obs_factor,
                             inputvars=inputvars,
                             use_filter_coeffs=config.use_filter_coeffs,
                             stride=stride,
                             coeff_prior_factor=coeff_prior_factor,
                             bias_prior_factor=bias_prior_factor,
                             use_decomp_filtfac=config.factors.recon.decompose,
                             use_sum_filtfac=config.factors.recon.sum_filters,
                             use_feedforward_filtfac=config.factors.recon.feedforward,
                             use_bias=config.factors.recon.use_bias,
                             coeff_pad=coeff_padding)

    layclasses = {'gbp': GBPConvLayer}
    extra_args = {}
    if is_reinit:
        layer = layer_prev
    else:
        layer = layclasses['gbp'](
            img_in=img_noisy,
            filter_vars=filtervars,
            input_vars=inputvars,
            bias_vars=biasvars,
            filter_factor=filter_factor,
            filter_prior_factor=filter_prior_factor,
            coeff_prior_factor=coeff_prior_factor,
            bias_prior_factor=bias_prior_factor,
            pixel_obs_factor=pixel_obs_factor,
            stride=stride,
            coeff_vars=coeffvars,
            img_ground_truth=img_gt if is_first_layer else None,
            **extra_args)
    if config.factors.recon.fixed_inputs:
        layer.fix_inputs()

    return layer


def init_dense_layer(config: dotdict,
                     weights_init: np.array,
                     input_vars: [PixelVariable, CoeffVariable],
                     coeff_init_seed: int = 722,
                     fac_to_var_chunksize: int = 1,
                     layer_prev: [None, GBPDenseLayer] = None):
    is_reinit = layer_prev is not None

    # Instantiate variables
    ndim_in, ndim_out = weights_init.shape
    n_data = input_vars.shape[0]
    if is_reinit:
        layer_prev.weight_vars = WeightVariable(weights_init, sigma_init=config.factors.weight_prior.sigma)
        weightvars = layer_prev.weight_vars
    else:
        weightvars = WeightVariable(weights_init, sigma_init=config.factors.weight_prior.sigma)

    if config.factors.dense.use_bias:
        prod = tf.einsum('ba,cb->ca', weightvars.mu, tf.reshape(input_vars.mu, [input_vars.mu.shape[0], -1]))
        bias_init = - np.mean(prod, axis=0)    # Choose init bias s.t. outputs zero-centred
        if is_reinit:
            layer_prev.bias_vars = BiasVariable(bias_init, sigma_init=config.factors.bias_prior.sigma)
            biasvars = layer_prev.bias_vars
        else:
            biasvars = BiasVariable(bias_init, sigma_init=config.factors.bias_prior.sigma)
    else:
        biasvars = None
    if config.random_coeff_init:
        # Init coeffs to be consistent with weights, inputs and biases
        coeff_var_init = tf.einsum('ba,cb->ca', weightvars.mu, tf.reshape(input_vars.mu, [input_vars.mu.shape[0], -1]))
        coeff_var_init += bias_init[None] if config.factors.dense.use_bias else 0.
    else:
        coeff_var_init = tf.ones((n_data, ndim_out)) * config.coeff_init
    if is_reinit:
        layer_prev.coeff_vars = CoeffVariable(coeff_var_init, sigma_init=config.factors.coeff_prior.sigma)
        coeffvars = layer_prev.coeff_vars
    else:
        coeffvars = CoeffVariable(coeff_var_init, sigma_init=config.factors.coeff_prior.sigma)


    # Create edges to dense factor
    n_data = input_vars.shape[0]
    edge_args = dict(momentum=config.momentum, dropout=config.dropout)
    decompose = bool(config.factors.dense.decompose) if config.factors.dense.decompose is not None else True
    in_shp = [n_data, ndim_out, ndim_in] if decompose else [n_data, ndim_in]
    input_edge = Edge(
        var_to_fac_eta=tf.broadcast_to(tf.reshape(input_vars.eta, [n_data, 1, ndim_in]), in_shp) if decompose else tf.reshape(input_vars.eta, in_shp),    # Just for init, will be updated during GBP
        var_to_fac_Lambda=tf.broadcast_to(tf.reshape(input_vars.Lambda, [n_data, 1, ndim_in]), in_shp) if decompose else tf.reshape(input_vars.Lambda, in_shp),
        fac_to_var_eta=tf.zeros(in_shp),
        fac_to_var_Lambda=tf.zeros(in_shp),
        name='input_edge',
        **edge_args)
    weight_edge = Edge(var_to_fac_eta=tf.broadcast_to(weights_init.T[None] / config.factors.weight_prior.sigma ** 2., [n_data, ndim_out, ndim_in]),   # Just for init, will be updated during GBP
                       var_to_fac_Lambda=tf.ones([n_data, ndim_out, ndim_in]) / config.factors.weight_prior.sigma ** 2.,
                       fac_to_var_eta=tf.zeros([n_data, ndim_out, ndim_in]),
                       fac_to_var_Lambda=tf.zeros([n_data, ndim_out, ndim_in]),
                       name='weight_edge',
                       **edge_args)
    output_edge = Edge(var_to_fac_eta=tf.broadcast_to(coeff_var_init / config.factors.coeff_prior.sigma ** 2., [n_data, ndim_out]),    # Just for init, will be updated during GBP
                       var_to_fac_Lambda=tf.ones([n_data, ndim_out]) / config.factors.coeff_prior.sigma ** 2.,
                       fac_to_var_eta=tf.zeros([n_data, ndim_out]),
                       fac_to_var_Lambda=tf.zeros([n_data, ndim_out]),
                       name='coeff_edge',
                       **edge_args)
    if config.factors.dense.use_bias:
        bias_edge = Edge(var_to_fac_eta=tf.broadcast_to(bias_init / config.factors.bias_prior.sigma ** 2., [n_data, ndim_out]),     # Just for init, will be updated during GBP
                         var_to_fac_Lambda=tf.ones([n_data, ndim_out]) / config.factors.bias_prior.sigma ** 2.,
                         fac_to_var_eta=tf.zeros([n_data, ndim_out]),
                         fac_to_var_Lambda=tf.zeros([n_data, ndim_out]),
                         name='bias_edge',
                         **edge_args)
    else:
        bias_edge = None

    # Edges to prior factors
    coeff_prior_edge = Edge(
        var_to_fac_eta=tf.zeros([n_data, ndim_out]),
        var_to_fac_Lambda=tf.zeros([n_data, ndim_out]),
        fac_to_var_eta=tf.zeros([n_data, ndim_out]),
        fac_to_var_Lambda=tf.ones([n_data, ndim_out]) / config.factors.coeff_prior.sigma ** 2.,
        name='coeff_prior_edge',
        **edge_args)
    weight_prior_edge = Edge(var_to_fac_eta=tf.zeros([ndim_in, ndim_out]),
                             var_to_fac_Lambda=tf.zeros([ndim_in, ndim_out]),
                             fac_to_var_eta=tf.broadcast_to(weights_init / config.factors.weight_prior.sigma ** 2., [ndim_in, ndim_out]),   # Just for init, will be updated during GBP
                             fac_to_var_Lambda=tf.ones([ndim_in, ndim_out]) / config.factors.weight_prior.sigma ** 2.,
                             name='weight_prior_edge',
                             **edge_args)

    # Create factors
    init_lin_point = [input_vars.mu, weightvars.mu, coeffvars.mu]
    if config.factors.dense.use_bias:
        init_lin_point.insert(2, biasvars.mu)
    if config.factors.dense.noiseless_input:
        init_lin_point.pop(0)
    dense_factor = \
        DenseFactor(sigma=config.factors.dense.sigma,
                    input_var_edges=input_edge,
                    output_var_edges=output_edge,
                    weight_var_edges=weight_edge,
                    bias_var_edges=bias_edge,
                    init_lin_point=init_lin_point,
                    relin_freq=config.factors.dense.relin_freq,
                    N_rob=config.factors.dense.N_rob,
                    rob_type=config.factors.dense.rob_type,
                    kmult=config.factors.dense.kmult,
                    noiseless_input=config.factors.dense.noiseless_input,
                    input_obs=input_vars.mu,
                    nonlin=config.factors.dense.nonlin or 'none',
                    nonlin_yscale=config.factors.dense.nonlin_yscale or 1.,
                    nonlin_xscale=config.factors.dense.nonlin_xscale or 1.,
                    fac_to_var_chunksize=fac_to_var_chunksize,
                    decompose=decompose)
    dense_factor.relinearise(0, init_lin_point)

    coeff_prior_factor = \
        NonLinearUnaryFactor(sigma=config.factors.coeff_prior.sigma,
                             obs=tf.zeros_like(coeffvars.mu),
                             var_edges=coeff_prior_edge,
                             init_lin_point=[coeffvars.mu],
                             N_rob=config.factors.coeff_prior.N_rob,
                             rob_type=config.factors.coeff_prior.rob_type,
                             kmult=config.factors.coeff_prior.kmult,
                             relin_freq=config.factors.coeff_prior.relin_freq)
    weight_prior_factor = \
        NonLinearUnaryFactor(sigma=config.factors.weight_prior.sigma,
                             obs=tf.zeros_like(weightvars.mu),
                             var_edges=weight_prior_edge,
                             init_lin_point=[weightvars.mu],
                             N_rob=config.factors.weight_prior.N_rob,
                             rob_type=config.factors.weight_prior.rob_type,
                             kmult=config.factors.weight_prior.kmult,
                             relin_freq=config.factors.weight_prior.relin_freq)
    if config.factors.dense.use_bias:
        bias_prior_edge = Edge(fac_to_var_eta=bias_init / config.factors.bias_prior.sigma ** 2.,    # Just for init, will be updated during GBP
                               fac_to_var_Lambda=tf.ones([ndim_out]) / config.factors.bias_prior.sigma ** 2.,
                               var_to_fac_eta=tf.zeros([ndim_out]),
                               var_to_fac_Lambda=tf.zeros([ndim_out]),
                               name='bias_prior_edge',
                               **edge_args)
        bias_prior_factor = \
            NonLinearUnaryFactor(sigma=config.factors.bias_prior.sigma,
                                 obs=bias_init,
                                 var_edges=bias_prior_edge,
                                 init_lin_point=[biasvars.mu],
                                 N_rob=config.factors.bias_prior.N_rob,
                                 rob_type=config.factors.bias_prior.rob_type,
                                 kmult=config.factors.bias_prior.kmult,
                                 relin_freq=config.factors.bias_prior.relin_freq)
    else:
        bias_prior_factor = None

    if is_reinit:
        layer_prev.dense_factor = dense_factor
        layer_prev.weight_prior_factor = weight_prior_factor
        layer_prev.coeff_prior_factor = coeff_prior_factor
        layer_prev.bias_prior_factor = bias_prior_factor
        layer_prev.input_vars = input_vars
        return layer_prev

    return GBPDenseLayer(dense_factor=dense_factor,
                         coeff_prior_factor=coeff_prior_factor,
                         weight_prior_factor=weight_prior_factor,
                         input_vars=input_vars,
                         weight_vars=weightvars,
                         coeff_vars=coeffvars,
                         bias_vars=biasvars,
                         bias_prior_factor=bias_prior_factor)


def init_classification_layer(config: dotdict,
                              input_vars: [PixelVariable, CoeffVariable],
                              input_prior_factor: UnaryFactor,
                              class_label: np.array,
                              momentum: float = 0.,
                              classifier_type: [str, None] = None,
                              layer_prev: [None, GBPSoftmaxClassObservationLayer] = None,
                              classes_sub: [None, list, tuple] = None):
    is_reinit = layer_prev is not None
    input_to_class_factor_edge = \
        Edge(var_to_fac_eta=input_prior_factor.var_edges.fac_to_var_eta,
             var_to_fac_Lambda=input_prior_factor.var_edges.fac_to_var_Lambda,
             fac_to_var_eta=tf.zeros(input_vars.shape),
             fac_to_var_Lambda=tf.zeros(input_vars.shape),
             momentum=momentum,
             name='class_input_edge')

    factor_type = SoftmaxClassObservationFactor
    layer_type = GBPSoftmaxClassObservationLayer

    class_factor = factor_type(label=class_label,
                               sigma=config.sigma,
                               logit_var_edges=input_to_class_factor_edge,
                               relin_freq=config.relin_freq,
                               init_lin_point=[input_vars.mu],
                               kmult=config.kmult,
                               N_rob=config.N_rob,
                               rob_type=config.rob_type,
                               classes_sub=classes_sub)
    if is_reinit:
        layer_prev.input_vars = input_vars
        layer_prev.softmax_factor = class_factor
        return layer_prev

    class_layer = layer_type(class_factor, input_vars=input_vars)
    return class_layer


def init_regression_obs_layer(y_obs: np.array,
                              input_vars: CoeffVariable,
                              config: dotdict):
    # Create output obs factor
    obs_to_var_eta = y_obs / config.sigma ** 2.
    obs_to_var_Lambda = tf.ones_like(y_obs) / config.sigma ** 2.
    output_fac_edge = Edge(var_to_fac_eta=tf.zeros_like(y_obs),
                           var_to_fac_Lambda=tf.zeros_like(y_obs),
                           fac_to_var_eta=obs_to_var_eta,
                           fac_to_var_Lambda=obs_to_var_Lambda)

    regression_obs_factor = UnaryFactor(config.sigma,
                                        obs=y_obs,
                                        var_edges=output_fac_edge,
                                        N_rob=config.N_rob,
                                        rob_type=config.rob_type,
                                        kmult=config.kmult,)
    layr = GBPRegressionObsLayer(regression_obs_factor=regression_obs_factor, input_vars=input_vars)
    return layr


def init_input_obs_layer(img_noisy: np.array, config: dotdict, img_mask=None):
    # Create pixl obs factor
    pixobs_to_var_eta = img_noisy / config.factors.pixel_obs.sigma ** 2.
    pixobs_to_var_Lambda = tf.ones_like(img_noisy) / config.factors.pixel_obs.sigma ** 2.
    if img_mask is not None:
        pixobs_to_var_eta *= tf.cast(img_mask, pixobs_to_var_eta.dtype)
        pixobs_to_var_Lambda *= tf.cast(img_mask, pixobs_to_var_Lambda.dtype)
    input_fac_edge = Edge(var_to_fac_eta=tf.zeros_like(img_noisy),
                          var_to_fac_Lambda=tf.zeros_like(img_noisy),
                          fac_to_var_eta=pixobs_to_var_eta,
                          fac_to_var_Lambda=pixobs_to_var_Lambda)

    pixel_obs_factor = PixelObservationFactor(config.factors.pixel_obs.sigma,
                                              obs=img_noisy,
                                              init_lin_point=[img_noisy],
                                              relin_freq=config.factors.pixel_obs.relin_freq,
                                              var_edges=input_fac_edge,
                                              N_rob=config.factors.pixel_obs.N_rob,
                                              rob_type=config.factors.pixel_obs.rob_type,
                                              kmult=config.factors.pixel_obs.kmult,
                                              mask=img_mask,
                                              mask_prec=config.factors.pixel_obs.mask_prec)
    input_vars = PixelVariable(mu_init=img_noisy, sigma_init=config.factors.pixel_obs.sigma)
    layr = GBPInputObsLayer(pixel_obs_factor=pixel_obs_factor, input_vars=input_vars)
    return layr


def init_layers(config: dotdict,
                img_obs: np.array,
                weight_init_std: float,
                weight_init_seed: int = 897,
                coeff_init_seed: int = 222,
                img_mask: [None, np.array] = None,
                img_gt: [None, np.array] = None,
                pix_var_init: [None, np.array] = None,
                filter_var_init: [None, Iterable[np.array]] = None,
                coeff_init: [None, Iterable[np.array]] = None,
                output_obs: [np.array, None] = None,
                layers_prev: [None, list, tuple] = None,
                classes_sub: [None, list, tuple] = None,
                batch_id: int = 0):

    def _layer_config_override(global_conf: dotdict, layer_conf: dotdict):
        if layer_conf.factors is not None:
            # Copy global conf to not overwrite defaults
            local_conf = global_conf.copy()
            local_conf.factors = global_conf.factors.copy()
            for f in layer_conf.factors:
                # Overwrite global (default) factor config elems with local
                # Need to create local copy for each overwritten factor type
                local_conf.factors[f] = global_conf.factors[f].copy()
                for k in layer_conf.factors[f]:
                    local_conf.factors[f][k] = layer_conf.factors[f][k]
            return local_conf
        else:
            return global_conf

    # sample initial network weights
    layers = []
    layers_prev = layers_prev or [None] * len(config.architecture)
    height, width, n_filts_in = img_obs.shape[1:]
    for lay_id, (layconf, layprev) in enumerate(zip(config.architecture, layers_prev)):
        print(height, width, n_filts_in, layconf)
        local_config = _layer_config_override(config, layconf)
        if layconf.type == 'conv':
            kmeans_filter_init = layconf.kmeans_init if layconf.kmeans_init else False

            if filter_var_init is not None:
                if isinstance(filter_var_init[lay_id], np.array):
                    lay_weights = filter_var_init[lay_id]
                else:
                    raise TypeError
            else:
                lay_weights = get_init_conv_weights(img_obs, layconf.n_filters,
                                                    n_filts_in=n_filts_in,
                                                    std=weight_init_std,
                                                    kmeans=kmeans_filter_init,
                                                    stride=local_config.factors.recon.stride,
                                                    seed=(weight_init_seed + lay_id ** 2),  # Ensure all layers have diff init weights
                                                    ksize=local_config.factors.recon.ksize,
                                                    zero_centre=False)

            print(f'coeff init seeed: {(coeff_init_seed - lay_id ** 2) * batch_id % np.iinfo(np.int32).max + coeff_init_seed}')
            conv_layer = \
                init_conv_layer(filters_init=lay_weights,
                                config=local_config,
                                inputvars=layers[-1].coeff_vars if lay_id > 0 else None,
                                img_noisy=img_obs if lay_id == 0 else None,
                                img_gt=img_gt if lay_id == 0 and img_gt is not None else None,
                                layer_id=lay_id,
                                coeff_init=coeff_init[lay_id] if isinstance(coeff_init, Iterable) else coeff_init,
                                coeff_padding=config.architecture[lay_id + 1].padding if len(config.architecture) > lay_id + 1 else 0,
                                padding=config.architecture[lay_id].padding,
                                pix_init=pix_var_init,
                                img_mask=img_mask,
                                n_filts=layconf.n_filters,
                                ksize=local_config.factors.recon.ksize,
                                stride=local_config.factors.recon.stride,
                                coeff_init_seed=(coeff_init_seed - lay_id ** 2) * batch_id % np.iinfo(np.int32).max + coeff_init_seed,  # Diff coeff seed for each batch and layer
                                fac_to_var_chunksize=local_config.factors.recon.fac_to_var_chunksize or 1,
                                layer_prev=layprev)
            if config.experiment.fix_pixels and lay_id == 0:
                conv_layer.fix_inputs()
            layers.append(conv_layer)

            n_filts_in = layconf.n_filters
            height, width = get_n_strides(ksize=local_config.factors.recon.ksize,
                                          stride=local_config.factors.recon.stride,
                                          img_width=width,
                                          img_height=height)

        elif layconf.type == 'input_obs':
            input_obs_layer = init_input_obs_layer(img_noisy=img_obs,
                                                   config=local_config,
                                                   img_mask=img_mask)
            layers.append(input_obs_layer)

        elif layconf.type == 'dense':
            ndim_in = n_filts_in * height * width
            layoutdim = layconf.outdim
            if filter_var_init is None:
                lay_weights = get_init_dense_weights(indim=ndim_in,
                                                     outdim=layoutdim,
                                                     std=weight_init_std,
                                                     seed=(weight_init_seed + lay_id ** 2))
            else:
                lay_weights = filter_var_init[lay_id]
            dense_layer = init_dense_layer(config=local_config,
                                           weights_init=lay_weights,
                                           input_vars=layers[-1].coeff_vars,
                                           coeff_init_seed=(coeff_init_seed - lay_id ** 2) * batch_id % np.iinfo(np.int32).max,
                                           fac_to_var_chunksize=local_config.factors.dense.fac_to_var_chunksize or 1,
                                           layer_prev=layprev)
            layers.append(dense_layer)
            n_filts_in = lay_weights.shape[1]
            height = 1
            width = 1

        elif layconf.type.endswith('_pool'):
            pool_config = local_config.factors.pool.copy()

            # Get initial linearisation point for coeffs of prev layer
            if isinstance(layers[-1], GBPConvLayer):
                input_init = layers[-1].filter_factor.var0[2]
            else:
                raise ValueError(f'Layer preceeding pooling layer with id {lay_id} not supported.')
            pool_layer = init_pool_layer(input_vars=layers[-1].coeff_vars,
                                         pool_factor_config=pool_config,
                                         coeff_prior_factor_config=config.factors.coeff_prior,
                                         momentum=config.momentum,
                                         pool_type=layconf.type.split('_pool')[0],
                                         input_init=input_init,
                                         layer_prev=layprev)
            layers.append(pool_layer)
            assert height % pool_config.ksize == 0
            assert width % pool_config.ksize == 0
            height //= pool_config.ksize
            width //= pool_config.ksize

        elif layconf.type == 'upsample':
            upsample_config = local_config.factors.upsample.copy()

            pool_layer = init_upsample_layer(input_vars=layers[-1].coeff_vars,
                                             upsample_factor_config=upsample_config,
                                             coeff_prior_factor_config=config.factors.coeff_prior,
                                             momentum=config.momentum,
                                             dropout=config.dropout,
                                             coeff_init_seed=(coeff_init_seed + lay_id ** 2) * batch_id % np.iinfo(
                                                 np.int32).max + coeff_init_seed,
                                             coeff_init_std=config.init_coeff_std,
                                             layer_prev=layprev)
            layers.append(pool_layer)
            assert height % upsample_config.ksize == 0
            assert width % upsample_config.ksize == 0
            height //= upsample_config.ksize
            width //= upsample_config.ksize

        elif layconf.type == 'softmax_class_obs':
            softmax_layer = \
                init_classification_layer(input_vars=layers[-1].coeff_vars,
                                          input_prior_factor=layers[-1].coeff_prior_factor,
                                          momentum=config.momentum,
                                          class_label=output_obs,
                                          config=local_config.factors.softmax_class_obs,
                                          classifier_type='softmax',
                                          layer_prev=layprev,
                                          classes_sub=classes_sub)
            layers.append(softmax_layer)
        elif layconf.type == 'regression_obs':
            regression_layer = \
                init_regression_obs_layer(input_vars=layers[-1].coeff_vars,
                                          y_obs=output_obs,
                                          config=local_config.factors.regression_obs)
            layers.append(regression_layer)
        else:
            raise ValueError(f"layconf.type={layconf.type} is not recognised.")

    for layid in range(1, len(layers)):
        layers[layid].link_layer_before(layers[layid - 1])

    return layers


def get_vartype_marginals(model, vartype: [str, list, tuple] = None):
    weights = []
    if isinstance(vartype, str):
        vartype = (vartype,)
    for lay in model.layers:
        lvars = None
        for v in vartype:
            if hasattr(lay, v):
                lvars = getattr(lay, v)
                break
        if lvars is not None:
            weights.append(dotdict(eta=lvars.eta, Lambda=lvars.Lambda))
    return weights


def init_model(config, x,
               y=None,
               model_to_reinit=None,
               track_edges=False,
               prec_rescaling=1.,
               prec_rescaling_conv_only=True,
               n_iter=None,
               force_new_graph=False,
               classes_sub=None,
               batch_id=0,
               input_mask=None):
    print(f'init model, weight seed {config.experiment.weight_init_seed} coeff seed {config.experiment.coeff_init_seed}')
    if config.factors.recon.pairwise:
        config.factors.recon.use_bias = False

    use_bias = config.factors.recon.use_bias
    if config.factors.dense is not None:
        use_bias = use_bias or config.factors.dense.use_bias
    if model_to_reinit is not None and config.experiment.copy_lin_points:
        old_lin_points = [lay.linearisation_points for lay in model_to_reinit.layers]

    if model_to_reinit is not None and config.experiment.do_filtering:
        old_posteriors = dict()
        old_posteriors['weights'] = get_vartype_marginals(model_to_reinit, ('weight_vars', 'filter_vars'))
        if use_bias:
            old_posteriors['biases'] = get_vartype_marginals(model_to_reinit, ('bias_vars',))
        old_posteriors['coeffs'] = get_vartype_marginals(model_to_reinit, ('coeff_vars',))

    if model_to_reinit is not None:
        # Force new graph if different batch size
        old_bs = model_to_reinit.layers[0].pixel_obs_factor.var_edges.shape[0]
        new_bs = x.shape[0]
        if old_bs != new_bs:
            force_new_graph = True

    is_classification = config.experiment.with_gbp_softmax_layer
    is_regression = config.architecture[-1].type == 'regression_obs'
    layers = init_layers(img_obs=x,
                         config=config,
                         weight_init_std=config.init_weight_std,
                         weight_init_seed=config.experiment.weight_init_seed,
                         coeff_init_seed=config.experiment.coeff_init_seed,
                         output_obs=y if is_regression or is_classification else None,
                         layers_prev=None if model_to_reinit is None or force_new_graph else model_to_reinit.layers,
                         classes_sub=classes_sub,
                         batch_id=batch_id,
                         img_mask=input_mask)

    gbp_net = GBPLearner(layers=layers,
                         layer_schedule=config.layer_schedule,
                         random_layer_schedule=config.random_layer_schedule)

    if model_to_reinit is not None:
        # if config.experiment.full_model_init:
        #     return prev_posterior
        if config.experiment.copy_lin_points:
            copy_linearisation_points(to_copy=old_lin_points,
                                      gbplearner_new=gbp_net)

        if config.experiment.do_filtering:
            def get_prior_sigma(layer, vartype='weight'):
                assert vartype in ('weight', 'bias')
                if hasattr(layer, 'use_bias'):
                    if layer.use_bias:
                        if isinstance(layer, GBPDenseLayer):
                            return getattr(layer, f'{vartype}_prior_factor').sigma
                        elif isinstance(layer, GBPConvLayer):
                            vtype = 'filter' if vartype == 'weight' else vartype
                            return getattr(layer, f'{vtype}_prior_factor').sigma

            # Add prev posterior over weights and biases as prior
            filtering_on_weights(old_posteriors['weights'], gbp_net,
                                 prec_rescale_factor=prec_rescaling,
                                 prec_rescale_conv_only=prec_rescaling_conv_only,
                                 alpha=config.experiment.filtering_alpha,
                                 init_prior_sigmas=[get_prior_sigma(lay, 'weight') for lay in gbp_net.layers])

            if config.experiment.filter_biases and use_bias:
                filtering_on_biases(old_posteriors['biases'], gbp_net,
                                    prec_rescale_factor=prec_rescaling,
                                    prec_rescale_conv_only=prec_rescaling_conv_only,
                                    alpha=config.experiment.filtering_alpha,
                                    init_prior_sigmas=[get_prior_sigma(lay, 'bias') for lay in gbp_net.layers]
                                    )

            if config.experiment.filter_coeffs:
                filtering_on_coeffs(old_posteriors['coeffs'], gbp_net,
                                    prec_rescale_factor=prec_rescaling)

            if config.experiment.fix_coeffs_after_filtering:
                for l in gbp_net.layers:
                    l.update_coeff_marginals()
                gbp_net.fix_layer_coeffs()

        if not force_new_graph:
            gbp_net.compiled = model_to_reinit.compiled

    if config.experiment.fix_pixel_channels:
        n_img_chans = x.shape[-1]
        gbp_net.layers[0].fixed_first_n_channels = n_img_chans

    if track_edges:
        set_edge_tracking(model=gbp_net,
                          track_msg_diffs=config.experiment.track_msg_diffs,
                          record_all_msg_diffs=config.experiment.record_all_msg_diffs,
                          n_iters=(n_iter or config.n_iters_per_train_batch) * 2 + 1)  # x 2 because each GPB iter sweeps from input to output then back again
    return gbp_net


def set_edge_tracking(model, track_msg_diffs, record_all_msg_diffs, n_iters=None):
    if track_msg_diffs:
        for lay in model.layers:
            for ed in lay.edges:
                ed.track_msg_diffs = True
                ed.record_all_diffs = record_all_msg_diffs
                ed.n_iters = n_iters
                ed.init_msg_diff_recorder()


def update_edge_tracking(model, edge_stats=None):
    """Will append to edge_stats if given else create new data structures"""
    if edge_stats is None:
        edge_stats = {}
        for l_id, lay in enumerate(model.layers):
            edge_stats[l_id] = {}
            for e, ed in enumerate(lay.edges):
                edge_stats[l_id][str(ed)] = {}
                for m_id, mtype in enumerate(ed.msg_diffs):
                    edge_stats[l_id][str(ed)][mtype] = ed.msg_diffs[mtype]
    else:
        for l_id, lay in enumerate(model.layers):
            for e, ed in enumerate(lay.edges):
                for m_id, mtype in enumerate(ed.msg_diffs):
                    if str(ed) in edge_stats[l_id]:
                        edge_stats[l_id][str(ed)][mtype] = \
                            tf.concat([edge_stats[l_id][str(ed)][mtype], ed.msg_diffs[mtype]], axis=0)
                    else:
                        # Batch size may have changed
                        # Remove bs from id str and then check if they match
                        old_en_match = match_edgenames_without_bs(new_en=str(ed), old_ens=list(edge_stats[l_id].keys()))
                        edge_stats[l_id][old_en_match][mtype] = \
                            tf.concat([edge_stats[l_id][old_en_match][mtype], ed.msg_diffs[mtype]], axis=0)

    return edge_stats
