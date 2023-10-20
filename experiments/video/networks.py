# coding=utf-8
from core.utils.utils import dotdict as dd

conv_tpose_single_layer_perpixel_4 = [dd(name='conv1', type='conv', n_filters=4, factors=dd(recon=dd(tpose=True, nonlin='none')))]
conv_tpose_five_layer_upsample_perpixel_4 = \
    [dd(name='conv1', type='conv', n_filters=4, factors=dd(recon=dd(tpose=True, nonlin='none'))),
     dd(name='ups1', type='upsample'),
     dd(name='conv2', type='conv', n_filters=8, factors=dd(recon=dd(N_rob=None, tpose=True, nonlin='leaky_relu'))),
     dd(name='ups2', type='upsample'),
     dd(name='conv3', type='conv', n_filters=8, factors=dd(recon=dd(N_rob=None, tpose=True, nonlin='leaky_relu')))]


architectures = {'conv_tpose_single_layer_4': conv_tpose_single_layer_perpixel_4,
                 'conv_tpose_five_layer_upsample_4': conv_tpose_five_layer_upsample_perpixel_4,
                 'pairwise': [dd(name='pwise', type='conv', n_filters=0)]}


def get_network(net_name, conv_sigmas=None, dense_coeff_prior_sigma=None, dense_weight_prior_sigma=None,
                conv_coeff_prior_sigmas=None, conv_coeff_prior_N_robs=None, conv_weight_prior_sigmas=None):
    net = architectures[net_name]

    if conv_sigmas is not None:
        # Set the strength of recon factor in each layer
        conv_lay_id = 0
        for layconf in net:
            if layconf.type == 'conv':
                if 'factors' not in layconf:
                    layconf.factors = dd()
                if 'recon' not in layconf.factors:
                    layconf.factors.recon = dd()
                layconf.factors.recon.sigma = conv_sigmas[conv_lay_id]
                conv_lay_id += 1

        assert conv_lay_id == len(conv_sigmas), \
            f"Length of recon factor sigmas {conv_sigmas} " \
            f"does not match number of conv recon layers in network {net_name}: " \
            f"{len(conv_sigmas)} != {conv_lay_id}"

    if conv_weight_prior_sigmas is not None:
        # Set the strength of recon factor in each layer
        conv_lay_id = 0
        for layconf in net:
            if layconf.type == 'conv':
                if 'factors' not in layconf:
                    layconf.factors = dd()
                if 'weight_prior' not in layconf.factors:
                    layconf.factors.weight_prior = dd()
                layconf.factors.weight_prior.sigma = conv_weight_prior_sigmas[conv_lay_id]
                conv_lay_id += 1

    if conv_coeff_prior_sigmas is not None:
        # Set the strength of recon factor in each layer
        conv_lay_id = 0
        for layconf in net:
            if layconf.type == 'conv':
                if 'factors' not in layconf:
                    layconf.factors = dd()
                if 'coeff_prior' not in layconf.factors:
                    layconf.factors.coeff_prior = dd()
                layconf.factors.coeff_prior.sigma = conv_coeff_prior_sigmas[conv_lay_id]
                conv_lay_id += 1

    if conv_coeff_prior_N_robs is not None:
        # Set the strength of recon factor in each layer
        conv_lay_id = 0
        for layconf in net:
            if layconf.type == 'conv':
                if 'factors' not in layconf:
                    layconf.factors = dd()
                if 'coeff_prior' not in layconf.factors:
                    layconf.factors.coeff_prior = dd()
                layconf.factors.coeff_prior.N_rob = conv_coeff_prior_N_robs[conv_lay_id]
                conv_lay_id += 1

    dense_to_update = (('coeff_prior', dense_coeff_prior_sigma),) if dense_coeff_prior_sigma is not None else ()
    dense_to_update += (('weight_prior', dense_weight_prior_sigma),) if dense_weight_prior_sigma is not None else ()
    if len(dense_to_update):
        for layconf in net:
            if layconf.type == 'dense':
                if 'factors' not in layconf:
                    layconf.factors = dd()
                for fact_type, sigma in dense_to_update:
                    if fact_type not in layconf.factors:
                        setattr(layconf.factors, fact_type, dd())
                    getattr(layconf.factors, fact_type).sigma = sigma

    return net
