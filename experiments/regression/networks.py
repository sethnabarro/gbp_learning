# coding=utf-8
from core.utils.utils import dotdict as dd

two_layer_mlp_16 = [
                    dd(name='input_obs1', type='input_obs'),
                    dd(name='dense1', type='dense', outdim=16, factors=dd(dense=dd(sigma=5e-3, nonlin='sigmoid'))),
                    dd(name='dense2', type='dense', outdim=1, factors=dd(dense=dd(sigma=1e-2, nonlin='none'))),
                    dd(name='output_obs1', type='regression_obs'),]

two_layer_mlp_12 = [
                    dd(name='input_obs1', type='input_obs'),
                    dd(name='dense1', type='dense', outdim=12, factors=dd(dense=dd(sigma=5e-3, nonlin='sigmoid'))),
                    dd(name='dense2', type='dense', outdim=1, factors=dd(dense=dd(sigma=1e-2, nonlin='none'))),
                    dd(name='output_obs1', type='regression_obs'),]

architectures = {
    'two_layer_mlp_16': two_layer_mlp_16,
    'two_layer_mlp_12': two_layer_mlp_12,
}


def get_network(net_name, conv_sigmas=None, last_coeff_prior_sigma=None, last_weight_prior_sigma=None, dense_coeff_prior_sigmas=None, dense_coeff_prior_N_robs=None):
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

    if dense_coeff_prior_sigmas is not None:
        # Set the strength of recon factor in each layer
        den_lay_id = 0
        for layconf in net:
            if layconf.type == 'dense':
                if 'factors' not in layconf:
                    layconf.factors = dd()
                if 'coeff_prior' not in layconf.factors:
                    layconf.factors.coeff_prior = dd()
                layconf.factors.coeff_prior.sigma = dense_coeff_prior_sigmas[den_lay_id]
                den_lay_id += 1

    if dense_coeff_prior_N_robs is not None:
        # Set the strength of recon factor in each layer
        den_lay_id = 0
        for layconf in net:
            if layconf.type == 'dense':
                if 'factors' not in layconf:
                    layconf.factors = dd()
                if 'coeff_prior' not in layconf.factors:
                    layconf.factors.coeff_prior = dd()
                layconf.factors.coeff_prior.N_rob = dense_coeff_prior_N_robs[den_lay_id]
                den_lay_id += 1

    dense_to_update = (('coeff_prior', last_coeff_prior_sigma),) if last_coeff_prior_sigma is not None else ()
    dense_to_update += (('weight_prior', last_weight_prior_sigma),) if last_weight_prior_sigma is not None else ()
    if len(dense_to_update):
        layconf = net[-2]
        if layconf.type == 'dense':
            if 'factors' not in layconf:
                layconf.factors = dd()
            for fact_type, sigma in dense_to_update:
                if fact_type not in layconf.factors:
                    setattr(layconf.factors, fact_type, dd())
                getattr(layconf.factors, fact_type).sigma = sigma

    return net
