# coding=utf-8
import matplotlib.pyplot as plt
import tensorflow as tf


def combine_missing_region(bool_mask, inpaint, pixels):
    mask_cast = tf.cast(bool_mask, tf.float32)

    pixels_shp = pixels.shape
    mask_shp = inpaint.shape
    starty = tf.math.argmax(tf.reduce_sum(mask_cast, axis=2), axis=1)[0, 0]
    startx = tf.math.argmax(tf.reduce_sum(mask_cast, axis=1), axis=1)[0, 0]
    paddings = [[0, 0],
                [starty, pixels_shp[1] - starty - mask_shp[1]],
                [startx, pixels_shp[2] - startx - mask_shp[2]],
                [0, 0]]
    return pixels * (1. - mask_cast) + tf.pad(inpaint, paddings)


def langevin_sampling(gbp_net,
                      pixels_init,
                      components_init=None,
                      weights_init=None,
                      stepsize=0.01,
                      noise_std=0.01,
                      n_iters=100,
                      pixels_mask=None):
    pixels = pixels_init
    weights = weights_init
    components = components_init
    n_layers = len(gbp_net.layers)

    if pixels_mask is not None:
        pixels_mask = tf.broadcast_to(pixels_mask, pixels_init.shape)

    all_vars = {'inputs': pixels, 'weights': weights, 'components': components}
    active_vars = {}
    for k, v in all_vars.items():
        if isinstance(v, tf.Tensor):
            if k == 'inputs':   # Pixel vars
                if f'layer_0' not in active_vars:
                    active_vars[f'layer_0'] = {}
                print(pixels_mask.shape, tf.reduce_sum(tf.cast(pixels_mask, tf.float32)), v.shape)
                mask_sum2 = tf.cast(tf.minimum(tf.reduce_sum(tf.cast(pixels_mask, tf.int32), axis=2), 1), tf.bool)
                mask_sum1 = tf.cast(tf.minimum(tf.reduce_sum(tf.cast(pixels_mask, tf.int32), axis=1), 1), tf.bool)
                vmask = tf.boolean_mask(tf.boolean_mask(v, mask_sum2[0, ..., 0], axis=1), mask_sum1[0, ..., 0], axis=2) if pixels_mask is not None else v
                active_vars['layer_0'][k] = tf.Variable(vmask)
        elif isinstance(v, (list, tuple)):
            for layid, vlayer in enumerate(v):
                if f'layer_{layid}' not in active_vars:
                    active_vars[f'layer_{layid}'] = {}
                active_vars[f'layer_{layid}'][f'{k}'] = tf.Variable(vlayer)
        elif v is None:
            pass
    for lay in range(n_layers):
        # Ensure inputs to next layer are weights (activations) from last layer
        if lay == 0:
            continue
        active_vars[f'layer_{lay}']['inputs'] = \
            active_vars[f'layer_{lay - 1}']['weights']

    for itr in range(n_iters):
        with tf.GradientTape() as tape:
            active_vars_copy = active_vars.copy()
            if pixels_mask is not None:
                active_vars_copy['layer_0'] = active_vars['layer_0'].copy()
                active_vars_copy['layer_0']['inputs'] = combine_missing_region(pixels_mask, active_vars_copy['layer_0']['inputs'], pixels_init)
            E_dict = gbp_net.energy(sum_all=False, **active_vars_copy)
            if pixels_mask is None:
                # Not inpainting
                E_dict['layer_0'].pop('pixel_obs')
            E = tf.reduce_sum(list([tf.reduce_sum(list(d.values())) for d in E_dict.values()]))
        if itr % 200 == 0:
            print(f"Iter: {itr}, energies:", {lay: {en: ee.numpy() for en, ee in e.items()} for lay, e in E_dict.items()})
        all_vars = [v for vv in active_vars.values() for v in [vv[vnam] for vnam in ['inputs', 'weights', 'components'] if vnam in vv]]
        E_grads = tape.gradient(E, all_vars)
        for v_id, v in enumerate(all_vars):
            new_var = v - E_grads[v_id] * stepsize + tf.random.normal(mean=0.,
                                                                       stddev=noise_std,
                                                                       shape=v.shape)
            v = 0
            for lay in range(n_layers):
                for vnam in ['inputs', 'weights', 'components']:
                    if vnam in active_vars[f'layer_{lay}']:
                        if v == v_id:
                            # if lay == 0 and \
                            #         pixels_mask is not None and \
                            #         vnam == 'inputs':
                            #     # Only change value of pixels in inpainting region
                            #     new_var = tf.where(pixels_mask, new_var, active_vars[f'layer_{lay}'][vnam])
                            active_vars[f'layer_{lay}'][vnam].assign(new_var)
                        v += 1
        for lay in range(n_layers):
            # Ensure inputs to next layer are weights (activations) from last layer
            if lay == 0:
                continue

            tf.assert_equal(active_vars[f'layer_{lay}']['inputs'], active_vars[f'layer_{lay - 1}']['weights'])

    return combine_missing_region(pixels_mask, active_vars['layer_0']['inputs'], pixels_init),  # [active_vars[vname] for vname in ['pixels', 'weights', 'components'] if vname in active_vars]


def mcmc_sample_from_factor_graph(img_shp,
                                  factgraph,
                                  coeff_init_std,
                                  sampling_config,
                                  img_mask=None):

    pixels_init = tf.random.uniform([sampling_config['sampling_batch_size']] + list(img_shp)[1:]) if sampling_config['sample_pixels'] is not None else None
    if img_mask is not None:
        img_orig = factgraph.layers[0].pixel_obs_factor.obs
        img_mask_cast = tf.cast(img_mask, tf.float32)
        pixels_init = pixels_init * img_mask_cast + img_orig * (1. - img_mask_cast)
        plt.imshow(pixels_init[0, ..., 0], interpolation='none')
        plt.show()
    if sampling_config['sample_components']:
        components_init = []
        for lay in factgraph.layers:
            compsampshp = [sampling_config['sampling_batch_size']] + lay.component_vars.mu.shape.as_list()[1:]
            compinit = tf.random.normal(stddev=0.2, shape=compsampshp)
            components_init.append(compinit)
    else:
        components_init = None
    if sampling_config['sample_weights']:
        weight_init = []
        for lay in factgraph.layers:
            weightinit_lay = tf.random.normal(stddev=coeff_init_std,
                                              shape=[sampling_config['sampling_batch_size']] +
                                                     lay.weight_vars.shape.as_list()[1:])
            weight_init.append(weightinit_lay)
    else:
        weight_init = None
    pixels_sample = \
        langevin_sampling(factgraph,
                          components_init=components_init,
                          weights_init=weight_init,
                          pixels_init=pixels_init,
                          stepsize=sampling_config['stepsize'],
                          n_iters=sampling_config['n_iters'],
                          noise_std=sampling_config['noise_std'],
                          pixels_mask=img_mask)[0]
    plot_pixel_samples(pixels_init, pixels_sample, plotdir=sampling_config['sample_plotfile'])
    return tf.convert_to_tensor(pixels_sample)


def plot_pixel_samples(init, sample, plotdir=None):
    nchan = init.shape[-1]
    is_colour = nchan == 3

    for i in range(init.shape[0]):
        fig, axs = plt.subplots(1, 2)
        init_i = init[i, 3:-3, 3:-3] if is_colour else init[i, 3:-3, 3:-3, 0]
        sample_i = sample[i, 3:-3, 3:-3] if is_colour else sample[i, 3:-3, 3:-3, 0]
        imsample = axs[1].imshow(sample_i, interpolation='none')
        iminit = axs[0].imshow(init_i, interpolation='none')
        for a in axs:
            a.set_xticks([])
            a.set_yticks([])
        fig.colorbar(imsample, ax=axs[1], fraction=0.02, pad=0.04)
        fig.colorbar(iminit, ax=axs[0], fraction=0.02, pad=0.04)
        plt.tight_layout()
        plt.suptitle(f'Sample {i}')

        if plotdir is None:
            plt.show()
        else:
            plt.savefig(f'{plotdir}/mcmc_sample{i}.png')