# coding=utf-8
from copy import deepcopy
import shutil
from functools import partial
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import tensorflow as tf

# from core.bm3d_noise_utils import get_experiment_noise

mpl_colors = list(mpl.colors.CSS4_COLORS)
np.random.seed(9)
np.random.shuffle(mpl_colors)
mpl.rcParams['image.cmap'] = 'gray'
imshow = partial(plt.imshow, interpolation='none')


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    https://stackoverflow.com/a/23689767/7785309
    https://stackoverflow.com/questions/49901590/python-using-copy-deepcopy-on-dotdict
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def from_dict(dct):
        dd = dotdict()
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = dotdict.from_dict(value)
            elif isinstance(value, (tuple, list)):
                value = type(value)([dotdict.from_dict(v) if hasattr(v, 'keys') else v for v in value])

            dd[key] = value
        return dd

    def __deepcopy__(self, memo=None):
        return dotdict(deepcopy(dict(self), memo=memo))

    def copy(self):
        d = super().copy()
        return dotdict(**d)


def get_filter_factor_recon(layer):
    filt_axis_parl_fac = -3 if layer.use_decomp_filter_factors else -2
    margs = {}
    n_pix_channels = layer.input_vars.shape[-1]
    for ptype in ('eta', 'Lambda'):
        msg_from_filtfacs = getattr(layer.filter_factor.input_var_edges,
                                    f'fac_to_var_{ptype}')[:, :, :, :n_pix_channels]
        if not layer.use_sum_filter_factors:
            msg_from_filtfacs = tf.reduce_sum(msg_from_filtfacs, axis=filt_axis_parl_fac)
        margs[ptype] = layer.get_input_marginal(msg_from_filtfacs, tf.zeros(layer.input_vars.shape))
    pix_mu = margs['eta'] / margs['Lambda']
    return pix_mu


def get_init_conv_weights(img, n_filters=2, std=1., seed=898, n_filts_in=3,
                          ksize=3, kmeans=False, stride=1, zero_centre=True):
    np.random.seed(seed)
    if kmeans:
        img_for_kmeans = img  # load_image(img, H=imgH, W=imgW, normalise_image=True)
        filts = get_ground_truth_filters(img_for_kmeans, n_filters,
                                         additive_factor=True,
                                         kernsize=ksize,
                                         add_flat_filter=False,
                                         stride=stride,
                                         seed=766).numpy()  # method='get_unique'
    else:
        filtshp = (n_filts_in, n_filters, ksize ** 2)
        filts = np.random.normal(loc=0., scale=std, size=filtshp).astype(np.float32)
        if zero_centre:
            filts[..., int(ksize ** 2 / 2)] = 0.  # filter factors encode difference between central pixel and neighbours
        filts = make_tf_filter_shp(filts, ksize)
    return filts


def get_init_dense_weights(indim, outdim, std=1., seed=898):
    np.random.seed(seed)
    weightshp = (indim, outdim)
    return np.random.normal(0., std, size=weightshp).astype(np.float32)


def get_n_strides(stride, ksize, img_in=None, img_height=None, img_width=None):
    assert (img_in is not None) or ((img_height is not None) and (img_width is not None))
    H, W = img_in.shape[1:3] if img_in is not None else (img_height, img_width)
    n_filt_y = (H - ksize) / stride + 1
    n_filt_x = (W - ksize) / stride + 1
    assert n_filt_x.is_integer() and n_filt_y.is_integer()
    return int(n_filt_x), int(n_filt_y)


class FixedAttributeFieldsObject(object):
    """
    Raises error when non-existent attribute is
    set after fields have been fixed
    """
    _attr_fields_fixed = False

    def fix_attr_fields(self):
        self._attr_fields_fixed = True

    def __setattr__(self, name, value):
        if hasattr(self, name) or not self._attr_fields_fixed:
            super(FixedAttributeFieldsObject, self).__setattr__(name, value)
        else:
            raise AttributeError(f'"{type(self).__name__}" object has no '
                                 f'attribute "{name}", so cannot set it.')


def make_tf_filter_shp(filts, ksize):
    inchannels = filts.shape[0]
    filter_shp_tf = (ksize, ksize, inchannels, -1)
    return tf.reshape(tf.transpose(filts, (2, 0, 1)), filter_shp_tf)


def flatten_filters(param):
    ksize = param.shape[0]
    inchannels = param.shape[2]

    prm_rs = tf.reshape(param, shape=(ksize ** 2, inchannels, -1))
    return tf.transpose(prm_rs, perm=(1, 2, 0))


def patchify_image(image, ksize_x=3, ksize_y=None, stride=1):
    """
    Patchifies image
    Get vector of pixel intensities for
    each neighbourhood of pixels"""
    in_channels = image.shape[-1]
    ksize_y = ksize_y or ksize_x
    patch_opts = dict(sizes=[1, ksize_y, ksize_x, 1],
                      strides=[1, stride, stride, 1],
                      rates=[1, 1, 1, 1],
                      padding='VALID')
    patches = tf.image.extract_patches(image, **patch_opts)
    patches = tf.transpose(tf.reshape(patches, patches.shape[:3] + [ksize_y * ksize_x, in_channels]), [0, 1, 2, 4, 3])
    # patches = tf.reshape(patches, patches.shape[:3] + [in_channels, ksize_y * ksize_x])
    return patches


def peak_snr(img_clean, img_denoised, mask=None, normalised_img=True, clip_img=True, crop_border=0):
    """See https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio"""
    max_intensity = 1. if normalised_img else 255.
    mse_score = mse(img_clean, img_denoised, mask=mask, clip_img=clip_img, crop_border=crop_border)
    ratio = max_intensity ** 2. / mse_score
    return 10. * tf.math.log(ratio) / tf.math.log(10.)  # convert to base 10


def mse(img_clean, img_denoised, mask, crop_border=0, clip_img=True):
    if clip_img:
        img_denoised = tf.clip_by_value(img_denoised, 0., 1.)
    diff = (img_clean - img_denoised)
    if crop_border > 0:
        cb = crop_border
        diff = (img_clean - img_denoised)[:, cb:-cb, cb:-cb]
    if mask is not None:
        if crop_border > 0:
            mask = mask[:, cb:-cb, cb:-cb]
        mask_bc = tf.broadcast_to(mask, diff.shape)
        msqe = tf.reduce_mean(tf.boolean_mask(diff ** 2., tf.logical_not(mask_bc)))
    else:
        msqe = tf.reduce_mean(diff ** 2.)
    return msqe


def denoise_eval(denoised, frame_clean, mask=None, crop_border=0):
    psnr = peak_snr(frame_clean, denoised.mu, mask=mask, crop_border=crop_border)
    mse_score = mse(frame_clean, denoised.mu, mask=mask, crop_border=crop_border)
    ll_elems = diag_gauss_logpdf(frame_clean, denoised.mu, denoised.sigma)
    if mask is not None:
        ll_elems *= (1. - tf.cast(mask, ll_elems.dtype))
    if crop_border > 0:
        cb = crop_border
        ll_elems = ll_elems[:, cb:-cb, cb:-cb]
    ll = tf.reduce_sum(ll_elems)
    return psnr, ll, mse_score


def diag_gauss_logpdf(x, mu, std):
    dim = mu.shape[-1]
    lognorm = -(tf.math.log(2. * np.pi) + 2. * tf.math.log(std)) * 0.5
    return lognorm - 0.5 * ((x - mu) / std) ** 2.


def get_pairwise_cossim(filters, get_avg=False):
    filters_flat = flatten_filters(filters)
    filters_flat = tf.reshape(tf.transpose(filters_flat, (1, 0, 2)), (-1, filters_flat.shape[-1]))
    fTf = tf.reduce_sum(filters_flat[None] * filters_flat[:, None], axis=-1)
    fnorm = tf.norm(filters_flat, axis=-1)
    fnormfnormT = fnorm[None] * fnorm[:, None]
    cossim = fTf / fnormfnormT
    if get_avg:
        cossim = cossim.numpy()[np.triu_indices(cossim.shape[0], k=1)]
        cossim = tf.reduce_mean(tf.abs(cossim))
    return cossim


def load_image(image_path, H=None, W=None, ypixstart=0, xpixstart=0, normalise_image=True, plot_image=True, colour=False, scale_factor=1.):
    img = Image.open(image_path)#.resize((200, 200))
    if scale_factor != 1.:
        img = img.resize((int(img.size[0] * scale_factor), int(img.size[1] * scale_factor)))

    image = np.array(img).astype(float)

    # image = image[..., None]
    if image.ndim > 2 and not colour:
        image = np.mean(image, -1)
    if H is not None and W is not None:
        image = image[ypixstart:H + ypixstart, xpixstart:W + xpixstart]
    else:
        image = image[ypixstart:, xpixstart:]

    if normalise_image:
        image -= np.min(image)
        image /= np.max(image)
    image = np.expand_dims(image, 0)
    if colour:
        image = image[..., :3]   # remove fourth (transparency) channel if present
    else:
        image = np.expand_dims(image, -1)
    if plot_image:
        implot = image[0] if colour else image[0, ..., 0]
        imshow(implot)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    return image.astype(np.float32)


def load_video(video_path, H=None, W=None, normalise_video=True):
    """Loads all image in directory to array"""
    def resize(img, h, w):
        have_h_rs = h is not None
        have_w_rs = w is not None
        w_orig, h_orig = img.size
        if have_h_rs and not have_w_rs:
            w = int(h / h_orig * w_orig)
        if have_w_rs and not have_h_rs:
            h = int(w / w_orig * h_orig)
        print(f'Resizing to {(w, h)}')
        return img.resize((w, h), Image.BILINEAR)

    vid_array = np.array([])
    img_id = 0
    for f in sorted(os.listdir(video_path)):
        if f.endswith(('.png', 'jpg')):
            print(f)
            img = Image.open(os.path.join(video_path, f))
            if H is not None or W is not None:
                img = resize(img, H, W)

            imloaded = np.array(img).astype(float)
            if img_id == 0:
                vid_array = imloaded[None]
            else:
                vid_array = np.concatenate([vid_array, imloaded[None]])
            img_id += 1
        else:
            print(f'Not loading file to video as not png: {f}')
    print(f'Loaded {img_id} frames into video array.')

    # image = image[..., None]
    is_in_colour = vid_array.ndim == 4 and vid_array.shape[-1] == 3
    # if is_in_colour:
    #     # convert to b+w
    #     vid_array = np.mean(vid_array, -1)

    # Crop
    H = H or vid_array.shape[1]
    W = W or vid_array.shape[2]
    vid_array = vid_array[:, :H, :W]

    if normalise_video:
        vid_array -= np.min(vid_array)
        vid_array /= np.max(vid_array)
    vid_array = np.expand_dims(vid_array, 0)

    # if is_in_colour:
    #     vid_array = np.expand_dims(vid_array, -1)
    return vid_array.astype(np.float32)


def corrupt_image(img, frac_noise, noise_state=88, noise_seed=89,
                  noise_type=float, noise_dist='uniform', mask_size=20, border_without_mask=4, mask_topleft=None,
                  noise_channel=None, get_noise_mask=False):
    """
    Adds random [0, 1] noise to input image.
        frac_noise: fraction of pixels to be replaced by noise
    """
    allowed_noise_dists = ['mask', 'uniform', 'normal']
    stateseed = [noise_state, noise_seed]
    if noise_dist == 'mask':
        if mask_topleft is None:
            xystart = tf.random.stateless_uniform((img.shape[0], 2),
                                                 minval=border_without_mask,
                                                 maxval=img.shape[2] - border_without_mask - mask_size,
                                                 dtype=tf.int32,
                                                 seed=stateseed)
        else:
            ystart, xstart = mask_topleft
            xystart = tf.convert_to_tensor([xstart, ystart])

        w_cond = tf.logical_and(xystart[:, None, 1] <= tf.range(img.shape[2])[None],
                                tf.range(img.shape[2])[None] < xystart[:, None, 1] + mask_size)
        h_cond = tf.logical_and(xystart[:, None, 0] <= tf.range(img.shape[1])[None],
                                tf.range(img.shape[1])[None] < xystart[:, None, 0] + mask_size)
        mask_cond = tf.logical_and(w_cond[:, None], h_cond[:, :, None])[..., None]
        # img_corrupt = tf.where(mask_cond, tf.random.uniform(img.shape, minval=tf.reduce_min(img), maxval=tf.reduce_max(img)), img)
        img_corrupt = tf.where(mask_cond, 0., img)
        # img_corrupt = tf.where(block_ids_grid, tf.random.stateless_normal(img.shape, mean=0.5, stddev=0.2, seed=[noise_seed, noise_state]), img)
        # img_corrupt = tf.where(block_ids_grid, 0., img)
        return img_corrupt, tf.logical_not(mask_cond)   # Will treat True values in mask as being observed pixels
    # Generate noise with same dims as input image
    # Use seeded stateless noise generation for repeatability
    stateseed = [noise_state, noise_seed]
    mask = None
    if noise_type is bool:
        noise = tf.random.stateless_uniform(img.shape, maxval=2, dtype=tf.int32, seed=stateseed)
        noise = tf.cast(noise, tf.float32)
    elif noise_type is float:
        if noise_dist == 'uniform':
            noise = tf.random.stateless_uniform(img.shape, maxval=1., dtype=tf.float32, seed=stateseed)
        elif noise_dist.startswith('normal'):
            if len(noise_dist.split('_')) >= 2 and noise_dist.split('_')[-1].replace('.', '').isdigit():
                std = float(noise_dist.split('_')[-1])
            else:
                std = 0.3
            if 'add' in noise_dist:
                noise = img + tf.random.stateless_normal(img.shape, mean=0., stddev=std, seed=stateseed)
            else:
                noise = tf.random.stateless_normal(img.shape, mean=0.5, stddev=std, seed=stateseed)
        # elif noise_dist.startswith('bm3d'):
        #     noise_type_code, noise_std = noise_dist.split('_')[1:]
        #     noise = get_experiment_noise(noise_type=noise_type_code,
        #                                  noise_var=float(noise_std) ** 2,
        #                                  realization=stateseed[1],
        #                                  sz=img.shape[1:])[0]
        #
        #     noise = img + noise[None]

    else:
        raise ValueError(f'`noise_dist` must be in {allowed_noise_dists}, got {noise_dist}')

    # Randomly select which pixels are going to have noise added
    # by sampling boolean mask
    stateseed = [noise_state, noise_seed + 1]
    noise_mask = tf.random.stateless_uniform(img.shape[:3], maxval=1., seed=stateseed)
    noise_mask = tf.cast(noise_mask < frac_noise, tf.float32)
    mask = tf.cast(1. - noise_mask, tf.bool) if get_noise_mask else None

    if noise_channel is not None:
        nchan = img.shape[3]
        noise_mask = tf.where(tf.range(nchan) == noise_channel, noise_mask, 0.)

    # Subtract current intensities from randomly selected pixels
    img_corrupt = img - img * noise_mask[..., None]

    # Add noise in place of zeros
    img_corrupt = img_corrupt + noise_mask[..., None] * noise
    return img_corrupt, mask


# def plot_filters(filters, itr=-1, fig=None, axs=None, filter_stds=None):
#     n_filters = filters.shape[-1]
#     axs_given = axs is not None
#     if fig is None and axs is None:
#         fig, axs = plt.subplots(2, 8)
#     filters = filters.numpy()
#     ims = []
#     ts = []
#     for i in range(2):
#         for j in range(4):
#             k = i * 4 + j
#             if i * 4 + j >= n_filters:
#                 continue
#
#             rgb = np.repeat(np.array(mpl.colors.to_rgb(mpl_colors[k]))[None, None], 3, axis=0)
#
#             im = axs[i, j * 2].imshow(filters[..., 0, 4 * i + j],
#                                   vmin=np.min(filters),
#                                   vmax=np.max(filters))
#             if filter_stds is not None:
#                 t = axs[i, j * 2].text(0, -0.5, f'Avg std: {filter_stds[4 * i + j]:.4f}')
#
#             axs[i, j * 2 + 1].imshow(rgb)
#             # axs[i, j].set_clim([-np.min(filters), np.max(filters)])
#             axs[i, j * 2 + 1].set_xticks([])
#             axs[i, j * 2 + 1].set_yticks([])
#             axs[i, j * 2].set_yticks([])
#             axs[i, j * 2].set_xticks([])
#
#             ims.append(im)
#             ts.append(t)
#     if not axs_given:
#         plt.suptitle(f'Iter: {itr}')
#         fig.subplots_adjust(right=0.85)
#         cbar_ax = fig.add_axes([0.87, 0.08, 0.05, 0.7])
#         fig.colorbar(im, cax=cbar_ax)
#         # plt.tight_layout()
#         plt.show()
#     return ims, ts


def kmeans(x, additive_factor, k, n_iter=200, seed=666, ksize=3):
    if additive_factor:
        x -= x[:, int(ksize ** 2 / 2)][:, None]
    assignments = tf.random.stateless_uniform((x.shape[0],), maxval=k, dtype=tf.int32, seed=[seed, 9])
    means = tf.stack([tf.reduce_mean(tf.gather(x, tf.where(assignments == c)[:, 0]), axis=0) for c in range(k)],
                     axis=0)
    for i in range(n_iter):
        sqr_dist = tf.reduce_mean(tf.square(means[None] - x[:, None, :]), axis=-1)
        assignments = tf.argmin(sqr_dist, axis=1)
        means = tf.stack([tf.reduce_mean(tf.gather(x, tf.where(assignments == c)[:, 0]), axis=0)
                          for c in range(k)], axis=0)
    return means


def pca(x, k):
    x -= tf.reduce_mean(x, axis=0, keepdims=True)
    x /= tf.math.reduce_std(x, axis=0, keepdims=True)
    return tf.linalg.eigh(tf.tensordot(tf.transpose(x), x, axes=1))[1][-k:]


def get_ground_truth_filters(img, n_filters, kernsize,
                             additive_factor=False, plot_filters=False, stride=1,
                             add_flat_filter=False, seed=123, method=None):
    method = method or 'kmeans'

    patches = patchify_image(img.astype(np.float32), ksize_x=kernsize, stride=stride)
    patches_rs = tf.reshape(patches, [-1, kernsize ** 2])
    if method == 'kmeans':
        means = kmeans(patches_rs, additive_factor, k=n_filters, seed=seed, ksize=kernsize)
    elif method == 'get_unique':
        vectors, idx = tf.raw_ops.UniqueV2(x=patches_rs, axis=[0])
        idx_chosen = tf.argsort(tf.unique_with_counts(idx).count, direction='DESCENDING')[:n_filters]
        means = tf.gather(vectors, idx_chosen)
        means = tf.convert_to_tensor([means[0], means[1] - 1.])

    if add_flat_filter:
        # Add flat filter
        means = tf.tensor_scatter_nd_update(means, [[n_filters - 1]], np.zeros((1, kernsize ** 2)))

    if additive_factor:
        if plot_filters:
            for m in means:
                imshow(np.reshape(m, (3, 3)))
                plt.colorbar()
                plt.show()

        return tf.reshape(tf.transpose(means, (1, 0)), (kernsize, kernsize, 1, n_filters))

    # find orthogonal filters
    filters_gt = np.ones((n_filters, kernsize ** 2))
    for i, m in enumerate(means):
        filters_gt[i, i] -= 1. + np.sum([m[j] for j in range(len(m)) if j != i]) / m[i]
    filters_gt /= tf.linalg.norm(filters_gt, axis=-1, keepdims=True)
    filters_gt = tf.cast(filters_gt, dtype=tf.float32)

    if plot_filters:
        for m in filters_gt:
            imshow(np.reshape(m, (3, 3)))
            plt.show()

    return tf.reshape(tf.transpose(filters_gt, (1, 0)), (3, 3, 1, n_filters))


# def plot_filter_energies(x_img, x_img_recon, filters, N_rob, energy_fn, components=None,
#                          itr=None, plotfile=None, inc_lowest_energy_factor=False,
#                          plot_lowest_energy_map=False, filters_init=None, coeffs=None,
#                          energies=None, plot_robust_threshold=False, energy_plot_title=None,
#                          plot_robust_energies=False):
#
#     with_coeffs = coeffs is not None
#     with_robust_factor = plot_robust_threshold or ((N_rob is not None) and not with_coeffs)
#     with_components = components is not None
#     filters_plot = tf.transpose(filters, [3, 0, 1, 2]).numpy()
#     plot_init_filters = filters_init is not None
#     if plot_init_filters:
#         filters_init = tf.transpose(filters_init, [3, 0, 1, 2])[..., 0].numpy()
#     else:
#         filters_init = tf.zeros_like(filters_plot)
#     energies_rob = energy_fn(x_img, components if with_components else filters,
#                              coeffs=coeffs, aggregate=False)
#     energies_rob = tf.reduce_mean(energies_rob, axis=-2)
#     if energies is None:
#         energies = energy_fn(x_img_recon, components if with_components else filters,
#                              coeffs=coeffs, robust=False, aggregate=False)
#         energies = tf.reduce_mean(energies, axis=-2)
#     nfilt = filters.shape[-1]
#     use_sum_factors = energies.shape[-1] == 1 and nfilt != 1
#     perm = [3, 1, 2, 0]
#     energies_rob = tf.transpose(energies_rob, perm)[..., 0].numpy()
#     energies = tf.transpose(energies, perm)[..., 0].numpy()
#     energies_min = tf.reduce_min(energies, axis=0)
#     if use_sum_factors:
#         energies = np.concatenate([energies] * nfilt, axis=0)
#         energies_rob = np.concatenate([energies_rob] * nfilt, axis=0)
#
#     for i, (esr, es, fs, fs_init) in enumerate(zip(energies_rob, energies, filters_plot, filters_init)):
#         plt.close('all')
#         if inc_lowest_energy_factor or with_coeffs:
#             subplt_shp = (2, 2) if with_robust_factor or with_coeffs else (1, 3)
#             w_rat = (2, 2) if with_robust_factor or with_coeffs else (2, 3, 3)
#             fsize = (6, 5) if with_robust_factor or with_coeffs else (10, 3)
#         else:
#             subplt_shp = (1, 2 + int(with_robust_factor or with_coeffs))
#             w_rat = (2, 2, 3) if with_robust_factor or with_coeffs else (2, 2)
#             fsize = (10, 3) if with_robust_factor or with_coeffs else (6, 5)
#         fig, axs = plt.subplots(subplt_shp[0], subplt_shp[1], figsize=fsize,
#                                 gridspec_kw={'width_ratios': w_rat})
#         if not hasattr(axs[0], '__getitem__'):
#             axs = [axs]
#         # print(np.max(np.abs(fs)), np.max(np.abs(weights[0, ..., i])), 'max')
#         im = axs[0][0].imshow(fs, interpolation='none')
#         axs[0][0].set_xticks([])
#         axs[0][0].set_yticks([])
#         axs[0][0].set_title('Filter')
#         fig.colorbar(im, ax=axs[0][0], fraction=0.02, pad=0.04)
#         if plot_init_filters:
#             im = axs[0][1].imshow(fs_init, interpolation='none')
#             axs[0][1].set_xticks([])
#             axs[0][1].set_yticks([])
#             axs[0][1].set_title('Init Filter')
#         else:
#             im = axs[0][1].imshow(esr if plot_robust_energies else es, interpolation='none')
#             axs[0][1].set_xticks([])
#             axs[0][1].set_yticks([])
#             axs[0][1].set_title(energy_plot_title or (f'Energies (robust)' if plot_robust_energies else 'Energies (non-robust)'))
#         fig.colorbar(im, ax=axs[0][1], fraction=0.02, pad=0.04)
#         if with_robust_factor:
#             axrob = axs[1][0] if inc_lowest_energy_factor or with_coeffs else axs[0][2]
#             im = axrob.imshow((es >= N_rob ** 2).astype(int), interpolation='none')
#             axrob.set_xticks([])
#             axrob.set_yticks([])
#             axrob.set_title('In robust region (True/False)')
#             im.set_clim(0., 1.)
#             # fig.colorbar(im, ax=axs[2], fraction=0.02, pad=0.04)
#         if with_coeffs:
#             axrob = axs[1][1] if with_robust_factor or with_coeffs else axs[0][2]
#             im = axrob.imshow(coeffs[0, ..., i], interpolation='none')
#             axrob.set_xticks([])
#             axrob.set_yticks([])
#             fig.colorbar(im, ax=axrob, fraction=0.02, pad=0.04)
#             axrob.set_title('Factor coeff')
#         elif inc_lowest_energy_factor and not use_sum_factors:
#             axlow = axs[1][1] if with_robust_factor or with_coeffs else axs[0][2]
#             im = axlow.imshow((es == energies_min.numpy()).astype(int), interpolation='none')
#             axlow.set_xticks([])
#             axlow.set_yticks([])
#             axlow.set_title('Is min E factor (True/False)')
#             im.set_clim(0., 1.)
#
#         fig.suptitle(f'filter {i}' if itr is None else f'iter {itr}, filter {i}')
#         plt.tight_layout()
#         if plotfile is not None:
#             plt.savefig(plotfile.replace('.png', f'_filt{i}.png'), bbox_inches='tight')
#             plt.close('all')
#         else:
#             plt.show()
#     if plot_lowest_energy_map and not use_sum_factors:
#         plot_lowest_energy_filter_map(x_img_recon, filters, energy_fn, itr, coeffs=coeffs)
#
#
# def plot_recon(filters, coeffs, stride, recon_shp, plotfile, act_fn=None):
#     act_fn = act_fn or tf.identity
#     ksize = filters.shape[0]
#     filters_tp = tf.transpose(filters, (3, 0, 1, 2))
#
#     def plot_conv_recon(acts, ax, fg):
#         pixel_marg = tf.zeros(recon_shp)
#
#         stride_ids = \
#             tf.transpose(tf.meshgrid(tf.range(coeffs.shape[1] * stride, delta=stride),
#                                      tf.range(coeffs.shape[2] * stride, delta=stride)), (2, 1, 0))
#         for v in range(ksize ** 2):
#             vx, vy = v // ksize, v % ksize
#             pixel_marg = tf.tensor_scatter_nd_add(pixel_marg,
#                                                   stride_ids + [[vx, vy]],
#                                                   acts[0, :, :, vx, vy])
#         im = ax.imshow(pixel_marg, interpolation=None)
#         # im.set_clim(0., 1.0)
#         fg.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
#
#     outfilts = filters.shape[2]
#     nrow = int(np.ceil(np.sqrt(outfilts)))
#     ncol = int(np.ceil(outfilts / nrow))
#     fig, axs = plt.subplots(nrow, ncol)
#     if nrow == 1:
#         axs = [axs]
#     if ncol == 1:
#         axs = [[a] for a in axs]
#     for axx in axs:
#         for ax in axx:
#             ax.set_axis_off()
#     for o in range(outfilts):
#         wTf = - tf.reduce_sum(coeffs[..., None, None] * filters_tp[None, None, None, ..., o], axis=3)
#         r, c = o // ncol, o % ncol
#         plot_conv_recon(act_fn(wTf), ax=axs[r][c], fg=fig)
#     plt.tight_layout()
#     plt.savefig(plotfile, bbox_inches='tight')
#     plt.close('all')
#
#
# def plot_lowest_energy_filter_map(x_img_recon, filters=None, energy_fn=None,
#                                   itr=-1, with_legend=False, ax=None, energies=None,
#                                   add_title=False, coeffs=None):
#     energies = energies if energies is not None else energy_fn(x_img_recon, filters, coeffs=coeffs, robust=False, aggregate=False)
#     n_filt = energies.shape[-1] if energies is not None else filters.shape[-1]
#     energies_min = tf.reduce_min(energies, axis=-1).numpy()
#     ax_given = ax is not None
#     ax = ax or plt
#
#     # color = ['red', 'blue', 'black', 'orange', 'green', 'purple', 'yellow', 'cyan']
#     ims = []
#     for f in range(n_filt):
#         cmap = mpl.colors.ListedColormap(['white', mpl_colors[f]])
#         my_cmap = cmap(np.arange(cmap.N))
#         my_cmap[:, -1] = [0., 1.]
#         cmap = mpl.colors.ListedColormap(my_cmap)
#
#         bounds = [0, 0.5, 2.]
#         norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#
#         is_min = (energies[0, ..., f].numpy() == energies_min[0]).astype(int)
#         im = ax.imshow(is_min, cmap=cmap)
#         ims.append(im)
#         ax.scatter([], [], color=mpl_colors[f], label=str(f))
#
#     if ax_given:
#         if add_title:
#             ax.set_title(f"iter: {itr}")
#         ax.set_xticks([])
#         ax.set_yticks([])
#     else:
#         if add_title:
#             plt.title(f"iter: {itr}")
#         plt.xticks([])
#         plt.yticks([])
#         plt.tight_layout()
#     if with_legend:
#         plt.legend()
#
#     if not ax_given:
#         plt.show()
#     return ims


def plot_psnrs(psnrs):
    plt.plot(psnrs)
    plt.xlabel('Iters')
    plt.ylabel('PSNR')
    plt.show()


def bmatrix(a):
    """
    From https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix/17131750
    Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').replace('.', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    return '\n'.join(rv)


def set_allow_gpu_mem_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def tf_fn_wrapper(fn, use_static_graph=True, xla_compile=False):
    if use_static_graph:
        @tf.function(jit_compile=xla_compile)
        def _fn(*args, **kwargs):
            return fn(*args, **kwargs)
    else:
        def _fn(*args, **kwargs):
            return fn(*args, **kwargs)

    return lambda *fnargs, **fnkwargs: _fn(*fnargs, **fnkwargs)


def flatten_iterable(items, seqtypes=(list, tuple)):
    for i, x in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i+1] = items[i]
    return items


def match_edgenames_without_bs(new_en, old_ens):
    # Batch size may have changed
    # Remove bs from id str and then check if they match
    def remove_bs(edgename):
        substrs = str(edgename).split('_')
        for s_id, subs in enumerate(substrs):
            if subs.isdigit():
                # First substring which is int is bs
                bs_id = s_id
                break
        substrs.pop(bs_id)
        return '_'.join(substrs)

    new_ename_no_bs = remove_bs(str(new_en))
    for o in old_ens:
        old_ename_no_bs = remove_bs(o)
        if old_ename_no_bs == new_ename_no_bs:
            return o
