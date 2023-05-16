# coding=utf-8
import numpy as np
import tensorflow as tf

from experiments.utils.graph_setup_utils import init_conv_layer, get_init_conv_weights, init_layers
from core.inference.gbp.learner import GBPLearner
from core.utils.utils import dotdict as dd, flatten_filters


def generate_img_with_filters(img_shp, filts, batch_size, bias=None):
    k_size = filts.shape[0]
    n_channels_in = img_shp[-1]
    n_filt_gt = 2
    filters = tf.random.normal([3, 3, n_channels_in, n_filt_gt])
    coeffs = np.array([np.array([np.linspace(0., (1. + float(c)) / 2., num=img_shp[0] - k_size + 1)[None] ** 2. + \
        np.cos(np.pi * np.linspace(0., (1. + float(b)), num=img_shp[1] - k_size + 1)[:, None]) ** 2.
                for c in range(n_filt_gt)]).T
               for b in range(batch_size)])
    coeffs = np.reshape(np.array(coeffs, dtype=np.float32), (batch_size, img_shp[0] - k_size + 1, img_shp[1] - k_size + 1, n_filt_gt))

    # Transpose conv to make image
    toy_img = tf.nn.conv2d_transpose(coeffs, filters, output_shape=(batch_size,) + img_shp, padding='VALID', strides=[1, 1, 1, 1])

    toy_img -= tf.reduce_min(toy_img)
    toy_img /= tf.reduce_max(toy_img)

    if bias is not None:
        toy_img += bias[None, None, None]
    return toy_img


def gbp_reconstruct(img_dim=(9, 9, 1), batch_size=2, with_bias=False, n_iters=100,
                    use_bias=True, img_scale=1., padding=0, use_gt_filters=True, coeff_padding=0):
    # Config
    assert img_dim[0] >= 9
    assert img_dim[1] >= 9
    k_size = 3
    stride = 1
    n_filters = 4
    arch = [dd(name='conv1', type='conv', n_filters=n_filters)]
    if coeff_padding > 0:
        arch *= 2  # 2 layers
        arch[1].padding = coeff_padding
    conf = dd(factors=dd(pixel_obs=dd(sigma=0.01, mask_prec=1e-3),
                         recon=dd(sigma=0.2,
                                  ksize=k_size,
                                  stride=stride,
                                  padding=padding,
                                  relin_freq=1,
                                  sum_filters=True,
                                  relative_to_centre=False,
                                  decompose=True,
                                  use_bias=use_bias),
                         coeff_prior=dd(sigma=1., mean=0.),
                         weight_prior=dd(sigma=2.),  # Filter prior
                         bias_prior=dd(sigma=6., mean=0.)),
              momentum=0.9,
              architecture=arch,
              inference='gbp',
              use_component_vars=False,
              use_filter_coeffs=True,
              coeff_init_scale=1.,
              dropout=0.,
              experiment=dd())
    n_channels_in = img_dim[-1]

    # Generate simple image(s) - some pattern in first layer coeffs upsample convolved with random filters
    bias_gt = tf.random.normal((n_channels_in,)) * 0.2 if with_bias else tf.zeros((n_channels_in,))
    filters = tf.random.normal((k_size, k_size, 2, n_channels_in))
    if use_gt_filters:
        toy_img = generate_img_with_filters(img_dim, filters, batch_size, bias_gt) * img_scale + bias_gt
    else:
        xs = np.linspace(0., img_dim[0], img_dim[0])
        ys = np.linspace(0., img_dim[1], img_dim[1])
        zs = np.linspace(1., img_dim[2], img_dim[2])
        toy_img = xs[:, None, None] * ys[None, :, None] * zs[None, None]
        toy_img -= tf.reduce_min(toy_img)
        toy_img /= tf.reduce_max(toy_img)
        toy_img = tf.cast(toy_img[None], tf.float32)

    # Initialise weights, set up layer
    if len(arch) == 1:
        filters = get_init_conv_weights(img=None, n_filters=n_filters, n_filts_in=n_channels_in, zero_centre=False)
        conv_layer = init_conv_layer(filters_init=filters,
                                     config=conf,
                                     n_filts=n_filters,
                                     img_noisy=toy_img,
                                     padding=padding)
        for itr in range(n_iters):
            conv_layer.intra_layer_inference_iter(itr)
    else:
        layrs = init_layers(config=conf, img_obs=toy_img, weight_init_std=1.)
        gbpnet = GBPLearner(layrs)
        gbpnet.run_inference(n_iters)
        conv_layer = gbpnet.layers[0]

    filts_flat = flatten_filters(conv_layer.filter_vars.mu)
    coeffs = conv_layer.coeff_vars.mu
    recon_diff_patches = tf.reduce_sum(filts_flat[None, None, None] * coeffs[..., None, :, None], axis=-2)

    bias_model = conv_layer.bias_vars.mu if use_bias else None
    return toy_img, recon_diff_patches, bias_gt, bias_model


class TestGBPConvLayer(tf.test.TestCase):
    def reconstruction(self, batch_size=1, img_dim=(10, 10, 1), n_iters=100,
                       use_bias=False, img_rescale=1., padding=0, coeff_padding=0):
        tf.random.set_seed(8)
        np.random.seed(8)

        img_gt, model_recon_patches, bias_ground_truth, bias_model = \
            gbp_reconstruct(batch_size=batch_size,
                            img_dim=img_dim,
                            n_iters=n_iters,
                            with_bias=use_bias,
                            img_scale=img_rescale,
                            padding=padding,
                            use_gt_filters=coeff_padding == 0,
                            coeff_padding=coeff_padding)
        relative_to_centre = False
        k_size = int(np.sqrt(model_recon_patches.shape[-1]))

        # Get diffs between central pixel and surrounding pixels for each neighbourhood
        # Assert model-reconstructed differences and true differences are similar
        # Only test centrish region of the image to avoid edge effects
        for y in range(k_size + 1 + padding, img_gt.shape[1] - k_size - 1 - padding):
            for x in range(k_size + 1 + padding, img_gt.shape[2] - k_size - 1 - padding):
                khalf = int((k_size - 1) / 2)
                if relative_to_centre:
                    # Recon differences between neighbouring and central pixels wTx ≈ x - x_{2,2}
                    model_diffs = tf.transpose(model_recon_patches[:, y - khalf, x - khalf], (0, 2, 1))
                    model_diffs = tf.reshape(model_diffs, (model_diffs.shape[0], k_size, k_size, -1))

                    # Subtract central pixel
                    img_diffs = \
                        img_gt[:, y - khalf:y + khalf + 1, x - khalf:x + khalf + 1] - img_gt[:, y, x][:, None, None]
                else:
                    # Learnt bias - recon is wTx + b
                    model_diffs = tf.transpose(model_recon_patches[:, y - khalf, x - khalf], (0, 2, 1))
                    model_diffs = tf.reshape(model_diffs, (model_diffs.shape[0], k_size, k_size, -1)) + bias_model
                    img_diffs = \
                        img_gt[:, y - khalf:y + khalf + 1, x - khalf:x + khalf + 1]

                self.assertAllClose(model_diffs, img_diffs, atol=0.15)
        return img_gt, model_recon_patches, bias_ground_truth, bias_model

    def test_conv_layer(self):
        self.reconstruction(batch_size=1)
        self.reconstruction(batch_size=1, use_bias=True)
        self.reconstruction(batch_size=2, img_dim=(20, 20, 1))
        self.reconstruction(batch_size=2, img_dim=(20, 20, 1), use_bias=True)
        self.reconstruction(batch_size=1, img_dim=(20, 20, 3), n_iters=200, use_bias=True)
        self.reconstruction(batch_size=1, img_dim=(20, 20, 3), n_iters=100, use_bias=False)

    def test_conv_bias(self):
        bias_gt, bias_model = \
            self.reconstruction(batch_size=1, img_dim=(10, 10, 1), n_iters=100, use_bias=True, img_rescale=0.)[-2:]
        self.assertAllClose(bias_gt, bias_model, atol=0.05)

    def test_padding(self):
        # Test gbp in model still runs with padded input
        # and observed pixels still reconstructed well
        self.reconstruction(padding=1)

    def test_coeff_padding(self):
        # Test gbp in model still runs with 2 layer net where coeffs of first layer are padded
        self.reconstruction(coeff_padding=1)

