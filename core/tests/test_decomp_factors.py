# coding=utf-8
"""Tests to confirm that the decomposed factors
are the same as using the original, large factors"""
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import unittest

from core.archive.gbp_learning import learn_filters
from core.img_model import filter_energy, total_energy
from core.utils.utils import corrupt_image, peak_snr, load_image, plot_denoising, plot_filter_energies

SEED = 28
config = dict()
config['n_filters'] = 8
config['img_h'] = 100
config['img_w'] = 100
config['frac_added_image_noise'] = 0.05
config['ksize'] = 3   # kernel size, must be odd
config['gbp_kwargs'] = dict(additive_factor=True,
                            dynamic_robust_thresh_mixture_weight=False,
                            sigma_filter=0.1,
                            sigma_obs=0.25,
                            rob_type='tukey',
                            N_rob=1.,           # Threshold
                            n_iters=30,
                            momentum=0.,
                            sigma_filter_prior=10.,
                            select_lowest_energy_filter=True,
                            serialise_marginalisation=True,
                            woodbury_inv=False,
                            multidim_filter_vars=False,
                            decomp_factors=False)
config['init_filter_std'] = 0.1
img_path = '../../inputs/brick-texture.png'


def get_random_filters(n_filters, std=1., ksize=3, seed=None):
    """Randomly initialised filters"""
    if seed is not None:
        np.random.seed(seed)
    filts = np.random.normal(loc=0., scale=std, size=(n_filters, ksize ** 2)).astype(np.float32)
    filts[:, 4] = 0.  # filter factors encode difference between central pixel and neighbours
    filts = np.reshape(filts, (ksize, ksize, 1, -1))
    return filts


def gbp_learn(img, filters, **kwargs):
    gbp_config = kwargs['gbp_kwargs']
    patch_energy_fn = partial(filter_energy,
                              additive_factor=gbp_config['additive_factor'],
                              robust_type=gbp_config['rob_type'],
                              N_rob=gbp_config['N_rob'],
                              sigma=gbp_config['sigma_filter'])

    energies = []
    psnrs = []

    def _callback(imgrecon, filters, itr):
        E = total_energy(x=imgrecon,
                         img=img,
                         filters=filters,
                         sigma_lhood=gbp_config['sigma_obs'],
                         filter_energy_fn_closed=patch_energy_fn,
                         robust=gbp_config['N_rob'] is not None,
                         inc_measurement_energy=True,
                         only_lowest_energy_filter=gbp_config['select_lowest_energy_filter'])
        # print(itr)
        # plot_filter_energies(imgrecon, imgrecon, filters, N_rob=gbp_config['N_rob'],
        #                      energy_fn=patch_energy_fn, itr=itr,
        #                      inc_lowest_energy_factor=True,
        #                      plot_lowest_energy_map=itr in (2, 3, 4),
        #                      plotfile=f"../../experiments/filter_energies/{itr}.png")
        energies.append(E)
        psnr = peak_snr(img, imgrecon)
        psnrs.append(psnr)

    if kwargs['frac_added_image_noise'] > 0.:
        img_in = corrupt_image(img, kwargs['frac_added_image_noise'], noise_type=bool)
    else:
        img_in = img
    filters_eta, filters_Lambda, pixel_eta, pixel_Lambda = \
        learn_filters(use_static_graph=False,
                      img=tf.convert_to_tensor(img_in),
                      init_filters=filters,
                      filter_energy_fn=patch_energy_fn,
                      callback_fn=_callback,
                      **gbp_config)

    filters_out = filters_eta / filters_Lambda
    imgrecon = pixel_eta / pixel_Lambda
    return filters_out, imgrecon,  psnrs, energies, patch_energy_fn


class TestDecompFactors(unittest.TestCase):
    def set_up(self, conf):
        self.filters = get_random_filters(conf['n_filters'],
                                          std=conf['init_filter_std'],
                                          seed=SEED)
        self.img = load_image(img_path, conf['img_h'], conf['img_w'], plot_image=False)

    def test_decomp_factors(self):
        self.set_up(config)
        import time

        config_decomp = config.copy()
        config_decomp['gbp_kwargs'] = config['gbp_kwargs'].copy()
        config_decomp['gbp_kwargs']['decomp_factors'] = True
        config_decomp['gbp_kwargs']['serialise_marginalisation'] = True
        t3 = time.time()
        filters_d, recon_d, psnrs_d, energies_d, energy_fn_d, = gbp_learn(self.img, self.filters, **config_decomp)
        t4 = time.time()
        plot_filter_energies(recon_d, recon_d, filters_d, N_rob=config_decomp['gbp_kwargs']['N_rob'],
                             energy_fn=energy_fn_d, itr=-1,
                             inc_lowest_energy_factor=True,
                             plot_lowest_energy_map=True,
                             plotfile=f"../../experiments/filter_energies/final_decomp.png")
        t1 = time.time()
        filters, recon, psnrs, energies, energy_fn = gbp_learn(self.img, self.filters, **config)
        t2 = time.time()
        plot_filter_energies(recon, recon, filters, N_rob=config_decomp['gbp_kwargs']['N_rob'],
                             energy_fn=energy_fn, itr=-1,
                             inc_lowest_energy_factor=True,
                             plot_lowest_energy_map=True,
                             plotfile=f"../../experiments/filter_energies/final_no_decomp.png")

        plt.plot(psnrs_d, label='Decomposed factors')
        plt.plot(psnrs, '--', label='Large factors')
        plt.ylabel('PSNR')
        plt.legend()
        plt.show()
        plt.plot(energies_d, label='Decomposed factors')
        plt.plot(energies, '--', label='Large factors')
        plt.ylabel('Energy')
        plt.legend()
        plt.show()
        plot_denoising(self.img, recon_d, -1)
        plot_denoising(self.img, recon, -1)
        print(f'big factors take {t2 - t1} secs. small factors take {t4-t3} secs.')



