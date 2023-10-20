# coding=utf-8
import os

from experiments.video.networks import architectures, get_network
from core.utils.utils import dotdict as dd
from core.utils.argparsing import get_argparser


def parse_command_line_args():
    ap = get_argparser(inc_filtering=True,
                       inc_plotting=True,
                       inc_validation=True)

    ap.add_argument('--plot-denoise-recon', action='store_true')
    ap.add_argument('--n-iters-final-frame', type=int)
    ap.add_argument('--pairwise-smoothing', action='store_true')

    ap.add_argument('--dataset-name', type=str, default='walking')
    ap.add_argument('--corruption-seed', type=int, default=101)
    ap.add_argument('--corruption-type', type=str, default='uniform')
    ap.add_argument('--frac-noise', type=float, default=0.1)
    ap.add_argument('--reinit-input-only', action='store_true')
    ap.add_argument('--save-denoised-video', action='store_true')
    ap.add_argument('--eval-crop-border', type=int, default=0)
    ap.add_argument('--filtering-alpha', type=float)

    # Model config
    ap.add_argument('--architecture', type=str,
                    choices=list(architectures.keys()),
                    default='conv_tpose_single_layer_4')
    return ap.parse_args()


def get_config():
    cmd_args = parse_command_line_args()
    exp_args = dd(dataset_name=cmd_args.dataset_name,
                  do_filtering=not cmd_args.no_filtering,
                  filter_biases=not cmd_args.no_filter_biases,
                  filtering_alpha=cmd_args.filtering_alpha,
                  precision_rescaling=cmd_args.precision_rescaling,
                  precision_rescaling_conv_only=cmd_args.precision_rescaling_conv_only,
                  doing_validation=cmd_args.validation,
                  n_validation_data=cmd_args.n_validation_data,
                  weight_init_seed=cmd_args.weight_seed,
                  coeff_init_seed=cmd_args.coeff_seed,
                  plot_weights=cmd_args.plot_weights,
                  plot_coeffs=cmd_args.plot_coeffs,
                  plot_denoise_recon=cmd_args.plot_denoise_recon,
                  plot_message_convergence=cmd_args.plot_convergence,
                  plot_generative=cmd_args.plot_generative,
                  plot_train_batch_freq=cmd_args.plot_train_batch_freq,
                  save_denoised_video=cmd_args.save_denoised_video,
                  results_dir=cmd_args.logdir,
                  tf_logdir=os.path.join(cmd_args.logdir, 'tf_profile') if cmd_args.profile else None,
                  track_msg_diffs=cmd_args.plot_convergence,
                  record_all_msg_diffs=cmd_args.plot_convergence)
    recon_factor_args = dd(additive_factor=True,
                           sigma=cmd_args.factors_recon_sigma,
                           N_rob=cmd_args.factors_recon_N_rob,
                           rob_type='tukey',
                           sum_filters=True,
                           kmult=1.,
                           decompose=True,
                           relin_freq=cmd_args.relin_freq,
                           nonlin=cmd_args.nonlin,
                           nonlin_yscale=1.,
                           nonlin_xscale=1.,
                           use_bias=not cmd_args.factors_recon_no_bias,
                           relative_to_centre=False,
                           feedforward=cmd_args.factors_recon_feedforward,
                           fac_to_var_chunksize=8,
                           ksize=3,
                           pairwise=cmd_args.pairwise_smoothing,
                           stride=1)
    upsample_factor = dd(sigma=cmd_args.factors_upsample_sigma,
                         ksize=2,
                         relin_freq=1)
    pixel_obs_factor = dd(sigma=cmd_args.factors_pixel_obs_sigma,
                          N_rob=cmd_args.factors_pixel_obs_N_rob,
                          rob_type='tukey',
                          kmult=1.,
                          relin_freq=cmd_args.relin_freq)
    avg_pool_factor = dd(sigma=cmd_args.factors_avg_pool_sigma,
                         ksize=2)
    filter_prior_factor = dd(sigma=cmd_args.factors_weight_prior_sigma, mean=0.)
    coeff_prior_factor = dd(sigma=cmd_args.factors_coeff_prior_sigma,
                            N_rob=cmd_args.factors_recon_coeff_prior_N_rob,
                            rob_type='tukey',
                            mean=0.,
                            relin_freq=cmd_args.relin_freq,
                            sigma_test_mult=1.)
    bias_prior_factor = dd(mean=0., sigma=cmd_args.factors_bias_prior_sigma)
    factors = dd(recon=recon_factor_args,
                 weight_prior=filter_prior_factor,
                 bias_prior=bias_prior_factor,
                 pixel_obs=pixel_obs_factor,
                 coeff_prior=coeff_prior_factor,
                 pool=avg_pool_factor,
                 upsample=upsample_factor)

    netconf = get_network(cmd_args.architecture,
                          conv_sigmas=cmd_args.factors_recon_sigma_layers,
                          conv_coeff_prior_N_robs=cmd_args.factors_coeff_prior_N_rob_layers,
                          conv_coeff_prior_sigmas=cmd_args.factors_coeff_prior_sigma_layers,
                          dense_coeff_prior_sigma=cmd_args.factors_dense_coeff_prior_sigma,
                          dense_weight_prior_sigma=cmd_args.factors_dense_weight_prior_sigma,
                          conv_weight_prior_sigmas=cmd_args.factors_recon_weight_prior_sigma_layers)
    config = dd(inference='gbp',
                architecture=netconf,
                corruption_seed=cmd_args.corruption_seed,
                corruption_type=cmd_args.corruption_type,
                frac_noise=cmd_args.frac_noise,
                reinit_input_only=cmd_args.reinit_input_only,
                layer_schedule=cmd_args.layer_schedule,
                random_layer_schedule=cmd_args.random_layer_schedule or False,
                experiment=exp_args,
                factors=factors,
                n_iters_per_train_batch=cmd_args.n_iters_per_train_batch,
                n_iters_final_frame=cmd_args.n_iters_final_frame,
                n_train_eval_breaks=cmd_args.n_train_eval_breaks,    # How many times to pause GBP and test time and eval
                deterministic_init=False,
                eval_crop_border=cmd_args.eval_crop_border,
                momentum=cmd_args.momentum,
                dropout=cmd_args.dropout,
                rescale=1.,
                select_n_lowest_energy_filters=False,
                use_filter_coeffs=True,
                use_component_vars=False,
                init_weight_std=cmd_args.weight_init_std,
                init_coeff_std=cmd_args.coeff_init_std,
                random_coeff_init=True,
                use_static_graph=False if cmd_args.not_static_graph is True else True,
                xla_compile=cmd_args.xla_compile)
    return config
