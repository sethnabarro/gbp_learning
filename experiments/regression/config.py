# coding=utf-8
import os

from experiments.regression.networks import architectures, get_network
from core.utils.utils import dotdict as dd
from core.utils.argparsing import get_argparser


def parse_command_line_args():
    ap = get_argparser(inc_filtering=True,
                       inc_testing=True,
                       inc_classification=True,
                       inc_plotting=True,
                       inc_validation=True)

    ap.add_argument('--fix-params-for-testing', action='store_true')
    ap.add_argument('--factors-regression-obs-sigma', type=float, default=0.01)
    ap.add_argument('--factors-dense-coeff-prior-sigma-layers', nargs='+', type=float)
    ap.add_argument('--factors-dense-coeff-prior-N-rob-layers', nargs='+', type=float)
    ap.add_argument('--factors-last-coeff-prior-sigma', type=float, default=1.)
    ap.add_argument('--factors-last-weight-prior-sigma', type=float, default=3.)

    # Model config
    ap.add_argument('--architecture', type=str,
                    choices=list(architectures.keys()),
                    default='two_layer_mlp_8')
    args = ap.parse_args()
    return args


def get_config():
    cmd_args = parse_command_line_args()
    exp_args = dd(fix_params_for_testing=cmd_args.fix_params_for_testing,
                  with_gbp_softmax_layer=not cmd_args.no_dense_softmax_layer,
                  do_filtering=not cmd_args.no_filtering,
                  filter_biases=not cmd_args.no_filter_biases,
                  precision_rescaling=cmd_args.precision_rescaling,
                  precision_rescaling_test=cmd_args.precision_rescaling_test,
                  precision_rescaling_conv_only=cmd_args.precision_rescaling_conv_only,
                  n_classes=2,
                  examples_per_class_train=cmd_args.examples_per_class_train,
                  examples_per_class_test=cmd_args.examples_per_class_test,
                  batchsize_train=cmd_args.batchsize_train,
                  batchsize_test=cmd_args.batchsize_test,
                  weight_init_seed=cmd_args.weight_seed,
                  coeff_init_seed=cmd_args.coeff_seed,
                  dataset_shuffle_seed=cmd_args.data_seed,
                  shuffle_batches=False,
                  plot_weights=cmd_args.plot_weights,
                  plot_coeffs=cmd_args.plot_coeffs,
                  plot_message_convergence=cmd_args.plot_convergence,
                  plot_after_each_batch=cmd_args.plot_every_batch,
                  plot_train_batch_freq=cmd_args.plot_train_batch_freq,
                  plot_test_batch_freq=cmd_args.plot_test_batch_freq,
                  results_dir=cmd_args.logdir,
                  profile_test=cmd_args.profile_test or False,
                  tf_logdir=os.path.join(cmd_args.logdir, 'tf_profile') if cmd_args.profile else None,
                  track_msg_diffs=cmd_args.plot_convergence,
                  record_all_msg_diffs=cmd_args.plot_convergence,
                  checkpoint_frequency_batches=cmd_args.checkpoint_frequency_batches,
                  copy_lin_points=cmd_args.copy_lin_points,
                  filter_coeffs=False
                  )
    recon_factor_args = dd(additive_factor=True,
                           sigma=cmd_args.factors_recon_sigma,
                           N_rob=None,
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
                           ksize=3,
                           stride=1,
                           fac_to_var_chunksize=8)
    pixel_obs_factor = dd(sigma=cmd_args.factors_pixel_obs_sigma,
                          N_rob=None,
                          rob_type='tukey',
                          kmult=1.,
                          relin_freq=cmd_args.relin_freq)
    filter_prior_factor = dd(sigma=cmd_args.factors_weight_prior_sigma, mean=0.)
    coeff_prior_factor = dd(sigma=cmd_args.factors_coeff_prior_sigma,
                            N_rob=cmd_args.factors_recon_coeff_prior_N_rob,
                            rob_type='tukey',
                            mean=0.,
                            relin_freq=cmd_args.relin_freq,
                            sigma_test_mult=1.)
    regression_obs_factor = dd(sigma=cmd_args.factors_regression_obs_sigma,
                            relin_freq=cmd_args.relin_freq)
    dense_factor = dd(sigma=cmd_args.factors_dense_sigma,
                      relin_freq=cmd_args.relin_freq,
                      use_bias=True,
                      fac_to_var_chunksize=32,
                      decompose=True)

    bias_prior_factor = dd(mean=0., sigma=cmd_args.factors_bias_prior_sigma)
    factors = dd(recon=recon_factor_args,
                 weight_prior=filter_prior_factor,
                 bias_prior=bias_prior_factor,
                 pixel_obs=pixel_obs_factor,
                 coeff_prior=coeff_prior_factor,
                 regression_obs=regression_obs_factor,
                 dense=dense_factor)

    netconf = get_network(cmd_args.architecture,
                          conv_sigmas=cmd_args.factors_recon_sigma_layers,
                          dense_coeff_prior_N_robs=cmd_args.factors_dense_coeff_prior_N_rob_layers,
                          dense_coeff_prior_sigmas=cmd_args.factors_dense_coeff_prior_sigma_layers,
                          last_coeff_prior_sigma=cmd_args.factors_last_coeff_prior_sigma,
                          last_weight_prior_sigma=cmd_args.factors_last_weight_prior_sigma)
    config = dd(inference=cmd_args.inference,
                tf_deterministic=cmd_args.tf_deterministic,
                architecture=netconf,
                layer_schedule=cmd_args.layer_schedule,
                experiment=exp_args,
                gd_optim='adam',
                gd_lr=5e-4,
                factors=factors,
                n_iters_per_train_batch=cmd_args.n_iters_per_train_batch,
                n_iters_per_test_batch=cmd_args.n_iters_per_test_batch,
                n_train_eval_breaks=cmd_args.n_train_eval_breaks,    # How many times to pause GBP and test time and eval
                n_test_eval_breaks=cmd_args.n_test_eval_breaks,    # How many times to pause training and eval
                n_train_batches=cmd_args.n_train_batches,
                n_test_batches=cmd_args.n_test_batches,
                momentum=cmd_args.momentum,
                dropout=cmd_args.dropout,
                use_filter_coeffs=True,
                init_weight_std=cmd_args.weight_init_std,
                init_coeff_std=cmd_args.coeff_init_std,
                random_coeff_init=True,
                plot_init_model=True,
                use_static_graph=False if cmd_args.not_static_graph is True else True,
                xla_compile=False)
    return config
