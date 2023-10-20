# coding=utf-8
import json
import os

from experiments.mnist.networks import architectures, get_network
from core.utils.utils import dotdict as dd
from core.utils.argparsing import get_argparser


def parse_command_line_args():
    ap = get_argparser(inc_filtering=True,
                       inc_testing=True,
                       inc_classification=True,
                       inc_plotting=True,
                       inc_validation=True)

    ap.add_argument('--fix-params-for-testing', action='store_true')    # Filtering (between train and test)
    ap.add_argument('--corrupt-test-inputs', type=str)             # Whether and how to add noise to test images
    ap.add_argument('--input-rescale', type=str, choices=('zero_one', 'mean_std'))  # How to rescale images
    ap.add_argument('--test-on-train-set', action='store_true')    # Check ability to overfit
    ap.add_argument('--remaining-tasks-train', type=eval)
    ap.add_argument('--mnist-experiment-type',
                    default='standard_mnist',
                    type=str,
                    choices=('standard_mnist',))
    ap.add_argument('--inference-test', type=str, choices=('gd', 'gbp'))

    # If below arg given then each batch will have same class proportions
    # as full training set, else each batch will have equal numbers of examples
    # from each class (until the end of the dataset is approached).
    ap.add_argument('--not-class-balanced-batches-train', action='store_true')

    # Model config
    ap.add_argument('--architecture', type=str,
                    choices=list(architectures.keys()),
                    default='three_layer_k5_ff_8')
    args = ap.parse_args()
    return args


def get_config():
    cmd_args = parse_command_line_args()
    if cmd_args.validation:
        assert isinstance(cmd_args.n_validation_data, int), \
            "Must specify `--n-validation-data` if running validation"
    exp_args = dd(fix_params_for_testing=cmd_args.fix_params_for_testing,
                  with_gbp_softmax_layer=not cmd_args.no_dense_softmax_layer,
                  test_on_train_set=cmd_args.test_on_train_set,       # To check memorisation
                  corrupt_test_inputs=cmd_args.corrupt_test_inputs,
                  do_filtering=not cmd_args.no_filtering,
                  filter_biases=not cmd_args.no_filter_biases,
                  precision_rescaling=cmd_args.precision_rescaling,
                  precision_rescaling_test=cmd_args.precision_rescaling_test,
                  precision_rescaling_conv_only=cmd_args.precision_rescaling_conv_only,
                  n_classes=10,
                  input_rescale=cmd_args.input_rescale,
                  doing_validation=cmd_args.validation,
                  n_validation_data=cmd_args.n_validation_data,
                  test_only=cmd_args.test_only,
                  examples_per_class_train=cmd_args.examples_per_class_train,
                  examples_per_class_test=cmd_args.examples_per_class_test,
                  batchsize_train=cmd_args.batchsize_train,
                  batchsize_test=cmd_args.batchsize_test,
                  class_balanced_batches_train=not bool(cmd_args.not_class_balanced_batches_train),
                  weight_init_seed=cmd_args.weight_seed,
                  coeff_init_seed=cmd_args.coeff_seed,
                  dataset_shuffle_seed=cmd_args.data_seed,
                  shuffle_batches=cmd_args.shuffle_batches,
                  plot_weights=cmd_args.plot_weights,
                  plot_coeffs=cmd_args.plot_coeffs,
                  plot_generative=cmd_args.plot_generative,
                  plot_message_convergence=cmd_args.plot_convergence,
                  plot_after_each_batch=cmd_args.plot_every_batch,
                  plot_train_batch_freq=cmd_args.plot_train_batch_freq,
                  plot_test_batch_freq=cmd_args.plot_test_batch_freq,
                  results_dir=cmd_args.logdir,
                  load_checkpoint_dir=cmd_args.load_checkpoint_dir,
                  config_filepath=cmd_args.config_filepath,
                  profile_test=cmd_args.profile_test or False,
                  tf_logdir=os.path.join(cmd_args.logdir, 'tf_profile') if cmd_args.profile else None,
                  track_msg_diffs=cmd_args.plot_convergence,
                  record_all_msg_diffs=cmd_args.plot_convergence,
                  checkpoint_frequency_batches=cmd_args.checkpoint_frequency_batches,
                  save_best_model=cmd_args.checkpoint_frequency_batches is not None,
                  copy_lin_points=cmd_args.copy_lin_points,
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
                           relative_to_centre=cmd_args.factors_recon_relative_to_centre if cmd_args.factors_recon_relative_to_centre is not None else False,
                           feedforward=cmd_args.factors_recon_feedforward,
                           ksize=3,
                           stride=1,
                           fac_to_var_chunksize=8)
    pixel_obs_factor = dd(sigma=cmd_args.factors_pixel_obs_sigma,
                          N_rob=None,
                          rob_type='tukey',
                          kmult=1.,
                          relin_freq=cmd_args.relin_freq,
                          mask_prec=1e1)
    pool_factor = dd(sigma=cmd_args.factors_avg_pool_sigma,
                     ksize=2,
                     relin_freq=1)
    upsample_factor = dd(sigma=cmd_args.factors_upsample_sigma,
                         ksize=2,
                         relin_freq=1)
    filter_prior_factor = dd(sigma=cmd_args.factors_weight_prior_sigma, mean=0.)
    coeff_prior_factor = dd(sigma=cmd_args.factors_coeff_prior_sigma,
                            N_rob=None,
                            rob_type='tukey',
                            mean=0.,
                            relin_freq=cmd_args.relin_freq,
                            sigma_test_mult=1.)
    softmax_obs_factor = dd(sigma=cmd_args.factors_softmax_obs_sigma,
                            relin_freq=cmd_args.relin_freq)
    dense_factor = dd(sigma=cmd_args.factors_dense_sigma,
                      relin_freq=cmd_args.relin_freq,
                      use_bias=(not cmd_args.factors_dense_no_bias) if cmd_args.factors_dense_no_bias is not None else True,
                      fac_to_var_chunksize=32,
                      decompose=True)
    bias_prior_factor = dd(mean=0., sigma=cmd_args.factors_bias_prior_sigma)
    factors = dd(recon=recon_factor_args,
                 weight_prior=filter_prior_factor,
                 bias_prior=bias_prior_factor,
                 pixel_obs=pixel_obs_factor,
                 coeff_prior=coeff_prior_factor,
                 pool=pool_factor,
                 upsample=upsample_factor,
                 softmax_class_obs=softmax_obs_factor,
                 dense=dense_factor)

    netconf = get_network(cmd_args.architecture,
                          conv_sigmas=cmd_args.factors_recon_sigma_layers,
                          conv_coeff_prior_N_robs=cmd_args.factors_coeff_prior_N_rob_layers,
                          conv_coeff_prior_sigmas=cmd_args.factors_coeff_prior_sigma_layers,
                          dense_coeff_prior_sigma=cmd_args.factors_dense_coeff_prior_sigma,
                          dense_coeff_prior_N_rob=cmd_args.factors_dense_coeff_prior_N_rob,
                          dense_weight_prior_sigma=cmd_args.factors_dense_weight_prior_sigma)
    config = dd(inference=cmd_args.inference,
                inference_test=cmd_args.inference_test,  # May be None
                exp_type=cmd_args.mnist_experiment_type,
                tf_deterministic=cmd_args.tf_deterministic,
                architecture=netconf,
                layer_schedule=cmd_args.layer_schedule,
                random_layer_schedule=cmd_args.random_layer_schedule or False,
                experiment=exp_args,
                factors=factors,
                n_iters_per_train_batch=cmd_args.n_iters_per_train_batch,
                n_iters_per_test_batch=cmd_args.n_iters_per_test_batch,
                n_train_eval_breaks=cmd_args.n_train_eval_breaks,    # How many times to pause GBP and test time and eval
                n_test_eval_breaks=cmd_args.n_test_eval_breaks,    # How many times to pause training and eval
                n_train_batches=cmd_args.n_train_batches,
                n_test_batches=cmd_args.n_test_batches,
                deterministic_init=False,
                momentum=cmd_args.momentum,
                dropout=cmd_args.dropout,
                gd_optim='adam',
                gd_lr=1e-3,
                rescale=1.,
                use_filter_coeffs=True,
                init_weight_std=cmd_args.weight_init_std,
                init_coeff_std=cmd_args.coeff_init_std,
                random_coeff_init=True,
                use_static_graph=False if cmd_args.not_static_graph is True else True,
                xla_compile=cmd_args.xla_compile)
    return config
