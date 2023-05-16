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
    ap.add_argument('--test-on-train-set', action='store_true')    # Check ability to overfit
    ap.add_argument('--no-task-crosstalk-train', action='store_true')
    ap.add_argument('--no-task-crosstalk-test', action='store_true')
    ap.add_argument('--remaining-tasks-train', type=eval)
    ap.add_argument('--mnist-experiment-type',
                    default='split_mnist',
                    type=str,
                    choices=('split_mnist', 'standard_mnist', 'unsupervised'))

    # For NN baseline
    ap.add_argument('--backprop-optimiser', type=json.loads)
    ap.add_argument('--nn-architecture', type=json.loads)

    # Model config
    ap.add_argument('--architecture', type=str,
                    choices=list(architectures.keys()),
                    default='bignet')
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
                  test_on_future_tasks=False,
                  remaining_tasks_train=cmd_args.remaining_tasks_train,
                  do_filtering=not cmd_args.no_filtering,
                  filter_biases=not cmd_args.no_filter_biases,
                  precision_rescaling=cmd_args.precision_rescaling,
                  precision_rescaling_test=cmd_args.precision_rescaling_test,
                  precision_rescaling_conv_only=cmd_args.precision_rescaling_conv_only,
                  n_classes=10,
                  doing_validation=cmd_args.validation,
                  n_validation_data=cmd_args.n_validation_data,
                  test_only=cmd_args.test_only,
                  examples_per_class_train=cmd_args.examples_per_class_train,
                  examples_per_class_test=cmd_args.examples_per_class_test,
                  batchsize_train=cmd_args.batchsize_train,
                  batchsize_test=cmd_args.batchsize_test,
                  weight_init_seed=cmd_args.weight_seed,
                  coeff_init_seed=cmd_args.coeff_seed,
                  dataset_shuffle_seed=cmd_args.data_seed,
                  shuffle_batches=cmd_args.shuffle_batches,
                  plot_weights=cmd_args.plot_weights,
                  plot_coeffs=cmd_args.plot_coeffs,
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
                  no_task_crosstalk_train=cmd_args.no_task_crosstalk_train,
                  no_task_crosstalk_test=cmd_args.no_task_crosstalk_test,
                  )
    recon_factor_args = dd(additive_factor=True,
                           sigma=cmd_args.factors_recon_sigma,
                           N_rob=4.,
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
                          N_rob=10.,
                          rob_type='tukey',
                          kmult=1.,
                          relin_freq=cmd_args.relin_freq)
    pool_factor = dd(sigma=cmd_args.factors_avg_pool_sigma,
                     ksize=2,
                     relin_freq=1)
    filter_prior_factor = dd(sigma=cmd_args.factors_weight_prior_sigma, mean=0.)
    coeff_prior_factor = dd(sigma=cmd_args.factors_coeff_prior_sigma,
                            N_rob=cmd_args.factors_recon_coeff_prior_N_rob,
                            rob_type='tukey',
                            mean=0.,
                            relin_freq=cmd_args.relin_freq,
                            sigma_test_mult=1.)
    softmax_obs_factor = dd(sigma=cmd_args.factors_softmax_obs_sigma,
                            relin_freq=cmd_args.relin_freq)
    dense_factor = dd(sigma=cmd_args.factors_dense_sigma,
                      relin_freq=cmd_args.relin_freq,
                      use_bias=True,
                      fac_to_var_chunksize=32,)
    bias_prior_factor = dd(mean=0., sigma=cmd_args.factors_bias_prior_sigma)
    factors = dd(recon=recon_factor_args,
                 weight_prior=filter_prior_factor,
                 bias_prior=bias_prior_factor,
                 pixel_obs=pixel_obs_factor,
                 coeff_prior=coeff_prior_factor,
                 pool=pool_factor,
                 softmax_class_obs=softmax_obs_factor,
                 dense=dense_factor)

    if cmd_args.inference == 'backprop':
        netconf = dd(cmd_args.nn_architecture)
    else:
        netconf = get_network(cmd_args.architecture,
                              conv_sigmas=cmd_args.factors_recon_sigma_layers,
                              conv_coeff_prior_N_robs=cmd_args.factors_coeff_prior_N_rob_layers,
                              conv_coeff_prior_sigmas=cmd_args.factors_coeff_prior_sigma_layers,
                              dense_coeff_prior_sigma=cmd_args.factors_dense_coeff_prior_sigma,
                              dense_weight_prior_sigma=cmd_args.factors_dense_weight_prior_sigma)
    config = dd(inference=cmd_args.inference,
                exp_type=cmd_args.mnist_experiment_type,
                tf_deterministic=cmd_args.tf_deterministic,
                architecture=netconf,
                backprop_optimiser=dd(cmd_args.backprop_optimiser) if cmd_args.backprop_optimiser is not None else None,
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
                dropout=0.0,
                rescale=1.,
                select_n_lowest_energy_filters=False,
                use_filter_coeffs=True,
                use_component_vars=False,
                init_weight_std=cmd_args.weight_init_std,
                init_coeff_std=cmd_args.coeff_init_std,
                random_coeff_init=True,
                use_static_graph=False if cmd_args.not_static_graph is True else True,
                xla_compile=False)
    return config
