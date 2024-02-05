# coding=utf-8
"""Training GBP nets on MNIST"""
import os
from contextlib import redirect_stdout
import json
import sys

import tensorflow as tf

from core.utils.utils import set_allow_gpu_mem_growth, dotdict

from experiments.img_classification.config import get_config
from experiments.img_classification.data import load_data
from core.utils.saveload import load_checkpoint
from experiments.utils.classification import test_classification
from experiments.utils.utils import create_results_dir
from experiments.utils.run_exp import exp_main

set_allow_gpu_mem_growth()


def plot_input_vars(model,
                    plot_dir,
                    train_batch_id,
                    train_itr,
                    test_batch_id=None,
                    test_itr=None,
                    max_n_imgs=4):
    invars = model.layers[0].input_vars if model.layers[0].input_vars is not None else model.layers[0].coeff_vars
    invar_mu = invars.mu
    import matplotlib.pyplot as plt
    for i in range(10):
        plt.imshow(invar_mu[i])
        plt.axis('off')
        plt.savefig(os.path.join(plot_dir, f'img{i}_trbatch{train_batch_id}_tritr{train_itr}_recon.png'), bbox_inches='tight')
        plt.close('all')
    invar_obs = model.layers[0].pixel_obs_factor.obs
    for i in range(10):
        plt.imshow(invar_obs[i])
        plt.axis('off')
        plt.savefig(os.path.join(plot_dir, f'img{i}_trbatch{train_batch_id}_tritr{train_itr}_obs.png'), bbox_inches='tight')
        plt.close('all')


def main():
    # Set up results dir
    conf = get_config()
    if conf.experiment.config_filepath is not None:
        with open(conf.experiment.config_filepath, 'r') as confjson:
            conf_loaded = dotdict.from_dict(json.load(confjson))

        # If evaluating loaded checkpoint, replace args previously used during training with below
        conf_loaded.experiment.fix_params_for_testing = conf.experiment.fix_params_for_testing
        conf_loaded.experiment.remaining_tasks_train = conf.experiment.remaining_tasks_train
        conf_loaded.experiment.test_only = conf.experiment.test_only
        conf_loaded.experiment.load_checkpoint_dir = conf.experiment.load_checkpoint_dir
        conf_loaded.experiment.examples_per_class_test = conf.experiment.examples_per_class_test
        conf_loaded.experiment.examples_per_class_train = conf.experiment.examples_per_class_train
        conf_loaded.n_test_batches = conf.n_test_batches
        conf_loaded.n_train_batches = conf.n_train_batches
        conf_loaded.experiment.batchsize_test = conf.experiment.batchsize_test
        conf_loaded.experiment.batchsize_train = conf.experiment.batchsize_train
        conf_loaded.experiment.dataset_shuffle_seed = conf.experiment.dataset_shuffle_seed
        conf_loaded.experiment.corrupt_test_inputs = conf.experiment.corrupt_test_inputs
        conf_loaded.experiment.conditional_generation_test = conf.experiment.conditional_generation_test
        conf_loaded.experiment.plot_coeffs = conf.experiment.plot_coeffs
        conf_loaded.experiment.plot_weights = conf.experiment.plot_weights
        conf_loaded.experiment.plot_message_convergence = conf.experiment.plot_message_convergence
        conf_loaded.inference = conf.inference
        conf_loaded.inference_test = conf.inference_test
        conf_loaded.gd_optim = conf.gd_optim
        conf_loaded.gd_lr = conf.gd_lr
        conf_loaded.use_static_graph = conf.use_static_graph
        conf_loaded.xla_compile = conf.xla_compile
        conf_loaded.exp_type = conf.exp_type
        conf_loaded.n_iters_per_test_batch = conf.n_iters_per_test_batch
        results_dir = conf_loaded.experiment.results_dir if os.path.exists(conf_loaded.experiment.results_dir) else conf.experiment.results_dir

        conf = conf_loaded
        conf.experiment.results_dir = os.path.join(results_dir, 'rerun/')
    conf.experiment.results_dir = create_results_dir(conf.experiment.results_dir)

    print(f'\n\nOutput to logdir: {conf.experiment.results_dir}\n\n')

    if conf.exp_type == 'unsupervised':
        conf.experiment.with_gbp_softmax_layer = False

    # Evaluating performance during/after testing
    lin_class_params = {'weights_lin_class': None, 'bias_lin_class': None}
    def eval(*args, **kwargs):
        for n in lin_class_params:
            kwargs[n] = lin_class_params[n]
        gbpnet_te, y_te, lin_w, lin_b, valacc = test_classification(*args, **kwargs)

        lin_class_params['weights_lin_class'] = lin_w
        lin_class_params['bias_lin_class'] = lin_b
        return gbpnet_te, y_te, valacc

    if conf.tf_deterministic:
        tf.config.experimental.enable_op_determinism()

    def run(config,
            model_in=None,
            no_te=False,
            no_tr=False,

            load_from_cp=False):
        config_temp = config

        sys.stderr = open(os.path.join(config_temp.experiment.results_dir, f'err.txt'), 'w', buffering=1)
        with open(os.path.join(config_temp.experiment.results_dir, f'out.txt'), 'w', buffering=1) as fo:
            with redirect_stdout(fo):
                # Get MNIST
                train_data, test_data = \
                    load_data(shuffle_seed=config_temp.experiment.dataset_shuffle_seed,
                              shuffle=True,
                              test_on_train_set=config_temp.experiment.test_on_train_set,
                              validation=config_temp.experiment.doing_validation,
                              n_validation=config_temp.experiment.n_validation_data,
                              examples_per_class_train=config_temp.experiment.examples_per_class_train,
                              examples_per_class_test=config_temp.experiment.examples_per_class_test,
                              rescale=config_temp.experiment.input_rescale if config_temp.experiment.input_rescale is not None else True,
                              dataset=config_temp.dataset)

                if load_from_cp:
                    model_in, model_spec = \
                        load_checkpoint(dataset=test_data if no_tr else train_data,
                                        checkpoint_dir=config_temp.experiment.load_checkpoint_dir,
                                        args=config_temp)
                    config_temp.experiment.load_checkpoint_dir = None  # Prevents checkpoint being reloaded in `exp_main()`
                model_out = exp_main(args=config_temp,
                                     gbp_net_init=model_in,
                                     train_data=train_data if not (config_temp.experiment.test_only or no_tr) else None,
                                     test_data=None if no_te else test_data,
                                     eval_fn=None if no_te else eval,
                                     plot_extra=plot_input_vars if config_temp.experiment.corrupt_test_inputs else None)
                return model_out, config_temp.experiment.results_dir

    run(config=conf)


if __name__ == '__main__':
    main()
