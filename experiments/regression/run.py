# coding=utf-8
"""Training GBP nets on small regression task"""
from copy import deepcopy
import os
from contextlib import redirect_stdout
from functools import partial
import numpy as np
import sys
import tensorflow as tf

import matplotlib.pyplot as plt
from core.utils.utils import set_allow_gpu_mem_growth

from experiments.regression.config import get_config
from experiments.regression.data import load_data
from experiments.utils.regression import test_regression
from experiments.utils.utils import create_results_dir
from experiments.utils.run_exp import exp_main
from core.inference.gbp.layers import GBPRegressionObsLayer

set_allow_gpu_mem_growth()


def pretrain_fn(conf, run_fn):
    conf_pre = deepcopy(conf)
    conf_pre.experiment.results_dir = \
        os.path.join(conf_pre.experiment.results_dir, 'pretrain')
    print('\n\n\nStarting pretraining\n\n\n')

    # Create dir to store exp output
    create_results_dir(conf_pre.experiment.results_dir)
    conf_pre.n_iters_per_train_batch = conf_pre.pretrain.n_iters_per_train_batch
    conf_pre.n_train_plot_breaks = conf_pre.pretrain.n_train_plot_breaks
    conf_pre.inference = conf.pretrain.inference

    # Run pretraining
    model = run_fn(conf_pre, model_init=None)

    return model


def plot_regression(model,
                    plot_dir,
                    train_batch_id,
                    train_itr,
                    xtr,
                    ytr,
                    test_batch_id=None,
                    test_itr=None,
                    max_n_imgs=4,
                    basis_fns=False,
                    determ=False):
    ins = model.layers[0].pixel_obs_factor.obs
    if isinstance(model.layers[-1], GBPRegressionObsLayer):
        output_vars = model.layers[-(2 + int(basis_fns))].coeff_vars
    else:
        output_vars = model.layers[-(1 + int(basis_fns))].coeff_vars
    tf.debugging.check_numerics(output_vars.mu, 'Plot coeffs NAN')
    ymu = output_vars.mu
    ystd = output_vars.sigma
    plt.rc('font', family='serif', size=15)
    f = plt.gcf()
    f.set_size_inches((6., 4.))

    cols = []
    for j in range(ymu.shape[1]):
        l, = plt.plot(ins[:, 0, 0, 0], ymu[:, j])
        cols.append(l.get_color())

    if determ:
        layr = model.layers[-(1 + int(isinstance(model.layers[-1], GBPRegressionObsLayer)))]
        if not hasattr(layr, 'threshold'):
            out_determ = layr.dense_factor.forward_deterministic(ymu)
            plt.plot(ins[:, 0, 0, 0], out_determ, label='Out determ.')

        layr = model.layers[-(2 + int(isinstance(model.layers[-1], GBPRegressionObsLayer)))]
        if not hasattr(layr, 'threshold'):
            ins = layr.input_vars.mu
            basis_determ = layr.dense_factor.forward_deterministic(ins)
            for j in range(basis_determ.shape[1]):
                plt.plot(ins[:, 0, 0, 0], basis_determ[:, j],
                         linestyle='--', color=cols[j],
                         label='Basis determ.' if j == basis_determ.shape[1] - 1 else '')
    plt.fill_between(ins[:, 0, 0, 0], ymu[:, 0] - ystd[:, 0], ymu[:, 0] + ystd[:, 0],
                     color=l.get_color(), alpha=0.2, label='Model $\mu_y\pm\sigma_y$')
    plt.scatter(xtr, ytr, label='Training data', s=8, color='slategrey')
    plt.legend(fontsize=11, loc='upper center')
    # plt.text(0.65, -1.2, f'Iter. {train_itr}')
    plt.xlim(np.min(ins[:, 0, 0, 0]), np.max(ins[:, 0, 0, 0]))
    plt.xticks([-1., 0., 1.])
    plt.yticks([-1., 0., 1.])
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    fig = plt.gcf()
    fig.set_size_inches(6, 3.)
    plt.savefig(os.path.join(plot_dir, f'{"basisfns_" if basis_fns else ""}regression_{train_itr}.pdf'),
                transparent=True, bbox_inches='tight')
    plt.close('all')


def main():
    # Set up results dir
    conf = get_config()
    conf.experiment.results_dir = create_results_dir(conf.experiment.results_dir)

    print(f'\n\nOutput to logdir: {conf.experiment.results_dir}\n\n')

    # Evaluating performance during/after testing
    def eval(*args, **kwargs):
        kwargs['weights_lin_class'] = None
        kwargs['bias_lin_class'] = None
        kwargs['pred_only'] = True
        gbpnet_te, y_te, lin_w, lin_b, valacc = test_regression(*args, **kwargs)
        print(f'Energy after testing: {gbpnet_te.energy(as_numpy=True, sum_all=False)}')
        return gbpnet_te, y_te, valacc

    def eval_and_plot(model,
                      train_batch_id,
                      train_itr,
                      xtr,
                      ytr,
                      args_test,
                      test_data,
                      test_batch_id=None,
                      test_itr=None,
                      max_n_imgs=4):
        args_test.use_static_graph = True
        model_te = None
        retry = 0
        while model_te is None:
            try:
                model_te, y_te, valacc = eval(model=model,
                     train_itr=train_itr,
                     train_batch_id=train_batch_id,
                                args_test=args_test,
                                test_data=test_data)
            except tf.errors.InvalidArgumentError as err:
                print(f"Testing failed, train iter {train_itr}")
                args_test.experiment.weight_init_seed += 1
                args_test.experiment.coeff_init_seed += 1
                retry += 1
                if retry > 10:
                    raise err

        plot_regression(model_te,
                        plot_dir=os.path.join(conf.experiment.results_dir, 'test', 'regression_plots'),
                        train_itr=train_itr,
                        train_batch_id=train_batch_id, xtr=xtr, ytr=ytr)
        plot_regression(model_te,
                        plot_dir=os.path.join(conf.experiment.results_dir, 'test', 'regression_plots'),
                        train_itr=train_itr,
                        train_batch_id=train_batch_id, xtr=xtr, ytr=ytr, basis_fns=True)
        plot_regression(model,
                        plot_dir=os.path.join(conf.experiment.results_dir, 'train', 'regression_plots'),
                        train_itr=train_itr,
                        train_batch_id=train_batch_id, xtr=xtr, ytr=ytr)
        plot_regression(model,
                        plot_dir=os.path.join(conf.experiment.results_dir, 'train', 'regression_plots'),
                        train_itr=train_itr,
                        train_batch_id=train_batch_id, xtr=xtr, ytr=ytr, basis_fns=True)
        return model_te, y_te, valacc

    def run_fn(args, model_init=None, no_tr=False):
        sys.stderr = open(os.path.join(args.experiment.results_dir, f'err.txt'), 'w', buffering=1)
        with open(os.path.join(args.experiment.results_dir, f'out.txt'), 'w', buffering=1) as fo:
            with redirect_stdout(fo):
                train_data, test_data = load_data(n_te_points=args.experiment.batchsize_test,
                                                  seed=args.experiment.dataset_shuffle_seed)
                plot_and_eval_partial = partial(eval_and_plot,
                                                xtr=[a[0] for a in train_data.arr],
                                                ytr=[a[1] for a in train_data.arr])
                return exp_main(gbp_net_init=model_init,
                                args=args,
                                train_data=None if no_tr else train_data,
                                test_data=test_data,
                                eval_fn=plot_and_eval_partial,
                                plot_extra=None,
                                plot_subdirs=['regression_plots'])

    model = None

    run_fn(args=conf, model_init=model)


if __name__ == '__main__':
    main()
