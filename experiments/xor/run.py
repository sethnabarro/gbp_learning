# coding=utf-8
"""Training GBP nets on XOR"""
from copy import deepcopy
import os
from contextlib import redirect_stdout
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from core.utils.utils import set_allow_gpu_mem_growth, dotdict

from experiments.xor.config import get_config
from experiments.xor.data import load_data
from experiments.utils.classification import test_classification
from experiments.utils.utils import create_results_dir
from experiments.utils.run_exp import exp_main
from core.inference.gbp.layers import GBPSoftmaxClassObservationLayer

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


def plot_xor(model,
             plot_dir,
             train_batch_id,
             train_itr,
             test_batch_id=None,
             test_itr=None,
             max_n_imgs=4):
    ins = model.layers[0].pixel_obs_factor.obs
    if isinstance(model.layers[-1], GBPSoftmaxClassObservationLayer):
        logits = model.layers[-2].coeff_vars.mu
    else:
        logits = model.layers[-1].coeff_vars.mu
    plt.rc('font', family='serif', size=15)
    probs = tf.nn.softmax(logits, axis=-1)
    f = plt.gcf()
    f.set_size_inches((5.5, 5.5))
    lo, hi = np.min(ins), np.max(ins)
    marg = 0.2
    plt.xlim([lo, hi])
    plt.ylim([lo, hi])
    plt.tricontourf(ins[:, 0, 0, 0], ins[:, 0, 0, 1], probs[:, 0], levels=100, cmap=plt.viridis())
    plt.scatter([lo + marg, hi - marg],
                [hi - marg, lo + marg],
                c=[0., 0.],
                s=120.,
                edgecolors='white',
                linewidth=2,
                label='Class 0 observations',
                vmin=0., vmax=1.,
                cmap=plt.viridis())
    plt.scatter([hi - marg, lo + marg],
                [hi - marg, lo + marg],
                c=[1., 1.],
                s=120.,
                edgecolors='white',
                linewidth=2,
                label='Class 1 observations',
                vmin=0., vmax=1.,
                cmap=plt.viridis())
    plt.legend(loc=(0.2, 0.05), handletextpad=0.1, fontsize=11)
    plt.xticks([lo + marg, hi - marg])
    plt.yticks([lo + marg, hi - marg])
    plt.colorbar()
    plt.savefig(os.path.join(plot_dir, 'xor.png'), bbox_inches='tight')
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
        gbpnet_te, y_te, lin_w, lin_b, valacc = test_classification(*args, **kwargs)
        print(f'Energy after testing: {gbpnet_te.energy(as_numpy=True, sum_all=False)}')
        return gbpnet_te, y_te, valacc

    def run_fn(args, model_init=None, no_tr=False):
        sys.stderr = open(os.path.join(args.experiment.results_dir, f'err.txt'), 'w', buffering=1)
        with open(os.path.join(args.experiment.results_dir, f'out.txt'), 'w', buffering=1) as fo:
            with redirect_stdout(fo):
                train_data, test_data = load_data(n_te_points=args.experiment.batchsize_test)
                return exp_main(gbp_net_init=model_init,
                                args=args,
                                train_data=None if no_tr else train_data,
                                test_data=test_data,
                                eval_fn=eval,
                                plot_extra=plot_xor)

    model = None
    if conf.pretrain is not None:
        if conf.pretrain.n_iters_per_train_batch is not None:
            model = pretrain_fn(conf, run_fn)


    run_fn(args=conf, model_init=model)


if __name__ == '__main__':
    main()
