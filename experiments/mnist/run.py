# coding=utf-8
"""Training GBP nets on MNIST"""
import os
from copy import deepcopy
from contextlib import redirect_stdout
import json
import sys

import numpy as np
import tensorflow as tf
import time

from core.utils.data import get_batch
from core.utils.utils import set_allow_gpu_mem_growth, dotdict

from experiments.mnist.config import get_config
from experiments.mnist.data import load_data
from core.utils.saveload import save_gbp_net, load_checkpoint
from experiments.utils.utils import get_n_eval_steps, create_results_dir, logit_perf, create_or_concat_npy, get_batchsize, get_n_batches
from experiments.utils.graph_setup_utils import init_model
from experiments.utils.run_exp import exp_main
# from experiments.mnist.linear import train_linear_classifier, test_linear_classifier

set_allow_gpu_mem_growth()


def test_on_batch(prev_net,
                  xtest,
                  labelstest,
                  args_te,
                  n_eval_breaks,
                  n_itr_per_loop,
                  evaluate_fn,
                  test_batch_id):
    test_net = init_model(args_te,
                          x=xtest,
                          model_to_reinit=prev_net,
                          prec_rescaling=args_te.experiment.precision_rescaling_test,
                          prec_rescaling_conv_only=args_te.experiment.precision_rescaling_conv_only,
                          force_new_graph=True,
                          batch_id=test_batch_id)

    if args_te.experiment.fix_params_for_testing and args_te.inference == 'gbp':
        test_net.fix_layer_params()

    scores = []
    # scores_lin_class = []

    if args_te.inference == 'backprop':
        pred = test_net(xtest)
        print(tf.reduce_mean(pred, axis=0))
        model_to_test = dotdict(layers=[dotdict(coeff_vars=dotdict(mu=pred, sigma=tf.zeros_like(pred)))])
        score_eval = \
            evaluate_fn(model_to_eval=model_to_test,
                        labels=labelstest,
                        itr=0,
                        test_batch_id=test_batch_id)
        scores.append(score_eval)
        return test_net, scores

    evaluate_fn(model_to_eval=test_net,
                labels=labelstest,
                itr=0,
                test_batch_id=test_batch_id)
    t_test = 0.
    for i in range(max(n_eval_breaks, 1)):
        itrs = n_itr_per_loop * i

        if args_te.experiment.profile_test:
            print('Generating tensorboard profile')
            tf.profiler.experimental.start(os.path.join(args_te.experiment.tf_logdir, 'test'))
            n_itr_per_loop = 3

        t_bef = time.time()
        test_net.run_inference(tf.convert_to_tensor(n_itr_per_loop),
                               n_iters_start=tf.convert_to_tensor(itrs),
                               use_static_graph=args_te.use_static_graph)
        t_test += time.time() - t_bef

        if args_te.experiment.profile_test:
            # If tf_logdir given, do only profiling then exit. No testing
            tf.profiler.experimental.stop()
            sys.exit(0)

        score_eval = \
            evaluate_fn(model_to_eval=test_net,
                        labels=labelstest,
                        itr=itrs + n_itr_per_loop,
                        test_batch_id=test_batch_id)
        scores.append(score_eval)

    print(f'Testing GBP took {t_test} seconds to do {args_te.n_iters_per_test_batch} iters')
    return test_net, scores


def test_classification(
        model,
        train_itr,
        weights_lin_class,
        bias_lin_class,
        args_test,
        test_data,
        train_batch_id=0):
    for cl in test_data:
        # Go back to the beginning of the test set for each class
        if not isinstance(cl, int):
            # For other entries in test_data which include dset info
            continue
        print(f'Reinitialising test set for class {cl}')
        test_data[cl].reinit()
    te_classes = list(test_data.keys())
    print(f'Train batch number: {train_batch_id}, train iter: {train_itr}, running testing.')
    print(f'Test classes: {te_classes}.\n')

    n_examples_per_batch = get_batchsize(args_test, train_or_test='test')
    n_test_batches = get_n_batches(args_test, test_data, n_examples_per_batch, train_or_test='test')
    total_iters_te = args_test.n_iters_per_test_batch * n_test_batches
    n_itr_per_eval_te = total_iters_te // args_test.n_test_eval_breaks
    # n_examples_per_class_per_batch = args_test.experiment.examples_per_class_test // n_test_batches

    # print('Training linear classifier')
    # Train a linear classifier to predict class from last layer activations
    last_layer_id = -(1 + 2 * int(args_test.experiment.with_gbp_softmax_layer))
    # last_layer_coeffs = model.layers[last_layer_id].coeff_vars.mu
    # y_train_batch = model.layers[-1].softmax_factor.one_hot
    # weights_lin_class, bias_lin_class, _, _ = \
    #     train_linear_classifier(x=last_layer_coeffs,
    #                             y=y_train_batch,
    #                             weight_init=weights_lin_class,
    #                             bias_init=bias_lin_class)

    net_old = model
    t_pre_test = time.time()

    test_score_filepath = os.path.join(args_test.experiment.results_dir, 'test_scores.npy')
    def evaluate_fn(model_to_eval, labels, test_batch_id, itr):
        to_prepend = [train_batch_id, train_itr, test_batch_id, itr]
        # last_layer_coeffs_te = model.layers[-2 if args_test.experiment.with_gbp_softmax_layer else -1].coeff_vars.mu
        # acc_te, ll_te = test_linear_classifier(weights_lin_class, bias_lin_class, last_layer_coeffs_te, labels)
        # create_or_concat_npy(arr=to_prepend + [acc_te.numpy(), ll_te.numpy()],
        #                      npy_path=test_score_filepath.replace('.npy', '_lin_class.npy'))
        # print(f'\tTest iter: {itr}, Offline linear classifier: acc: {acc_te.numpy()}, loglik: {ll_te.numpy()}')
        scores = None
        if args_test.experiment.with_gbp_softmax_layer or args_test.inference == 'backprop':
            loglik_samples_test, loglik_logit_mu_test, acc_samples_test, acc_logit_mu_test = \
                logit_perf(model_to_eval.layers[-1].coeff_vars,
                           labels=labels,
                           test_classes=te_classes if args_test.experiment.no_task_crosstalk_test else None)
            print(f'\tPreds test scores: '
                  f'log(avg(lik(logits))): {loglik_samples_test}, '
                  f'loglik(avg(logits)): {loglik_logit_mu_test}, '
                  f'acc(avg(probs)): {acc_samples_test}, '
                  f'acc(avg(logits)): {acc_logit_mu_test}')
            scores = to_prepend + [acc_samples_test,
                                   acc_logit_mu_test.numpy(),
                                   loglik_samples_test.numpy(),
                                   loglik_logit_mu_test.numpy()]
            create_or_concat_npy(arr=scores,
                                 npy_path=test_score_filepath)
        return scores

    batch_scores = []
    for batch_id_te in range(n_test_batches):
        print(f'Test batch {batch_id_te + 1} of {args_test.n_test_batches}')

        # Test set - sample some test images of each class for testing (dataset was shuffled when loaded)
        x_test, test_labels, _, ds_end = \
            get_batch(test_data,
                      n_examples=n_examples_per_batch,
                      n_examples_so_far=n_examples_per_batch * batch_id_te,
                      n_classes=args_test.experiment.n_classes,
                      classes_sub=args_test.experiment.test_classes_subset,
                      shuffle_batch=args_test.experiment.shuffle_batches,
                      shuffle_batch_seed=abs(args_test.experiment.dataset_shuffle_seed - train_batch_id ** 2 + batch_id_te + 3))

        if ds_end:
            print(f'MNIST testing: dataset is finished after {batch_id_te} '
                  f'batches of size {n_examples_per_batch}. '
                  f'Should have completed {args_test.n_test_batches}')
            break

        # How many times to stop and evaluate during test-time GBP?
        total_itrs_post_batch = (batch_id_te + 1) * args_test.n_iters_per_test_batch
        n_eval_steps = get_n_eval_steps(n_iters_per_batch=args_test.n_iters_per_test_batch,
                                        n_total_iters_post_batch=total_itrs_post_batch,
                                        n_iters_per_eval=n_itr_per_eval_te)

        net_test, test_scores = \
            test_on_batch(prev_net=net_old,
                          xtest=x_test,
                          labelstest=test_labels,
                          args_te=args_test,
                          n_eval_breaks=max(n_eval_steps, 1),   # Have to evaluate each batch at least once
                          n_itr_per_loop=min(args_test.n_iters_per_test_batch, n_itr_per_eval_te),
                          evaluate_fn=evaluate_fn,
                          test_batch_id=batch_id_te)
        batch_scores.append(test_scores[-1])  # Acc and LLs after final test GBP iter

        # For filtering before next batch
        net_old = net_test

    t_test = time.time() - t_pre_test
    print(f'MNIST testing: dataset is finished after {batch_id_te + 1} '
          f'batches of size {n_examples_per_batch}.')
    print(f'Testing took {round(t_test, 4)} seconds in total.')
    val_acc = tf.reduce_mean(batch_scores, axis=0)[-3]   # -3 is index of acc of logit mean
    return net_test, test_labels, weights_lin_class, bias_lin_class, val_acc

def plot_first_layer_recon(model, plot_dir):
    img_in = model.layers[0].pixel_obs_factor.obs
    filts = model.layers[0].filter_vars.mu
    coeffs = model.layers[0].coeff_vars.mu
    print(img_in.shape, filts.shape, coeffs.shape, plot_dir)
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plot_dir = os.path.join(plot_dir, 'recons')
    os.mkdir(plot_dir)
    crops = [[[13, 16], [13, 16]], [[28 - 23, 28 - 20], [15, 18]]]
    for i, im in enumerate(img_in):
        recon_imdir = os.path.join(plot_dir, f'recon{i}')
        os.mkdir(recon_imdir)
        plt.imshow(im[..., 0], vmin=0, vmax=1.)
        plt.axis('off')

        plt.savefig(os.path.join(recon_imdir, f'img{i}.png'), dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close('all')
        fig, ax = plt.subplots()
        ax.imshow(im[..., 0], vmin=0, vmax=1.)
        plt.axis('off')
        for crop in crops:
            rect = patches.Rectangle([crop[1][0] - 0.5, crop[0][0] - 0.5],
                                     crop[0][1] - crop[0][0],
                                     crop[1][1] - crop[1][0],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.savefig(os.path.join(recon_imdir, f'img{i}_box.png'), dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close('all')

        for cr, crop in enumerate(crops):
            plt.imshow(im[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], 0], vmin=0., vmax=1.)
            plt.axis('off')
            plt.savefig(os.path.join(recon_imdir, f'img{i}_crop{cr}.png'), dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close('all')

        for c, co in enumerate(np.transpose(coeffs[i], (2, 0, 1))):
            plt.imshow(co, vmin=-1., vmax=1.)
            plt.axis('off')
            plt.savefig(os.path.join(recon_imdir, f'coeff{c}.png'), dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close('all')
        for cr, crop in enumerate(crops):
            coeff_crop_recon = coeffs[i][crop[0][0], crop[1][0]]
            np.savetxt(os.path.join(recon_imdir, f'coeff_crop{cr}_recon.csv'),
                       coeff_crop_recon,
                       delimiter=',')

    for f, filt in enumerate(np.transpose(filts, (3, 0, 1, 2))):
        plt.imshow(filt[..., 0], vmin=-0.5, vmax=0.5)
        plt.axis('off')
        plt.savefig(os.path.join(plot_dir, f'filt{f}.png'), dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close('all')


def main():
    # Set up results dir
    conf = get_config()
    if conf.experiment.config_filepath is not None:
        with open(conf.experiment.config_filepath, 'r') as confjson:
            conf_loaded = dotdict.from_dict(json.load(confjson))
            print(conf_loaded)

        conf_loaded.experiment.fix_params_for_testing = conf.experiment.fix_params_for_testing
        conf_loaded.experiment.remaining_tasks_train = conf.experiment.remaining_tasks_train
        conf_loaded.experiment.test_only = conf.experiment.test_only
        conf_loaded.experiment.load_checkpoint_dir = conf.experiment.load_checkpoint_dir
        conf_loaded.experiment.examples_per_class_test = conf.experiment.examples_per_class_test
        conf_loaded.n_test_batches = conf.n_test_batches
        conf_loaded.n_test_eval_breaks = conf.n_test_eval_breaks
        conf_loaded.n_iters_per_test_batch = conf.n_iters_per_test_batch
        results_dir = conf_loaded.experiment.results_dir if os.path.exists(conf_loaded.experiment.results_dir) else conf.experiment.results_dir
        conf = conf_loaded
        conf.experiment.results_dir = os.path.join(results_dir, 'rerun/')
    conf.experiment.results_dir = create_results_dir(conf.experiment.results_dir)

    print(f'\n\nOutput to logdir: {conf.experiment.results_dir}\n\n')

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
            tr_classes=None,
            te_classes=None,
            no_te=False,
            no_tr=False,
            unsupervised=False,
            load_from_cp=False):
        if tr_classes is not None or te_classes is not None:
            config_temp = deepcopy(config)
            te_class_str = 'all' if te_classes is None else '_'.join([str(c) for c in te_classes])
            classes_subdir_tr = "" if no_tr else f"tr_{'_'.join([str(c) for c in tr_classes])}"
            class_subdir_te = "" if no_te else f"te_{te_class_str}"
            if not no_te and not no_tr:
                classes_subdir = f"{classes_subdir_tr}_{class_subdir_te}"
            else:
                classes_subdir = classes_subdir_tr + class_subdir_te
            config_temp.experiment.results_dir = \
                os.path.join(config_temp.experiment.results_dir, classes_subdir)
            config_temp.experiment.train_classes_subset = tr_classes
            config_temp.experiment.test_classes_subset = list(range(10)) if te_classes is None else te_classes
            create_results_dir(config_temp.experiment.results_dir)
        else:
            config_temp = conf

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
                              examples_per_class_test=config_temp.experiment.examples_per_class_test)

                if tr_classes is not None:
                    train_data = {c: train_data[c] for c in tr_classes + type(tr_classes)(['examples_per_class'])}
                if te_classes is not None:
                    test_data = {c: test_data[c] for c in te_classes + type(te_classes)(['examples_per_class'])}

                if unsupervised:
                    assert tr_classes is None
                    assert te_classes is None

                    train_data = np.concatenate([np.array([x for x in t]) for t in train_data.values() if not isinstance(t, dict)], axis=0)
                    te_to_stack = []
                    for t in test_data.values():
                        if not isinstance(t, dict):
                            while True:
                                try:
                                    te_to_stack.append(next(t))
                                except StopIteration:
                                    break
                    test_data = np.concatenate(te_to_stack, axis=0)

                if load_from_cp:
                    model_in, model_spec = \
                        load_checkpoint(dataset=test_data if no_tr else train_data,
                                        checkpoint_dir=config_temp.experiment.load_checkpoint_dir,
                                        args=config_temp,
                                        classes_sub=te_classes if no_tr else te_classes)
                    config_temp.experiment.load_checkpoint_dir = None  # Prevents checkpoint being reloaded in `exp_main()`

                model_out = exp_main(args=config_temp,
                                     gbp_net_init=model_in,
                                     train_data=train_data if not (config_temp.experiment.test_only or no_tr) else None,
                                     test_data=None if no_te else test_data,
                                     eval_fn=None if no_te else eval)
                return model_out, config_temp.experiment.results_dir

    if conf.exp_type == 'split_mnist':
        model = None
        resume_from_checkpoint = conf.experiment.load_checkpoint_dir is not None
        tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        if resume_from_checkpoint and not conf.experiment.test_only:
            taskid_resume = tasks.index(tuple(conf.experiment.remaining_tasks_train[0]))
        for taskid_tr, tr_binary_task in enumerate(tasks):
            load_from_cp = False
            if resume_from_checkpoint:
                if conf.experiment.test_only:
                    for te_binary_task in tasks:
                        # Test on all tasks (even tasks not trained on yet)
                        run(config=conf, te_classes=te_binary_task, model_in=model, no_tr=True, load_from_cp=True)
                    break
                elif taskid_resume > taskid_tr:
                    continue
                elif taskid_resume == taskid_tr:
                    load_from_cp = True
                    print(f"Resuming from task {tr_binary_task}")
            model, logdir = run(config=conf,
                                tr_classes=tr_binary_task,
                                model_in=model,
                                no_te=True,
                                load_from_cp=load_from_cp)
            if load_from_cp:
                # Prevents checkpoint being reloaded in future, now `model` is trained
                conf.experiment.load_checkpoint_dir = None
            conf_temp = deepcopy(conf)
            conf_temp.experiment.results_dir = logdir
            tasks_te = tasks if conf.experiment.test_on_future_tasks else tasks[:taskid_tr + 1]
            for te_binary_task in tasks_te:
                # Test on all tasks (even tasks not trained on yet)
                run(config=conf_temp, te_classes=te_binary_task, model_in=model, no_tr=True)
    if conf.exp_type == 'unsupervised':
        model, logdir = run(config=conf, unsupervised=True, no_te=True)
        plot_first_layer_recon(model, logdir)
    else:
        run(config=conf)


if __name__ == '__main__':
    main()
