# coding=utf-8
from copy import deepcopy
from contextlib import redirect_stdout
from functools import partial
import json
import numpy as np
import os
import sys
import tensorflow as tf
import time

from core.utils.data import get_batch
from core.utils.plot import plot_model_diagnostics
from core.utils.saveload import save_gbp_net, load_checkpoint
from experiments.utils.utils import create_plot_dirs, pretty_print_args, \
    check_test_schedule, check_train_schedule, logit_perf, get_n_eval_steps, to_float_or_int, get_batchsize, get_n_batches
from experiments.utils.graph_setup_utils import init_model, update_edge_tracking

MIN_N_BATCH_TO_PLOT_LAST = 5


def get_args_test(argdict):
    args_test = deepcopy(argdict)
    for f in args_test.factors:
        if 'sigma_test_mult' in args_test.factors[f]:
            args_test.factors[f].sigma *= args_test.factors[f].sigma_test_mult
    for layconf in args_test.architecture:
        if 'factors' in layconf:
            for f in layconf.factors:
                if 'sigma_test_mult' in layconf.factors[f]:
                    layconf.factors[f].sigma *= layconf.factors[f].sigma_test_mult

    # Remove softmax head for testing (where we don't have class observation)
    # If using factor graph
    if args_test.experiment.with_gbp_softmax_layer and args_test.inference == 'gbp':
        args_test.architecture = args_test.architecture[:-1]

        # Remove softmax head layer from message schedule
        if args_test.layer_schedule is not None:
            n_layers = len(args_test.architecture)
            args_test.layer_schedule = [s for s in args_test.layer_schedule if s < n_layers]

    return args_test


def train_on_batch(gbpnet,
                   n_itr_per_loop,
                   args,
                   x_train,
                   y_train=None,
                   y_train_seg=None,
                   n_itrs_start=0):
    if args.experiment.tf_logdir is not None:
        print('Generating tensorboard profile')
        tf.profiler.experimental.start(args.experiment.tf_logdir)
        n_itr_per_loop = 3

    t_before_train = time.time()
    if args.inference == 'gbp':
        print('Running GBP')
        gbpnet.run_inference(n_iters=tf.convert_to_tensor(n_itr_per_loop),
                             x_img=x_train,
                             maybe_y_label=y_train,
                             maybe_y_seg_label=y_train_seg,
                             use_static_graph=args.use_static_graph,
                             n_iters_start=tf.convert_to_tensor(n_itrs_start),
                             xla_compile=args.xla_compile)
    elif args.inference == 'gd':
        print('Running gradient descent')
        gbpnet.run_gd_inference(n_iters=tf.convert_to_tensor(n_itr_per_loop),
                                use_static_graph=args.use_static_graph,
                                n_iters_start=tf.convert_to_tensor(n_itrs_start),
                                xla_compile=args.xla_compile,
                                learning_rate=args.gd_lr,
                                optim=args.gd_optim)

    t_after_train = time.time()
    print(f'Training for {n_itr_per_loop} iters took {round(t_after_train - t_before_train, 4)} seconds')

    if args.experiment.tf_logdir is not None:
        # If tf_logdir given, do only profiling then exit. No testing
        tf.profiler.experimental.stop()
        if not args.experiment.profile_test:
            sys.exit(0)

    if args.experiment.with_gbp_softmax_layer:
        eval_classes = args.experiment.train_classes_subset if args.experiment.no_task_crosstalk_train else None
        loglik_samples_train, ll_logit_mu_train, acc_samples_train, acc_logit_mu_train = \
            logit_perf(gbpnet.layers[-2].coeff_vars,
                       labels=y_train,
                       test_classes=eval_classes)
        print(f'Preds train scores: '
              f'log(avg(lik(logits))): {loglik_samples_train}, '
              f'loglik(avg(logits)): {ll_logit_mu_train}, '
              f'acc(avg(probs)): {acc_samples_train}, '
              f'acc(avg(logits)): {acc_logit_mu_train}')

    return gbpnet


def save_checkpoint(model, batch_id, args, best=False):
    checkpoint_dir = os.path.join(args.experiment.results_dir,
                                  'train',
                                  'models')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    savedir = os.path.join(checkpoint_dir, 'best' if best else f'batch_{batch_id}')
    print(f'\nSaving model in dir: {savedir}\n')
    save_gbp_net(gbp_net=model,
                 args=args,
                 n_batch_so_far=batch_id,
                 n_iter_current_batch=0,
                 params_only=True,
                 savedir=savedir)


def train_and_eval(args, train_data, test_fn, model_tr=None, plot_fn=None):
    edge_stats = None
    model_te = None

    examples_per_batch = get_batchsize(args)
    n_train_batches = get_n_batches(args, train_data, examples_per_batch)

    total_iters = args.n_iters_per_train_batch * n_train_batches
    n_iters_per_eval = total_iters // args.n_train_eval_breaks
    print(f'Train iterations per eval: {n_iters_per_eval}')

    # shuffle_seed = args.experiment.dataset_shuffle_seed if args.experiment.shuffle_batches else None
    batcher_kwargs = dict(dataset=train_data,
                          n_examples=examples_per_batch,
                          n_classes=args.experiment.n_classes,
                          classes_sub=args.experiment.train_classes_subset,
                          n_seg_classes=args.experiment.n_seg_classes,
                          shuffle_batch=args.experiment.shuffle_batches)
    # x_train_batch, maybe_y_train_batch, maybe_y_train_batch_seg, _ = \
    #     get_batch(n_examples_so_far=0, shuffle_batch_seed=shuffle_seed, **batcher_kwargs)
    classes_sub_model_init = args.experiment.train_classes_subset if args.experiment.no_task_crosstalk_train else None
    # model_tr1 = init_model(config=args,
    #                       x=x_train_batch,
    #                       y=maybe_y_train_batch,
    #                       model_to_reinit=model_tr,
    #                       track_edges=args.experiment.plot_message_convergence,
    #                       prec_rescaling=args.experiment.precision_rescaling,
    #                       prec_rescaling_conv_only=args.experiment.precision_rescaling_conv_only,
    #                       segmentation_obs=maybe_y_train_batch_seg,
    #                       classes_sub=classes_sub_model_init,
    #                       force_new_graph=True,
    #                        batch_id=0)

    n_iters_per_plot = None
    best_val_score = None
    for batch_id in range(n_train_batches):
        # if batch_id == 0:
        #     try:
        #         print('GPU memory use:', tf.config.experimental.get_memory_info('GPU:0')['current'])
        #     except ValueError:
        #         # Not running on GPU
        #         pass
        print(f"Train batch {batch_id + 1} of {n_train_batches}")

        # Train set - sample a train image of each class (dataset was shuffled when loaded)
        so_far = examples_per_batch * batch_id
        shuffle_seed = args.experiment.dataset_shuffle_seed + batch_id if args.experiment.shuffle_batches else None
        x_train_batch, maybe_y_train_batch, maybe_y_train_batch_seg, ds_end = \
            get_batch(n_examples_so_far=so_far, shuffle_batch_seed=shuffle_seed, **batcher_kwargs)

        if ds_end:
            print(f'Dataset finished after {batch_id} '
                  f'batches of size {examples_per_batch}. '
                  f'Should have completed {args.n_train_batches}')
            break

        if args.inference == 'backprop':
            # Do simple GD update and eval NN, don't do the other stuff below
            model_tr.train_on_batch(x_train_batch, maybe_y_train_batch)
            eval_freq_batch = n_train_batches // args.n_train_eval_breaks
            if batch_id % eval_freq_batch == 0 and test_fn is not None:
                test_fn(model=model_tr,
                        train_itr=-1,
                        train_batch_id=batch_id)
            continue

        model_tr = init_model(config=args,
                              x=x_train_batch,
                              y=maybe_y_train_batch,
                              model_to_reinit=model_tr,
                              track_edges=args.experiment.plot_message_convergence,
                              prec_rescaling=args.experiment.precision_rescaling,
                              prec_rescaling_conv_only=args.experiment.precision_rescaling_conv_only,
                              segmentation_obs=maybe_y_train_batch_seg,
                              classes_sub=classes_sub_model_init,
                              batch_id=batch_id)

        iters_by_end = (batch_id + 1) * args.n_iters_per_train_batch
        n_eval_breaks = get_n_eval_steps(n_iters_per_batch=args.n_iters_per_train_batch,
                                         n_iters_per_eval=n_iters_per_eval,
                                         n_total_iters_post_batch=iters_by_end)
        n_itr_per_loop = min(n_iters_per_eval, args.n_iters_per_train_batch)
        n_tr_breaks = n_eval_breaks

        if args.n_train_plot_breaks is not None:
            n_iters_per_plot = total_iters // args.n_train_plot_breaks
            n_plot_breaks = get_n_eval_steps(n_iters_per_batch=args.n_iters_per_train_batch,
                                             n_iters_per_eval=n_iters_per_plot,
                                             n_total_iters_post_batch=iters_by_end)
            assert n_plot_breaks % n_eval_breaks == 0 or n_eval_breaks % n_plot_breaks == 0
            n_tr_breaks = max(n_plot_breaks, n_eval_breaks)
            n_itr_per_loop = min(n_iters_per_plot, n_itr_per_loop)
            plot_fn(model=model_tr,   # Initial plot
                    itr=0,
                    batch_id=batch_id,
                    edge_stats=edge_stats if args.experiment.plot_message_convergence else None)

        eval_freq = None if n_eval_breaks == 0 else n_tr_breaks // n_eval_breaks
        print(f'Energy before training: {model_tr.energy(as_numpy=True, sum_all=False)}')
        for tr_loop in range(max(n_tr_breaks, 1)):
            model_tr = \
                train_on_batch(model_tr,
                               n_itr_per_loop=n_itr_per_loop,
                               n_itrs_start=tr_loop * n_itr_per_loop,
                               args=args,
                               y_train=maybe_y_train_batch,
                               x_train=x_train_batch,
                               y_train_seg=maybe_y_train_batch_seg)

            print(f'Energy after training: {model_tr.energy(as_numpy=True, sum_all=False)}')
            itrs_so_far = (tr_loop + 1) * n_itr_per_loop

            if n_eval_breaks >= 1:
                if (tr_loop % eval_freq == 0 or tr_loop == max(n_tr_breaks, 1) - 1) and \
                        test_fn is not None:
                    # Results written to file inside test_fn()
                    model_te, y_class_te, valscore = \
                        test_fn(model=model_tr,
                                train_itr=(tr_loop + 1) * n_itr_per_loop,
                                train_batch_id=batch_id)

                    if args.experiment.save_best_model:
                        save_cp = False
                        if best_val_score is None:
                            save_cp = True
                        elif valscore > best_val_score:
                            save_cp = True
                        if save_cp:
                            save_checkpoint(model=model_tr,
                                            batch_id=batch_id + 1,
                                            args=args,
                                            best=True)
                            best_val_score = valscore

            if n_iters_per_plot is not None:
                if itrs_so_far % n_iters_per_plot == 0:
                    plot_fn(model=model_tr,
                            itr=itrs_so_far,
                            batch_id=batch_id,
                            edge_stats=edge_stats if args.experiment.plot_message_convergence else None)

        if args.experiment.plot_message_convergence:
            edge_stats = update_edge_tracking(model_tr, edge_stats=edge_stats)

        if args.experiment.plot_train_batch_freq is not None:
            if batch_id % args.experiment.plot_train_batch_freq == 0 or \
                    (batch_id == n_train_batches - 1 and
                     n_train_batches > MIN_N_BATCH_TO_PLOT_LAST):
                plot_fn(model=model_tr,
                        itr=args.n_iters_per_train_batch,
                        batch_id=batch_id,
                        edge_stats=edge_stats)
        if args.experiment.plot_test_batch_freq is not None and \
                model_te is not None:
            if batch_id % args.experiment.plot_test_batch_freq == 0:
                plot_fn(model=model_te,
                        itr=args.n_iters_per_train_batch,
                        batch_id=batch_id,
                        edge_stats=None,
                        test_itr=args.n_iters_per_test_batch,
                        test_batch_id=args.n_test_batches,
                        y_class=y_class_te)

        if args.experiment.checkpoint_frequency_batches is not None:
            if (batch_id + 1) % args.experiment.checkpoint_frequency_batches == 0 or \
                    batch_id == n_train_batches - 1:
                save_checkpoint(model=model_tr,
                                batch_id=batch_id + 1,
                                args=args)
    print(f'Dataset is finished after {batch_id + 1} batches of size {examples_per_batch}.')
    return model_tr, model_te, edge_stats


def exp_main(args, train_data,
             test_data=None,
             gbp_net_init=None,
             eval_fn=None,
             plot_subdirs=None,
             plot_extra=None):
    # Print config
    pretty_print_args(args)
    with open(os.path.join(args.experiment.results_dir, 'conf.txt'), 'w') as f:
        with redirect_stdout(f):
            pretty_print_args(args)

    # Save json file too
    with open(os.path.join(args.experiment.results_dir, 'conf.json'), 'w') as f:
        json.dump(args, f)

    check_train_schedule(args)

    train_only = test_data is None
    test_only = train_data is None

    if args.experiment.examples_per_class_train is not None:
        # Example per class args may be either float or int - convert from string
        args.experiment.examples_per_class_train = to_float_or_int(args.experiment.examples_per_class_train)

    # Folders to store results
    plot_subdirs = plot_subdirs or []
    if not test_only:
        train_plot_dir = os.path.join(args.experiment.results_dir, 'train')
        os.mkdir(train_plot_dir)
        create_plot_dirs(expdir=train_plot_dir,
                         subdirs=['coeffs', 'weights', 'msg_diffs'] + plot_subdirs,
                         n_layers=len(args.architecture))

    if not train_only:
        if args.experiment.examples_per_class_test is not None:
            args.experiment.examples_per_class_test = to_float_or_int(args.experiment.examples_per_class_test)
        args_test = get_args_test(args)
        check_test_schedule(args_test)
        test_plot_dir = os.path.join(args.experiment.results_dir, 'test')
        os.mkdir(test_plot_dir)
        create_plot_dirs(expdir=test_plot_dir,
                         subdirs=['coeffs', 'weights', 'msg_diffs'] + plot_subdirs,
                         n_layers=len(args_test.architecture))

    def plot_fn(model, itr, batch_id=None, edge_stats=None, test_batch_id=None, test_itr=None, y_class=None):
        train = test_itr is None and test_batch_id is None
        if plot_extra is not None:
            plot_extra(model,
                       plot_dir=train_plot_dir if train else test_plot_dir,
                       train_itr=itr,
                       train_batch_id=batch_id,
                       test_batch_id=test_batch_id,
                       test_itr=test_itr)
        itr_str = f"{batch_id}" if batch_id is None else f"batch{batch_id}_itr{itr}"
        plot_model_diagnostics(model,
                               itr_str,
                               train_plot_dir if train else test_plot_dir,
                               edge_stats=edge_stats,
                               plot_coeffs=args.experiment.plot_coeffs,
                               plot_weights=args.experiment.plot_weights,
                               plot_msg_convergence=args.experiment.plot_message_convergence,
                               y_class=y_class)

    test_kwargs = dict() if train_only else dict(args_test=args_test, test_data=test_data)
    test_fn = partial(eval_fn, **test_kwargs) if eval_fn is not None else None

    if args.experiment.load_checkpoint_dir is not None:
        classes_sub = args.experiment.train_classes_subset or args.experiment.test_classes_subset
        gbp_net_init, model_spec = \
            load_checkpoint(dataset=train_data or test_data,  # Whichever is not None
                            checkpoint_dir=args.experiment.load_checkpoint_dir,
                            args=args,
                            classes_sub=classes_sub)
    else:
        model_spec = None

    if test_only:
        # assert gbp_net_init is not None
        # TODO fix input_img and output args below, needed to load model but not used
        model_te, y_class_te, _ = \
            test_fn(model=gbp_net_init,
                    train_itr=-1,
                    train_batch_id=model_spec['n_batch_so_far'] if model_spec is not None else 0)
        model_tr = None
    else:
        tr_te_out = train_and_eval(args,
                                   train_data=train_data,
                                   model_tr=gbp_net_init,
                                   test_fn=test_fn,
                                   plot_fn=plot_fn)

        model_tr = tr_te_out[0]
        model_te = tr_te_out[1]
        edge_stats = tr_te_out[-1]

    total_iters = None
    if not test_only:
        print('Plotting train model')
        batchsize = get_batchsize(args)
        n_train_batches = get_n_batches(args, train_data, batchsize)
        total_iters = args.n_iters_per_train_batch * n_train_batches
        plot_model_diagnostics(model_tr, total_iters, train_plot_dir,
                               edge_stats=edge_stats,
                               plot_coeffs=args.experiment.plot_coeffs,
                               plot_weights=args.experiment.plot_weights,
                               plot_msg_convergence=args.experiment.plot_message_convergence)
    if not train_only:
        print('Plotting test model')
        plot_model_diagnostics(model_te, total_iters, test_plot_dir,
                               plot_coeffs=args.experiment.plot_coeffs,
                               plot_weights=args.experiment.plot_weights,
                               plot_msg_convergence=args.experiment.plot_message_convergence)

    return model_tr
