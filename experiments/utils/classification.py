# coding=utf-8
import os
import sys
import tensorflow as tf
import time

from core.utils.data import get_batch
from core.utils.utils import dotdict

from experiments.utils.utils import get_n_eval_steps, logit_perf, create_or_concat_npy, get_batchsize, get_n_batches
from experiments.utils.graph_setup_utils import init_model


def test_on_batch(prev_net,
                  xtest,
                  labelstest,
                  args_te,
                  n_eval_breaks,
                  n_itr_per_loop,
                  evaluate_fn,
                  test_batch_id,
                  corrupt_mask=None):
    test_net = init_model(args_te,
                          x=xtest,
                          model_to_reinit=prev_net,
                          prec_rescaling=args_te.experiment.precision_rescaling_test,
                          prec_rescaling_conv_only=args_te.experiment.precision_rescaling_conv_only,
                          force_new_graph=test_batch_id == 0,
                          batch_id=test_batch_id,
                          input_mask=corrupt_mask,
                          y=labelstest if args_te.experiment.conditional_generation_test else None)
    print(f'Energy before testing: {test_net.energy(as_numpy=True, sum_all=False)}')

    if args_te.experiment.fix_params_for_testing:
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
        if args_te.inference == 'gbp':
            print('Running testtime GBP')
            test_net.run_inference(n_iters=tf.convert_to_tensor(n_itr_per_loop),
                                 use_static_graph=args_te.use_static_graph,
                                 n_iters_start=tf.convert_to_tensor(itrs),
                                 xla_compile=args_te.xla_compile)
            print(f'Energy after testtime loop {i}: {test_net.energy(as_numpy=True, sum_all=False)}')
        elif args_te.inference == 'gd':
            print('Running testtime gradient descent')
            test_net.run_gd_inference(n_iters=tf.convert_to_tensor(n_itr_per_loop),
                                    use_static_graph=args_te.use_static_graph,
                                    n_iters_start=tf.convert_to_tensor(itrs),
                                    xla_compile=args_te.xla_compile,
                                    learning_rate=args_te.gd_lr,
                                    optim=args_te.gd_optim)
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
    print(f'Energy after testing: {test_net.energy(as_numpy=True, sum_all=False)}')
    return test_net, scores


def test_classification(
        model,
        train_itr,
        weights_lin_class,
        bias_lin_class,
        args_test,
        test_data,
        train_batch_id=0,
        pred_only=False):
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
        if pred_only or args_test.experiment.conditional_generation_test:
            return [scores]
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
        x_test, test_labels, _, ds_end, corr_mask = \
            get_batch(test_data,
                      unknown_classes=pred_only,
                      n_examples=n_examples_per_batch,
                      n_examples_so_far=n_examples_per_batch * batch_id_te,
                      n_classes=args_test.experiment.n_classes,
                      classes_sub=args_test.experiment.test_classes_subset,
                      shuffle_batch=args_test.experiment.shuffle_batches,
                      shuffle_batch_seed=abs(args_test.experiment.dataset_shuffle_seed - train_batch_id ** 2 + batch_id_te + 3),
                      corrupt_inputs=args_test.experiment.corrupt_test_inputs)

        if ds_end:
            print(f'Classification testing: dataset is finished after {batch_id_te} '
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
                          test_batch_id=batch_id_te,
                          corrupt_mask=corr_mask)
        batch_scores.append(test_scores[-1])  # Acc and LLs after final test GBP iter

        # For filtering before next batch
        net_old = net_test

    t_test = time.time() - t_pre_test
    print(f'Classification testing: dataset is finished after {batch_id_te + 1} '
          f'batches of size {n_examples_per_batch}.')
    print(f'Testing took {round(t_test, 4)} seconds in total.')
    if pred_only or args_test.experiment.conditional_generation_test:
        val_acc = None
    else:
        val_acc = tf.reduce_mean(batch_scores, axis=0)[-3]   # -3 is index of acc of logit mean
    return net_test, test_labels, weights_lin_class, bias_lin_class, val_acc
