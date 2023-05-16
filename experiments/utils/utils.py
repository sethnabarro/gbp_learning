# coding=utf-8
import numpy as np
import os
import shutil
import tensorflow as tf

from core.utils.utils import dotdict as dd


def loglik_per_example(model_logprobs, labels):
    return tf.reduce_sum(model_logprobs * labels, axis=-1)


def accuracy(logits, labels):
    # Average probs over logit samples
    pred_prob_avg = tf.reduce_mean(tf.nn.softmax(logits, axis=-1), axis=0)
    pred_argmax = tf.argmax(pred_prob_avg, axis=-1, output_type=tf.int32)
    labels_argmax = tf.argmax(labels, axis=-1, output_type=tf.int32)
    return tf.reduce_mean(tf.cast(pred_argmax == labels_argmax, tf.float32))


def logit_perf(logits, labels, test_classes=None, seed=888):
    # Evaluate two log-likelihoods:
    #   1. Log of Monte Carlo estimate of expectation over logits: \log \mathbb{E}_{l\sim logits}[p(y| l)]
    #   2. Estimate using mean of logits: \log p(y | mean(logits))
    logit_mu = logits.mu
    logit_sigma = logits.sigma

    if test_classes is not None:
        print('Test classes', test_classes)
        # If classfn task over subset of labels - take only these logits
        logit_mu = tf.gather(logit_mu, indices=test_classes, axis=-1)
        logit_sigma = tf.gather(logit_sigma, indices=test_classes, axis=-1)

        # Map label space to zero-indexed integers
        # so accs/lls correctly calculated, e.g. {3, 5} -> {0, 1}
        label_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(sorted(test_classes), range(len(test_classes))),
            default_value=-1)
        labels_dense = tf.argmax(labels, axis=-1, output_type=tf.int32)  # One-hot -> dense
        labels_dense_sub = label_table[labels_dense]
        labels = tf.one_hot(labels_dense_sub, depth=len(test_classes))  # Convert back to one-hot

    n_logit_samples = 100
    standard_normal_samples = tf.random.normal([n_logit_samples] + logit_mu.shape.as_list(), seed=seed)
    logit_samples = logit_mu[None] + standard_normal_samples * logit_sigma[None]
    logprobs_samples = tf.nn.log_softmax(logit_samples, axis=-1)

    # Mean logprobs over dataset (per logit sample)
    loglik_per_sample = tf.reduce_mean(loglik_per_example(logprobs_samples, labels), axis=-1)

    # Loglik 1. Avg lhoods over samples, convert back to logprobs
    loglik_samples = tf.reduce_logsumexp(loglik_per_sample, axis=0) - np.log(float(n_logit_samples))

    # Loglik 2. Loglik of mean logits
    loglik_logit_mu = tf.reduce_mean(loglik_per_example(tf.nn.log_softmax(logit_mu), labels))

    # Accuracy according to logit mean
    acc_logit_mu = accuracy(logit_mu[None], labels)

    # Accuracy according to logit samples
    acc_samples = accuracy(logit_samples, labels)

    print(f'Predicted class probs (test): {tf.nn.softmax(logit_mu).numpy()}')

    return tf.reduce_mean(loglik_samples), loglik_logit_mu, acc_samples, acc_logit_mu


def pretty_print_args(conf, indent=None):
    indent = indent or ''
    for k, v in conf.items():
        if isinstance(v, dd):
            print(f'{indent}{k}:')
            pretty_print_args(v, indent=indent + '\t')
        elif isinstance(v, (list, tuple)):
            def _print_iterable(itrble, ind):
                print(f'{ind}{k}:')
                has_list_or_dict_elem = False
                for i, vv in enumerate(itrble):
                    if isinstance(vv, dd):
                        print(f'{ind}{i}:')
                        has_list_or_dict_elem = True
                        pretty_print_args(vv, indent=ind + '\t')
                    elif isinstance(vv, (tuple, list)):
                        print(f'{ind}{i}:')
                        has_list_or_dict_elem = True
                        _print_iterable(vv, ind + '\t')
                if not has_list_or_dict_elem:
                    print(f'{ind}\t{k}={itrble}')
            _print_iterable(v, indent)
        else:
            print(f'{indent}{k}={v}')


def check_train_schedule(args):
    if args.n_train_batches is None:
        return
    assert (args.n_train_eval_breaks % args.n_train_batches == 0) or \
        (args.n_train_batches % args.n_train_eval_breaks == 0), \
        "Must have integer number of eval steps per training batch or vice versa"


def check_test_schedule(args):
    if args.n_test_batches is None:
        return
    assert (args.n_test_eval_breaks % args.n_test_batches == 0) or \
           (args.n_test_batches % args.n_test_eval_breaks == 0), \
           "Must have integer number of eval steps per test batch or vice versa"


def get_n_eval_steps(n_iters_per_batch, n_total_iters_post_batch, n_iters_per_eval):
    if n_iters_per_batch >= n_iters_per_eval:
        n_eval_steps = int(n_iters_per_batch / n_iters_per_eval)
    else:
        n_eval_steps = int(n_total_iters_post_batch % n_iters_per_eval == 0)
    return n_eval_steps


def get_batchsize(args, train_or_test=None):
    train_or_test = train_or_test or 'train'
    examples_per_class = getattr(args.experiment, f'examples_per_class_{train_or_test}')
    n_batches = getattr(args, f'n_{train_or_test}_batches')
    if examples_per_class is not None and n_batches is not None:
        classes_subset = getattr(args.experiment, f'{train_or_test}_classes_subset')
        n_classes = args.experiment.n_classes if \
            classes_subset is None else \
            len(classes_subset)
        return int(examples_per_class * n_classes / n_batches)
    else:
        return getattr(args.experiment, f'batchsize_{train_or_test}')


def get_n_batches(args, dataset, batchsize, train_or_test=None):
    train_or_test = train_or_test or 'train'
    if getattr(args, f'n_{train_or_test}_batches') is not None:
        return getattr(args, f'n_{train_or_test}_batches')
    else:
        examples_per_class = getattr(args.experiment, f'examples_per_class_{train_or_test}')
        classes_subset = getattr(args.experiment, f'{train_or_test}_classes_subset')
        if examples_per_class is not None:
            if isinstance(dataset, dict):
                cl_labels = [c for c in dataset if c != 'examples_per_class']
                n_classes = len(cl_labels) if classes_subset is None else len(classes_subset)
                ds_size = examples_per_class * n_classes
        elif classes_subset is None:
            ds_size = sum([dataset['examples_per_class'][cl] for cl in dataset if isinstance(cl, int)]) if \
                isinstance(dataset, dict) else dataset.shape[0]
        else:
            ds_size = sum([dataset['examples_per_class'][cl] for cl in classes_subset])
        return int(np.ceil(ds_size / batchsize))


def dir_exists_not_empty(dirc):
    is_false = False
    if os.path.exists(dirc):
        if len(os.listdir(dirc)) != 0:  # empty
            return True
    return is_false


def create_new_subdir(dirc):
    subdir = 1
    while os.path.exists(os.path.join(dirc, str(subdir))):
        subdir += 1
    newdir = os.path.join(dirc, str(subdir))
    os.mkdir(newdir)
    return newdir


def create_plot_dirs(expdir, subdirs=None, n_layers=1, subdirs_extra=None):
    if subdirs is None:
        subdirs = ['comp_energies',
                   'filters',
                   'filter_energies',
                   'reconstructions',
                   'pre_act_recon',
                   'post_act_recon']
    if subdirs_extra is not None:
        subdirs += subdirs_extra
    for subd in subdirs:
        plot_subdir = os.path.join(expdir, subd)

        if os.path.exists(plot_subdir):
            # Delete old
            shutil.rmtree(plot_subdir)

        # Create new
        os.mkdir(plot_subdir)

        for layid in range(n_layers):
            if subd == 'reconstructions':
                # Don't need individual folder for reconstructions at each layer
                break
            else:
                plot_subdir_layer = os.path.join(plot_subdir, f"layer_{layid}")

            # Create new
            os.mkdir(plot_subdir_layer)


def create_results_dir(results_dir_path):
    """"""
    results_dir_path_new = results_dir_path
    if dir_exists_not_empty(results_dir_path):
        results_dir_path_new = create_new_subdir(results_dir_path)
        print(results_dir_path_new)
    else:
        if os.path.exists(results_dir_path):
            shutil.rmtree(results_dir_path)
        os.mkdir(results_dir_path)
    return results_dir_path_new


def create_or_concat_npy(arr: [np.ndarray, list], npy_path: str):
    """Concats an array to saved array and rewrites file"""
    if os.path.exists(npy_path):
        arr_load = np.load(npy_path)
        arr = np.concatenate([arr_load, arr], axis=0)
    np.save(npy_path, arr)


def to_float_or_int(string):
    num = float(string)
    if num.is_integer():
        num = int(num)
    return num
