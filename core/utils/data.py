# coding=utf-8
import numpy as np
import tensorflow as tf


def get_batch_classification(dataset,
                             n_examples,
                             n_examples_so_far=0,
                             n_classes=10,
                             classes_sub=None,
                             shuffle_batch=False,
                             shuffle_batch_seed=None):

    # Only sample batch from a subset of classes
    n_classes_sample = len(classes_sub) if classes_sub is not None else n_classes

    # Start labels from class after that of last example in prev batch
    label_start = n_examples_so_far % n_classes_sample

    if classes_sub is None:
        label_map = dict(enumerate(range(n_classes)))
    else:
        # Map `labels` [0, 1, 2..., C] to the subset of possible classes
        label_map = dict(enumerate(classes_sub))

    def _inc_all_classes(cls_list):
        all_poss_classes = classes_sub if classes_sub is not None else list(range(n_classes))
        for c in all_poss_classes:
            if c not in cls_list:
                return False
        return True

    labels = []
    x_batch = []
    classes_finished = []
    n = 0
    while len(x_batch) < n_examples and not _inc_all_classes(classes_finished):
        label_id = (n + label_start) % n_classes_sample
        label = label_map[label_id]
        n += 1
        if label in classes_finished:
            continue
        try:
            x_b = next(dataset[label])
            x_batch.append(x_b)
            labels.append(label)
        except StopIteration:
            if label not in classes_finished:
                classes_finished.append(label)

    end_of_ds = _inc_all_classes(classes_finished) and len(x_batch) == 0
    if end_of_ds:
        x = np.empty(shape=(0, 28, 28, 1)).astype(np.float32)
        labels = np.empty(shape=(0,))
        end_of_ds = True
    else:
        x = np.stack(x_batch, axis=0).astype(np.float32)
        if shuffle_batch:
            x, labels = shuffle_fn(x, np.array(labels), shuffle_batch_seed=shuffle_batch_seed)

    x = tf.convert_to_tensor(x)
    y = tf.one_hot(list(labels), depth=n_classes)

    return x, y, end_of_ds


def shuffle_fn(*args, shuffle_batch_seed=None):
    if shuffle_batch_seed is not None:
        np.random.seed(shuffle_batch_seed)
    shuffle_idx = list(range(args[0].shape[0]))
    np.random.shuffle(shuffle_idx)
    return [a[shuffle_idx] for a in args]


def get_batch_unsupervised(dataset,
                           n_examples,
                           n_examples_so_far=0,
                           shuffle_batch=False,
                           n_classes=None,
                           shuffle_batch_seed=None):
    x = dataset[n_examples_so_far:n_examples + n_examples_so_far]
    if isinstance(dataset, tf.Tensor):
        x = x.numpy().astype(np.float32)

    if shuffle_batch:
        x = shuffle_fn(x, shuffle_batch_seed=shuffle_batch_seed)
    x = tf.convert_to_tensor(x)
    end_of_ds = x.shape[0] == 0
    if len(x.shape) == 5:
        x = x[0]

    return x, end_of_ds


def get_batch_segmentation(dataset,
                           n_examples,
                           n_examples_so_far=0,
                           shuffle_batch=False,
                           n_classes=None,
                           shuffle_batch_seed=None):
    x = get_batch_unsupervised(dataset, n_examples, n_examples_so_far, shuffle_batch)

    # Split img from seg labels
    x, y = x[..., :-n_classes], x[..., -n_classes:]
    if shuffle_batch:
        x, y = shuffle_fn(x, y, shuffle_batch_seed=shuffle_batch_seed)

    assert x.shape[-1] in (1, 3)

    end_of_ds = x.shape[0] == 0
    return x, y, end_of_ds


def get_batch(dataset,
              n_examples,
              n_examples_so_far=0,
              shuffle_batch=False,
              shuffle_batch_seed=None,
              n_classes=None,
              n_seg_classes=None,
              classes_sub=None):
    y = None
    y_seg = None
    if classes_sub is not None:
        assert isinstance(dataset, dict), \
            "`classes_sub` only supported for classification dataset"
    if isinstance(dataset, dict):
        dataset, y, end_of_ds = \
            get_batch_classification(dataset=dataset,
                                     n_examples=n_examples,
                                     n_examples_so_far=n_examples_so_far,
                                     n_classes=n_classes,
                                     shuffle_batch=shuffle_batch,
                                     shuffle_batch_seed=shuffle_batch_seed,
                                     classes_sub=classes_sub)
        n_examples = dataset.shape[0]
        n_examples_so_far = 0
        shuffle_batch = False
        dataset = dataset.numpy()

    if isinstance(dataset, np.ndarray) and n_seg_classes is not None:
        if dataset.shape[-1] in (n_seg_classes + 1, n_seg_classes + 3):
            dataset, y_seg, end_of_ds = get_batch_segmentation(dataset=dataset,
                                                               n_examples=n_examples,
                                                               n_examples_so_far=n_examples_so_far,
                                                               n_classes=n_seg_classes,
                                                               shuffle_batch=shuffle_batch,
                                                               shuffle_batch_seed=shuffle_batch_seed)

    else:
        dataset, end_of_ds = get_batch_unsupervised(dataset=dataset,
                                                    n_examples=n_examples,
                                                    n_examples_so_far=n_examples_so_far,
                                                    n_classes=n_classes,
                                                    shuffle_batch=shuffle_batch,
                                                    shuffle_batch_seed=shuffle_batch_seed)

    return tf.convert_to_tensor(dataset), y, y_seg, end_of_ds


# Map experiment type to batch getter fn
get_batch_fns = {'denoising': get_batch_unsupervised,
                 'segmentation': get_batch_segmentation,
                 'classification': get_batch_classification}
