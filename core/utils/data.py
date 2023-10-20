# coding=utf-8
import numpy as np
import tensorflow as tf

from core.utils.utils import corrupt_image


def get_batch_classification(dataset,
                             n_examples,
                             n_examples_so_far=0,
                             n_classes=10,
                             classes_sub=None,
                             shuffle_batch=False,
                             shuffle_batch_seed=None,
                             unknown_classes=False,
                             with_balanced_classes=True):

    if unknown_classes:
        end_of_ds = False
        x_batch = []
        for _ in range(n_examples):
            try:
                x_batch.append(next(dataset[None]))
            except StopIteration:
                end_of_ds = True
                break
        labels = -np.ones(len(x_batch), dtype=np.int32)
    else:
        # Only sample batch from a subset of classes
        n_classes_sample = len(classes_sub) if classes_sub is not None else n_classes

        # Start labels from class after that of last example in prev batch
        label_start = n_examples_so_far % n_classes_sample

        if classes_sub is None:
            label_map = dict(enumerate(range(n_classes)))
        else:
            # Map `labels` [0, 1, 2..., C] to the subset of possible classes
            label_map = dict(enumerate(classes_sub))

        all_poss_classes = classes_sub if classes_sub is not None else list(range(n_classes))
        def _inc_all_classes(cls_list):
            for c in all_poss_classes:
                if c not in cls_list:
                    return False
            return True

        labels = []
        x_batch = []
        classes_finished = []
        n_examples_class_got = {c: 0 for c in all_poss_classes}
        if with_balanced_classes:
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
                    n_examples_class_got[label] += 1
                except StopIteration:
                    if label not in classes_finished:
                        classes_finished.append(label)
            print(f'Batcher:\n\tgot class dist {n_examples_class_got}')

        else:
            # Sample from each class in proportion to its number of remaining examples
            n_remain = {c: dataset[c].n_remaining for c in dataset if c != 'examples_per_class'}
            if sum(n_remain.values()) < n_examples:
                # Set `n_examples_class_batch` to force `end_of_ds` to be True below, once all examples used
                n_examples_class_batch = {c: n_examples for c in all_poss_classes}
            else:
                np.random.seed(n_examples_so_far)  # Seed for sampling num examples of each class
                class_props = {c: n / sum(n_remain.values()) for c, n in n_remain.items()}
                n_examples_class_batch = {c: n for c, n in zip(all_poss_classes, np.random.multinomial(n=n_examples, pvals=[class_props[c] for c in all_poss_classes]))}
                while any([n_examples_class_batch[c] > dataset[c].n_remaining for c in n_examples_class_batch]):
                    for c in all_poss_classes:
                        to_adjust = n_examples_class_batch[c] > dataset[c].n_remaining
                        if to_adjust:
                            diff = n_examples_class_batch[c] - dataset[c].n_remaining
                            n_examples_class_batch[c] -= diff
                            for d in range(diff):
                                n_remain = {cc: dataset[cc].n_remaining - n_examples_class_batch[cc] for cc in
                                            all_poss_classes}
                                c_remain_max = [cc for cc in n_remain if n_remain[cc] == max(n_remain.values())][0]
                                n_examples_class_batch[c_remain_max] += 1
                            break

            n_examples_class_got = {c: 0 for c in n_examples_class_batch}
            for label in all_poss_classes:
                for _ in range(n_examples_class_batch[label]):
                    if label in classes_finished:
                        continue
                    try:
                        x_b = next(dataset[label])
                        x_batch.append(x_b)
                        labels.append(label)
                        n_examples_class_got[label] += 1
                    except StopIteration:
                        if label not in classes_finished:
                            classes_finished.append(label)
            print(f'Batcher:\n\tWant class dist {n_examples_class_batch}\n\tgot {n_examples_class_got}')

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


def get_batch_regression(dataset,
                         n_examples,
                         shuffle_batch=False,
                         shuffle_batch_seed=None):

    labels = []
    x_batch = []
    end_of_ds = False
    for i in range(n_examples):
        try:
            x, y = next(dataset)
            x_batch.append(x)
            labels.append(y)
        except StopIteration:
            end_of_ds = True
    x = np.concatenate(x_batch, axis=0).astype(np.float32)
    labels = np.array(labels)
    if shuffle_batch:
        x, labels = shuffle_fn(x, labels, shuffle_batch_seed=shuffle_batch_seed)
    return x, labels, end_of_ds


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
              classes_sub=None,
              unknown_classes=False,
              corrupt_inputs=None,
              with_balanced_classes=True):
    y = None
    y_seg = None
    for_classification = isinstance(dataset, dict)
    for_regression = isinstance(dataset, ReinitDataIterator)
    if classes_sub is not None:
        assert isinstance(dataset, dict), \
            "`classes_sub` only supported for classification dataset"
    if for_classification:
        dataset, y, end_of_ds = \
            get_batch_classification(dataset=dataset,
                                     n_examples=n_examples,
                                     n_examples_so_far=n_examples_so_far,
                                     n_classes=n_classes,
                                     shuffle_batch=shuffle_batch,
                                     shuffle_batch_seed=shuffle_batch_seed,
                                     classes_sub=classes_sub,
                                     unknown_classes=unknown_classes,
                                     with_balanced_classes=with_balanced_classes)
        n_examples = dataset.shape[0]
        n_examples_so_far = 0
        shuffle_batch = False
        dataset = dataset.numpy()
    if for_regression:
        dataset, y, end_of_ds = \
            get_batch_regression(dataset=dataset,
                                 n_examples=n_examples,
                                 shuffle_batch=shuffle_batch,
                                 shuffle_batch_seed=shuffle_batch_seed)

        n_examples = dataset.shape[0]
        n_examples_so_far = 0
        shuffle_batch = False

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

    mask = None
    if corrupt_inputs is not None:
        if not for_classification:
            y = dataset.copy() if isinstance(dataset, np.ndarray) else dataset.numpy().copy()
        if corrupt_inputs.startswith('mask'):
            splitstr = corrupt_inputs.split('_')
            masklen = int(splitstr[1]) if len(splitstr) > 1 else 8
            border = int(splitstr[2]) if len(splitstr) == 3 else 4
            dataset, mask = corrupt_image(dataset,
                                    noise_dist='mask',
                                    noise_seed=shuffle_batch_seed,
                                    noise_state=n_examples_so_far,
                                    border_without_mask=border,
                                    mask_size=masklen,
                                    frac_noise=None)
        elif corrupt_inputs.startswith('uniform'):
            splitstr = corrupt_inputs.split('_')
            frac_noise = float(splitstr[1]) if len(splitstr) == 2 else 0.1
            dataset, mask = corrupt_image(dataset,
                                    noise_seed=shuffle_batch_seed,
                                    noise_state=n_examples_so_far,
                                    frac_noise=frac_noise)
        else:
            raise ValueError(f'`corrupt_inputs` value "{corrupt_inputs}" not supported')
    return tf.convert_to_tensor(dataset), y, y_seg, end_of_ds, mask


class ReinitDataIterator(object):
    def __init__(self, arr):
        self.arr = arr
        self.id_curr = 0

    def __next__(self):
        if self.id_curr < self.n_elem:
            to_return = self.arr[self.id_curr]
        else:
            raise StopIteration
        self.id_curr += 1
        return to_return

    def reinit(self):
        self.id_curr = 0

    @property
    def n_elem(self):
        return len(self.arr)

    @property
    def n_remaining(self):
        return len(self.arr) - self.id_curr

# Map experiment type to batch getter fn
get_batch_fns = {'denoising': get_batch_unsupervised,
                 'segmentation': get_batch_segmentation,
                 'classification': get_batch_classification}
