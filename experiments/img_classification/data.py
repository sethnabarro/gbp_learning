# coding=utf-8
import numpy as np
import os
import tensorflow_datasets as tfds
import tensorflow as tf

from core.utils.data import ReinitDataIterator


def save_data_to_local(local_dir=None):
    # Where to save data to
    local_dir = local_dir or '../inputs/mnist/'

    # Load MNIST
    mnist_train, mnist_test = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )[0]

    def _convert_tfds_to_array_dict(ds):
        ds_iter = ds.as_numpy_iterator()
        ds_by_class = {x: np.array([]).reshape((0, 28, 28, 1)) for x in range(10)}
        for x, y in ds_iter:
            ds_by_class[y] = np.concatenate([x[None], ds_by_class[y]], axis=0)
        return ds_by_class

    train = _convert_tfds_to_array_dict(mnist_train)
    test = _convert_tfds_to_array_dict(mnist_test)
    for cl in range(10):
        np.save(os.path.join(local_dir, f'train{cl}.npy'), train[cl])
        np.save(os.path.join(local_dir, f'test{cl}.npy'), test[cl])


def get_ds_mean_std(ds_by_class):
    all_ds = []
    for cl in ds_by_class:
        all_ds.append(ds_by_class[cl])
    all_ds = np.concatenate(all_ds, axis=0)
    mean_ds = tf.reduce_mean(all_ds, axis=[0, 1, 2]).numpy()
    std_ds = tf.math.reduce_std(all_ds, axis=[0, 1, 2]).numpy()
    return mean_ds, std_ds


def load_data(datadir=None,
              shuffle=True,
              shuffle_seed=None,
              rescale=True,
              test_on_train_set=False,
              n_classes=None,
              examples_per_class_train=None,
              examples_per_class_test=None,
              validation=False,
              n_validation=None,
              data_subsample_seed=None,
              dataset=None):
    assert not ((shuffle_seed is not None) and not shuffle), \
        "`shuffle_seed` specified, but `shuffle=False`"
    dataset = dataset or 'mnist'
    dataset_to_dirname = {'fashion_mnist': 'fmnist',
                          'mnist': 'mnist',
                          'cifar10': 'cifar10'}
    relpath = datadir or f'../../inputs/{dataset_to_dirname[dataset]}/'
    repodir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(repodir, relpath)
    train_by_class = {}
    test_by_class = {}
    n_classes = n_classes or 10
    n_val_per_class = None
    if validation:
        print('Splitting trainset into train and validation')
        assert n_validation % n_classes == 0
        n_val_per_class = n_validation // n_classes
    for cl in range(n_classes):
        train_fname_cl = f'mnist_train{cl}.npy' if dataset == 'mnist' else f'train{cl}.npy'
        train_by_class[cl] = np.load(os.path.join(datadir, train_fname_cl)).astype(np.float32)
        if validation:
            # Split train into train and validation (allocated to test_by_class)
            train_by_class[cl], test_by_class[cl] = \
                train_by_class[cl][:-n_val_per_class], train_by_class[cl][-n_val_per_class:]
            print(f'Class {cl}, '
                  f'Num. train examples {train_by_class[cl].shape[0]}, '
                  f'Num. val examples {test_by_class[cl].shape[0]}')
        else:
            test_fname_cl = f'mnist_test{cl}.npy' if dataset == 'mnist' else f'test{cl}.npy'
            test_by_class[cl] = np.load(os.path.join(datadir, test_fname_cl)).astype(np.float32)

    if (isinstance(rescale, bool) and rescale) or rescale == 'zero_one':
        for cl in train_by_class:
            train_by_class[cl] /= 255.
            test_by_class[cl] /= 255.

    elif rescale == 'mean_std':
        mean_tr, std_tr = get_ds_mean_std(train_by_class)
        for cl in train_by_class:
            train_by_class[cl] = (train_by_class[cl] - mean_tr) / std_tr
            test_by_class[cl] = (test_by_class[cl] - mean_tr) / std_tr

    if examples_per_class_train or examples_per_class_test:
        # Shuffle for data subsampling
        data_subsample_seed = data_subsample_seed or 0
        np.random.seed(data_subsample_seed)
        for cl in range(n_classes):
            np.random.shuffle(train_by_class[cl])
            np.random.shuffle(test_by_class[cl])

    if test_on_train_set:
        test_by_class = train_by_class

    if examples_per_class_train is not None:
        examples_per_class_train = int(examples_per_class_train)
        train_by_class = {cl: tr[:examples_per_class_train] for cl, tr in train_by_class.items()}
    if examples_per_class_test is not None:
        examples_per_class_test = int(examples_per_class_test)
        test_by_class = {cl: te[:examples_per_class_test] for cl, te in test_by_class.items()}

    if shuffle:
        # Shuffle for batch ordering
        if shuffle_seed is not None:
            print('Data shuffle seed', shuffle_seed)
            np.random.seed(shuffle_seed)
        for cl in range(n_classes):
            np.random.shuffle(train_by_class[cl])
            np.random.shuffle(test_by_class[cl])

    # Make into generators
    train_by_class['examples_per_class'] = {}
    test_by_class['examples_per_class'] = {}
    for cl in range(n_classes):
        train_by_class['examples_per_class'][cl] = train_by_class[cl].shape[0]
        test_by_class['examples_per_class'][cl] = test_by_class[cl].shape[0]
        train_by_class[cl] = ReinitDataIterator(train_by_class[cl])
        test_by_class[cl] = ReinitDataIterator(test_by_class[cl])   # Test data may be revisited multiple times

    return train_by_class, test_by_class
