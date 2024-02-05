# coding=utf-8
import tensorflow as tf
from tensorflow.keras.regularizers import L2 as L2Reg
import tensorflow_datasets as tfds


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def get_data(validation=True, tr_size=None, shuffle_seed=3338):
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    n_tr_total = 60000
    ds_train = ds_train.shuffle(buffer_size=n_tr_total, seed=shuffle_seed)
    val_frac = 0.15

    if validation:
        nval = int(val_frac * n_tr_total)

        if tr_size is None:
            tr_size = n_tr_total - nval
        else:
            assert nval + tr_size <= n_tr_total
        ds_tr = ds_train.enumerate().filter(lambda i, xy: i < tr_size).map(lambda j, xxy: xxy)
        ds_test = ds_train.enumerate().filter(lambda i, xy: tf.logical_and(tr_size <= i, i < tr_size + nval)).map(lambda j, xxy: xxy)
        ds_train = ds_tr
    elif tr_size is not None:
        ds_train = ds_train.enumerate().filter(lambda i, xy: i < tr_size).map(lambda j, xxy: xxy)

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.cache()
    return ds_train, ds_test, ds_info


def get_model(l2_coeff):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, kernel_regularizer=L2Reg(l2=l2_coeff))
    ])

    model.build(input_shape=(None, 28, 28, 1))
    return model


def get_optim(name, lr):
    assert name in ('adam', 'sgd')
    if name == 'adam':
        opt = tf.keras.optimizers.Adam(lr)
    elif name == 'sgd':
        opt = tf.keras.optimizers.SGD(lr)
    return opt
