# coding=utf-8
import tensorflow as tf
from tensorflow.keras.regularizers import L2 as L2Reg
import tensorflow_datasets as tfds


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def get_data(validation=True, tr_size=None, shuffle_seed=3338, fashion=False):
    (ds_train, ds_test), ds_info = tfds.load(
        'fashion_mnist' if fashion else 'mnist',
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


def get_fresh_model(arch_name, activation, l2_coeff, pool_type, seed=897):
    tf.random.set_seed(seed=seed)
    if arch_name.startswith('conv_pool_dense_k'):
        ksize = int(arch_name.split('conv_pool_dense_k')[-1])
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=ksize, activation=activation, kernel_regularizer=L2Reg(l2=l2_coeff)),
            {'max': tf.keras.layers.MaxPool2D(2), 'avg': tf.keras.layers.AvgPool2D(2)}[pool_type],
            # tf.keras.layers.Conv2D(8, kernel_size=ksize, activation=activation, kernel_regularizer=L2Reg(l2=l2_coeff)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, kernel_regularizer=L2Reg(l2=l2_coeff))
        ])
    else:
        raise ValueError(f'Architecture {arch_name} not supported.')

    model.build(input_shape=(None, 28, 28, 1))
    return model


def get_optim(name, lr):
    if name == 'adam':
        opt = tf.keras.optimizers.Adam(lr)
    elif name == 'sgd':
        opt = tf.keras.optimizers.SGD(lr)
    return opt


class FIFOBuffer(object):
    def __init__(self, n_examples, x_dims, y_dims):
        self._x_arr = tf.zeros([n_examples] + list(x_dims))
        self._y_arr = tf.zeros([n_examples] + list(y_dims), dtype=tf.int64)
        self.counter = tf.constant(0, dtype=tf.int32)
        self.size = n_examples
        self.idx_updated = []

    def update_recent(self, x_new, y_new):
        assert x_new.shape[0] == y_new.shape[0], "Num. new inputs != num. new labels"
        n_new = x_new.shape[0]
        if n_new > self.size:
            self._x_arr = tf.convert_to_tensor(x_new[:self.size])
            self._y_arr = tf.convert_to_tensor(y_new[:self.size])
            self.counter += self.size
        else:
            rolls_over = (self.counter + n_new) // self.size > self.counter // self.size

            if rolls_over:
                def _update(to_update, update_with):
                    to_update = tf.concat([to_update[:self.counter % self.size],
                                           update_with[:self.size - self.counter % self.size]],
                                          axis=0)
                    return tf.concat([update_with[(self.size - self.counter % self.size):],
                                      to_update[(self.counter + n_new) % self.size:]],
                                     axis=0)
            else:
                def _update(to_update, update_with):
                    return tf.concat([to_update[:self.counter % self.size],
                                      update_with,
                                      to_update[(self.counter + n_new) % self.size:]],
                                     axis=0)
            self._x_arr = _update(self._x_arr, x_new)
            self._y_arr = _update(self._y_arr, y_new)
            self.counter += n_new
        assert self._x_arr.shape[0] == self.size
        assert self._y_arr.shape[0] == self.size

    @property
    def arrays(self):
        if len(self.idx_updated):
            if len(self.idx_updated) < self._x_arr.shape[0]:
                return tf.gather(self._x_arr, self.idx_updated), tf.gather(self._y_arr, self.idx_updated)
        return self._x_arr[:self.counter], self._y_arr[:self.counter]

    def as_batched_dataset(self, batchsize):
        assert (self.counter >= batchsize) and (self.size >= batchsize)
        x, y = self.arrays
        shuffled_idx = tf.random.shuffle(tf.range(tf.minimum(self.counter, self.size)))
        ds = []
        b = 0
        while b * batchsize < tf.minimum(self.counter, self.size):
            idx_b = shuffled_idx[b * batchsize:(b + 1) * batchsize]
            xb, yb = tf.gather(x, idx_b), tf.gather(y, idx_b)
            ds.append((xb, yb))
            b += 1
        return ds
