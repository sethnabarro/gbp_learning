
# coding=utf-8
import tensorflow as tf

LEAKY_RELU_ALPHA = 0.2
SOFT_LEAKY_RELU_SCALE = 25.


def get_nonlin_fns():
    nonlin_fns = {'none': lambda x: x,
                  'relu': tf.nn.relu,
                  'leaky_relu': lambda x: tf.nn.leaky_relu(x, alpha=LEAKY_RELU_ALPHA),
                  'soft_leaky_relu': lambda x:
                        LEAKY_RELU_ALPHA * x + (1. - LEAKY_RELU_ALPHA) * tf.math.log(1. + tf.exp(x * SOFT_LEAKY_RELU_SCALE)) / SOFT_LEAKY_RELU_SCALE,   # https://stats.stackexchange.com/a/329803
                  'elu': tf.nn.elu,
                  'tanh': tf.tanh,
                  'sigmoid': tf.sigmoid,
                  'softplus': tf.nn.softplus}
    return nonlin_fns


def get_nonlin_fn_grads():
    nonlin_grads = {'none': lambda x: tf.ones_like(x),
                    'relu': lambda x: tf.where(x > 0., 1., 0.),
                    'leaky_relu': lambda x: tf.where(x > 0., 1., LEAKY_RELU_ALPHA),
                    'soft_leaky_relu': lambda x: LEAKY_RELU_ALPHA + (1. - LEAKY_RELU_ALPHA) * tf.sigmoid(SOFT_LEAKY_RELU_SCALE * x),
                    'elu': lambda x: tf.where(x < 0., tf.exp(x), 1.),
                    'tanh': lambda x: (1. - tf.tanh(x) ** 2.),
                    'sigmoid': lambda x: tf.sigmoid(x) * (1. - tf.sigmoid(x)),
                    'softplus': tf.nn.sigmoid}
    return nonlin_grads
