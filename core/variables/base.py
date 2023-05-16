# coding=utf-8
import tensorflow as tf
from typing import Union

from core.utils.utils import FixedAttributeFieldsObject


class Variable(FixedAttributeFieldsObject):
    # TODO: include edges attribute to variable
    def __init__(self, mu_init: tf.Tensor, sigma_init: Union[float, tf.Tensor]):
        sigma_init = tf.ones_like(mu_init) * float(sigma_init) if isinstance(sigma_init, (float, int)) else sigma_init
        self._eta = mu_init / sigma_init ** 2.
        self._Lambda = 1. / sigma_init ** 2.

    def _check_shapes(self, new):
        if list(new.shape) != list(self.shape):
            raise ValueError(f'New shape ({list(new.shape)}) '
                             f'does not match existing shape '
                             f'({list(self.shape)}). ')

    @property
    def mu(self):
        return self._eta / self._Lambda
        # return tf.where(tf.logical_and(self._eta == 0., self._Lambda == 0.),
        #                 0.,
        #                 self._eta / self._Lambda)

    @property
    def sigma(self):
        return 1. / tf.math.sqrt(self._Lambda)

    @property
    def eta(self):
        return self._eta

    @property
    def Lambda(self):
        return self._Lambda

    @Lambda.setter
    def Lambda(self, new_Lambda):
        self._check_shapes(new_Lambda)
        self._Lambda = new_Lambda

    @eta.setter
    def eta(self, new_eta):
        self._check_shapes(new_eta)
        self._eta = new_eta

    @property
    def shape(self):
        return self._eta.shape
