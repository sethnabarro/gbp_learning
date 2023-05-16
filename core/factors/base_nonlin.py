# coding=utf-8
import tensorflow as tf
from core.factors.base import Factor


class NonLinearFactor(Factor):
    def __init__(self,
                 sigma,
                 init_lin_point,
                 relin_freq=1,
                 N_rob=None,
                 rob_type='tukey',
                 kmult=1.):
        Factor.__init__(self,
                        sigma,
                        N_rob=N_rob,
                        rob_type=rob_type,
                        kmult=kmult)
        self.var0 = init_lin_point
        self.relin_freq = relin_freq

    def relinearise(self, itr, current_vars):
        is_relin_iter = tf.math.mod(itr, self.relin_freq) == 0
        maybe_relin_point = [tf.where(is_relin_iter, cv, v0)
                             for cv, v0 in zip(current_vars, self.var0)]
        self.var0 = maybe_relin_point

    @property
    def state(self):
        state = [self.var0, self.get_edge_messages()]
        return state

    @state.setter
    def state(self, new_state):
        self.var0 = new_state[0]
        self.set_edge_messages(new_state[1])

    @property
    def named_state(self):
        state = (('var0', self.var0), ('edges', self.get_edge_messages(named=True)))
        return state
