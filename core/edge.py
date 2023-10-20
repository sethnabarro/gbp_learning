# coding=utf-8
import numpy as np
import tensorflow as tf
from typing import Union

from core.utils.utils import FixedAttributeFieldsObject


class Edge(FixedAttributeFieldsObject):
    def __init__(self,
                 var_to_fac_eta: Union[None, tf.Tensor] = None,
                 var_to_fac_Lambda: Union[None, tf.Tensor] = None,
                 fac_to_var_eta: Union[None, tf.Tensor] = None,
                 fac_to_var_Lambda: Union[None, tf.Tensor] = None,
                 momentum: float = 0.,
                 dropout: float = 0.,
                 track_msg_diffs: bool = False,
                 record_all_diffs: bool = False,
                 n_iters: [None, int] = None,
                 name: [None, str] = None):
        if record_all_diffs:
            assert track_msg_diffs, "`record_all_diffs=True` but `track_msg_diff=False`."

        super(Edge, self).__init__()
        self.momentum = momentum
        self.dropout = dropout
        self._var_to_fac_eta = var_to_fac_eta
        self._var_to_fac_Lambda = var_to_fac_Lambda
        self._fac_to_var_eta = fac_to_var_eta
        self._fac_to_var_Lambda = fac_to_var_Lambda
        self.name = name or f'edge_{np.random.randint(0, np.iinfo(np.int32).max)}'
        # self._first_iter = {'var_to_fac_eta': True,
        #                     'var_to_fac_Lambda': True,
        #                     'fac_to_var_eta': True,
        #                     'fac_to_var_Lambda': True}
        self.track_msg_diffs = track_msg_diffs
        self.record_all_diffs = record_all_diffs
        self._msg_diffs = None
        self.n_iters = n_iters
        if track_msg_diffs:
            self.init_msg_diff_recorder()

        self.iter = {'var_to_fac_eta': tf.zeros([1], dtype=tf.int32),
                     'var_to_fac_Lambda': tf.zeros([1], dtype=tf.int32),
                     'fac_to_var_eta': tf.zeros([1], dtype=tf.int32),
                     'fac_to_var_Lambda': tf.zeros([1], dtype=tf.int32)}
        self.fix_attr_fields()

    def init_msg_diff_recorder(self):
        if self.record_all_diffs:
            assert self.n_iters is not None, "If recording all diffs need to specify `n_iters` up front"
        self._msg_diffs = dict()
        for mtype in self._order:
            if self.record_all_diffs:
                self._msg_diffs[mtype] = tf.zeros(shape=(self.n_iters,), dtype=tf.float32)
            else:
                self._msg_diffs[mtype] = tf.constant(np.finfo(np.float32).max)

    def __str__(self):
        to_print = self.name
        msg = None
        o = 0
        while msg is None:
            msg = getattr(self, f'_{self._order[o]}')
            o += 1
        if msg is not None:
            to_print += f'_{msg.shape.as_list()}'.replace('[', '').replace(']', '').replace(', ', '_')
        return to_print

    @property
    def msg_diffs(self):
        if self.record_all_diffs:
            # Clip elements of msg diff tracking tensor which are not yet populated
            return {k: v[:self.iter[k][0]] for k, v in self._msg_diffs.items()}
        else:
            return self._msg_diffs

    @property
    def var_to_fac_eta(self):
        return self._var_to_fac_eta

    @property
    def var_to_fac_Lambda(self):
        return self._var_to_fac_Lambda

    @property
    def fac_to_var_eta(self):
        return self._fac_to_var_eta

    @property
    def fac_to_var_Lambda(self):
        return self._fac_to_var_Lambda

    def _apply_momtm(self, new, old, first_iter):
        if self.momentum == 0.:
            return new
        else:
            new_with_mom = old * self.momentum + new * (1. - self.momentum)
            return tf.where(first_iter, new, new_with_mom)

    def _apply_dropout(self, new_msg, old_msg, first_iter, seed=None):
        if self.dropout == 0.:
            return new_msg
        else:
            if seed is None:
                runif = tf.random.uniform(old_msg.shape, minval=0., maxval=1.)
            else:
                runif = tf.random.stateless_uniform(old_msg.shape, minval=0., maxval=1., seed=[seed, seed])
            drop_mask = tf.cast(runif < self.dropout, old_msg.dtype)
            new_with_dropout = new_msg * (1. - drop_mask) + old_msg * drop_mask
            return tf.where(first_iter, new_msg, new_with_dropout)

    def _check_msg_shapes(self, new, old):
        if old is not None:
            if list(old.shape) != list(new.shape):
                raise ValueError(f'New edge shape ({list(new.shape)}) '
                                 f'does not match existing edge shape '
                                 f'({list(old.shape)}). ')

    def compute_msg_diff(self, old, new):
        return tf.reduce_max(tf.abs(old - new))

    def _update_msg_diff(self, old, new, messtype):
        diff = self.compute_msg_diff(old, new)
        if self.record_all_diffs:
            self._msg_diffs[messtype] = tf.tensor_scatter_nd_add(self._msg_diffs[messtype],
                                                                 [self.iter[messtype]],
                                                                 [diff])
        else:
            self._msg_diffs[messtype] = diff

    @var_to_fac_eta.setter
    def var_to_fac_eta(self, eta_new):
        old = self._var_to_fac_eta
        self._check_msg_shapes(eta_new, self._var_to_fac_eta)
        self._var_to_fac_eta = eta_new

        if self.track_msg_diffs:
            self._update_msg_diff(old, eta_new, 'var_to_fac_eta')

        self.iter['var_to_fac_eta'] = self.iter['var_to_fac_eta'] + 1

    @var_to_fac_Lambda.setter
    def var_to_fac_Lambda(self, Lambda_new):
        old = self._var_to_fac_Lambda
        self._check_msg_shapes(Lambda_new, self._var_to_fac_Lambda)
        self._var_to_fac_Lambda = Lambda_new

        if self.track_msg_diffs:
            self._update_msg_diff(old, Lambda_new, 'var_to_fac_Lambda')

        self.iter['var_to_fac_Lambda'] = self.iter['var_to_fac_Lambda'] + 1

    @fac_to_var_eta.setter
    def fac_to_var_eta(self, eta_new):
        old = self._fac_to_var_eta
        self._check_msg_shapes(eta_new, self._fac_to_var_eta)
        eta_mom_new = self._apply_momtm(eta_new, self._fac_to_var_eta, self.iter['fac_to_var_eta'] == 0)
        self._fac_to_var_eta = self._apply_dropout(eta_mom_new, self._fac_to_var_eta, self.iter['fac_to_var_eta'] == 0, self.iter['fac_to_var_eta'][0])

        if self.track_msg_diffs:
            self._update_msg_diff(old, eta_new, 'fac_to_var_eta')

        self.iter['fac_to_var_eta'] = self.iter['fac_to_var_eta'] + 1

    @fac_to_var_Lambda.setter
    def fac_to_var_Lambda(self, Lambda_new):
        old = self._fac_to_var_Lambda
        self._check_msg_shapes(Lambda_new, self._fac_to_var_Lambda)
        Lambda_mom_new = self._apply_momtm(Lambda_new, self._fac_to_var_Lambda, self.iter['fac_to_var_Lambda'] == 0)
        self._fac_to_var_Lambda = self._apply_dropout(Lambda_mom_new, self._fac_to_var_Lambda, self.iter['fac_to_var_Lambda'] == 0, self.iter['fac_to_var_Lambda'][0])

        if self.track_msg_diffs:
            self._update_msg_diff(old, Lambda_new, 'fac_to_var_Lambda')

        self.iter['fac_to_var_Lambda'] = self.iter['fac_to_var_Lambda'] + 1

    @property
    def _order(self):
        return ['var_to_fac_eta', 'var_to_fac_Lambda', 'fac_to_var_eta', 'fac_to_var_Lambda']

    @property
    def state(self):
        edge_state = ([self.iter[o] for o in self._order],)
        edge_state += tuple(getattr(self, f'_{o}') for o in self._order)
        if self.track_msg_diffs:
            edge_state += ([self._msg_diffs[o] for o in self._order],)
        return edge_state

    @state.setter
    def state(self, new_state):
        for o, ns in zip(self._order, new_state[0]):
            self.iter[o] = ns
        for i, o in enumerate(self._order):
            setattr(self, f'_{o}', new_state[1 + i])

        if self.track_msg_diffs:
            for o, ns in zip(self._order, new_state[5]):
                self._msg_diffs[o] = ns

    @property
    def named_state(self):
        edge_state = (('iters', [self.iter[o] for o in self._order]),)
        edge_state += tuple((o, (getattr(self, f'_{o}',))) for o in self._order)
        if self.track_msg_diffs:
            edge_state += ([(f'{o}_diff', self._msg_diffs[o]) for o in self._order],)
        return edge_state

    @property
    def shape(self):
        return self._var_to_fac_eta.shape
