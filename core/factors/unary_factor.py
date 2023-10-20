# coding=utf-8
import tensorflow as tf

from core.factors.base import Factor
from core.factors.base_nonlin import NonLinearFactor


class UnaryFactor(Factor):
    def __init__(self,
                 sigma,
                 obs,
                 var_edges,
                 N_rob=None,
                 rob_type='tukey',
                 kmult=1.):
        super(UnaryFactor, self).__init__(sigma, N_rob, rob_type,
                                          kmult=kmult)
        self._obs = obs
        self.var_edges = var_edges
        self._set_eta_Lambda()
        self.fix_attr_fields()

    def energy(self, conn_vars, robust=None, aggregate=True):
        E = (conn_vars[0] - self.obs) ** 2 / self.sigma ** 2
        if robust:
            E = self._robust_correct_energy(E)
        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E

    def _set_eta_Lambda(self):
        self._eta = self.obs * tf.ones(self.var_edges.shape) / self.sigma ** 2.
        self._Lambda = tf.ones(self.var_edges.shape) / self.sigma ** 2.

    def get_eta_Lambda(self, conn_vars=None, **kwargs):
        if self.N_rob is not None:
            E = self.energy(conn_vars, robust=False, aggregate=False)
            k = self.get_k_robust(E)
            # print(tf.reduce_mean(k), 'K', tf.reduce_mean(self._Lambda))
            # print(tf.reduce_mean(self._Lambda * k), 'Lambda k')
            k = tf.where(tf.math.is_nan(k), 1., k)  # where E == 0.
            return self._eta * k, self._Lambda * k
        else:
            return self._eta, self._Lambda

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        """Unary, so don't need to deal with incoming messages"""
        eta, Lambda = self.get_eta_Lambda(conn_vars)
        self.var_edges._fac_to_var_eta = eta
        self.var_edges._fac_to_var_Lambda = Lambda

    def reset_eta_Lambda(self, eta_new, Lambda_new):
        self._eta = eta_new
        self._Lambda = Lambda_new
        self._obs = self._eta / self._Lambda
        self.update_outgoing_messages([None])

    def get_edge_messages(self, named=False):
        attr_to_get = 'named_state' if named else 'state'
        if named:
            return [(str(self.var_edges), getattr(self.var_edges, attr_to_get))]
        else:
            return [getattr(self.var_edges, attr_to_get)]

    def set_edge_messages(self, edge_msgs):
        self.var_edges.state = edge_msgs[0]

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, new_obs):
        self._obs = new_obs

    @property
    def state(self):
        st = [self.get_edge_messages()] + [self._obs, self._eta, self._Lambda]
        return st

    @state.setter
    def state(self, new_state):
        self.set_edge_messages(new_state[0])
        self._obs = new_state[1]
        self._eta = new_state[2]
        self._Lambda = new_state[3]

    @property
    def named_state(self):
        st = [('edges', self.get_edge_messages(named=True)),
              ('obs', self.obs),
              ('eta', self._eta),
              ('Lambda', self._Lambda)]
        return st


class NonLinearUnaryFactor(NonLinearFactor, UnaryFactor):
    def __init__(self,
                 sigma,
                 obs,
                 var_edges,
                 init_lin_point,
                 N_rob=None,
                 rob_type='tukey',
                 kmult=1.,
                 relin_freq=1):
        NonLinearFactor.__init__(self, sigma, init_lin_point,
                                 relin_freq=relin_freq,
                                 N_rob=N_rob,
                                 rob_type=rob_type)
        UnaryFactor.__init__(self, sigma, obs, var_edges,
                             N_rob=N_rob,
                             rob_type=rob_type,
                             kmult=kmult)

    def get_eta_Lambda(self, conn_vars=None, **kwargs):
        if self.N_rob is not None:
            E = self.energy(self.var0, robust=False, aggregate=False)
            k = self.get_k_robust(E)
            k = tf.where(tf.math.is_nan(k), 1., k)  # where E == 0.
            return self._eta * k, self._Lambda * k
        else:
            return self._eta, self._Lambda

    def get_edge_messages(self, named=False):
        return (str(self.var_edges), self.var_edges.named_state) if named else self.var_edges.state,

    def set_edge_messages(self, edge_msgs):
        self.var_edges.state, = edge_msgs

    @property
    def state(self):
        state = [self.var0, self.get_edge_messages(), self._obs, self._eta, self._Lambda]
        return state

    @state.setter
    def state(self, new_state):
        self.var0 = new_state[0]
        self.set_edge_messages(new_state[1])
        self._obs = new_state[2]
        self._eta = new_state[3]
        self._Lambda = new_state[4]

    @property
    def named_state(self):
        st = [('var0', self.var0),
              ('edges', self.get_edge_messages(named=True)),
              ('obs', self._obs),
              ('eta', self._eta),
              ('Lambda', self._Lambda)]
        return st
