# coding=utf-8
import tensorflow as tf

from core.factors.base_nonlin import NonLinearFactor


def softplus(x):
    return tf.math.log(1. + tf.exp(x))


class NonZeroWeightNormFactor(NonLinearFactor):
    def __init__(self,
                 sigma,
                 var_edges,
                 init_lin_point,
                 lengthscale=1.,
                 relin_freq=1,
                 N_rob=None,
                 rob_type='tukey',
                 norm_ord=1):

        super(NonZeroWeightNormFactor, self).__init__(sigma,
                                                      init_lin_point,
                                                      relin_freq,
                                                      N_rob,
                                                      rob_type)
        self.var_edges = var_edges
        self.lengthscale = lengthscale
        self.norm_ord = norm_ord
        self._set_eta_Lambda()
        self.fix_attr_fields()

    def _set_eta_Lambda(self):
        """
        Non-linear factor, so will need current
        values of filters to set variables
        """
        self._eta = None
        self._Lambda = None

    def norm(self, w):
        return tf.norm(w, axis=-1, ord=self.norm_ord)

    def energy(self, conn_vars, robust=None, aggregate=True):
        weight_norm = self.norm(conn_vars[0])
        h = weight_norm
        E = h ** 2 / self.sigma ** 2
        if robust is None:
            robust = self.N_rob is not None
        if robust:
            E = self._robust_correct_energy(E)
        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E

    def get_eta_Lambda(self, conn_vars, **kwargs):
        """
        h(w)= \sum_i |w_i|
        dh/dw = w_i / |w_i| = sign(w_i)
        """
        weights0 = self.var0[0]
        weight_norm = self.norm(weights0)
        if self.norm_ord == 2:
            dwnorm_dw = 2. * weights0 / weight_norm[..., None]
        elif self.norm_ord == 1:
            dwnorm_dw = tf.sign(weights0)
        J0 = dwnorm_dw

        Lambda = J0[..., :, None] * tf.transpose(J0[..., None], (0, 1, 2, 4, 3)) / self.sigma ** 2.

        J0Tw0 = tf.reduce_sum(J0 * weights0, axis=-1)

        h0 = weight_norm
        eta = J0 * (J0Tw0 - h0)[..., None] / self.sigma ** 2.

        E = self.energy(self.var0, robust=False, aggregate=False)
        if self.N_rob is not None:
            k = self.get_k_robust(E, N_rob=self.N_rob)
        else:
            k = 1.

        return eta * k, Lambda * k

    def _get_incoming_messages(self):
        msgs_in_eta_Lambda = []
        for etype in ('eta', 'Lambda'):
            msgs_in_eta_Lambda.append(getattr(self.var_edges, f'var_to_fac_{etype}'))
        return msgs_in_eta_Lambda

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        factor_eta, factor_Lambda = self.get_eta_Lambda(self.var0)
        msg_in_eta, msg_in_Lambda = self._get_incoming_messages()
        eta_fact_plus_msg = factor_eta + msg_in_eta
        Lambda_fact_plus_msg = factor_Lambda + tf.linalg.diag(msg_in_Lambda)
        factor_to_var_eta, factor_to_var_Lambda = \
            self.marginalise(eta_fact_plus_msg,
                             Lambda_fact_plus_msg,
                             factor_eta,
                             factor_Lambda)
        self.var_edges.fac_to_var_eta = factor_to_var_eta
        self.var_edges.fac_to_var_Lambda = factor_to_var_Lambda



class NonZeroWeightNormFactorSoftplus(NonLinearFactor):
    def __init__(self,
                 sigma,
                 var_edges,
                 init_lin_point,
                 lengthscale=1.,
                 relin_freq=1,
                 N_rob=None,
                 rob_type='tukey',
                 norm_ord=1):

        super(NonZeroWeightNormFactorSoftplus, self).__init__(sigma,
                                                      init_lin_point,
                                                      relin_freq,
                                                      N_rob,
                                                      rob_type)
        self.var_edges = var_edges
        self.lengthscale = lengthscale
        self.norm_ord = norm_ord
        self._set_eta_Lambda()
        self.fix_attr_fields()

    def _set_eta_Lambda(self):
        """
        Non-linear factor, so will need current
        values of filters to set variables
        """
        self._eta = None
        self._Lambda = None

    def norm(self, w):
        return tf.norm(w, axis=-1, ord=self.norm_ord)

    def energy(self, conn_vars, robust=None, aggregate=True):
        weight_norm = self.norm(conn_vars[0])
        h = softplus(weight_norm ** self.norm_ord / self.lengthscale ** self.norm_ord)
        z = tf.math.log([1. + 1. / self.lengthscale ** self.norm_ord])
        E = ((h - z) / self.sigma) ** 2.
        if robust is None:
            robust = self.N_rob is not None
        if robust:
            E = self._robust_correct_energy(E)
        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E

    def get_eta_Lambda(self, conn_vars, **kwargs):
        """
        h(w)=log(1 + exp( - |w|^2 / l^2))
        u := 1 + exp( - |w|^2 / l^2)
        dh / du = 1 / (1 + exp( - |w|^2 / l^2))
        du / d|w|^2 = - exp( - |w|^2 / l^2) / l^2
        d |w| ^ 2 / dw_i = 2w_i
        """
        weights0 = self.var0[0]
        weight_norm = self.norm(weights0)
        dh_du = 1. / (1. + tf.exp( weight_norm ** self.norm_ord / self.lengthscale ** self.norm_ord))
        du_dwnorm = tf.exp( weight_norm ** self.norm_ord / self.lengthscale ** self.norm_ord) / self.lengthscale ** self.norm_ord
        if self.norm_ord == 2:
            dwnorm_dw = 2. * weights0
        elif self.norm_ord == 1:
            dwnorm_dw = tf.sign(weights0)
        J0 = dh_du[..., None] * du_dwnorm[..., None] * dwnorm_dw

        Lambda = J0[..., :, None] * tf.transpose(J0[..., None], (0, 1, 2, 4, 3)) / self.sigma ** 2.

        J0Tw0 = tf.reduce_sum(J0 * weights0, axis=-1)
        weight_norm = self.norm(weights0)
        h0 = softplus(weight_norm ** self.norm_ord / self.lengthscale ** self.norm_ord)
        z = tf.math.log([1. + 1. / self.lengthscale ** self.norm_ord])
        eta = J0 * (J0Tw0 + z - h0)[..., None] / self.sigma ** 2.

        E = self.energy(self.var0, robust=False, aggregate=False)
        if self.N_rob is not None:
            k = self.get_k_robust(E, N_rob=self.N_rob)
        else:
            k = 1.

        return eta * k, Lambda * k

    def _get_incoming_messages(self):
        n_vars = self.var_edges.var_to_fac_eta.shape[0]
        msgs_in_eta_Lambda = []
        for etype in ('eta', 'Lambda'):
            msgs_in_eta_Lambda.append(getattr(self.var_edges, f'var_to_fac_{etype}'))
        return msgs_in_eta_Lambda

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        factor_eta, factor_Lambda = self.get_eta_Lambda(self.var0)
        msg_in_eta, msg_in_Lambda = self._get_incoming_messages()
        eta_fact_plus_msg = factor_eta + msg_in_eta
        Lambda_fact_plus_msg = factor_Lambda + tf.linalg.diag(msg_in_Lambda)
        factor_to_var_eta, factor_to_var_Lambda = \
            self.marginalise(eta_fact_plus_msg,
                             Lambda_fact_plus_msg,
                             factor_eta,
                             factor_Lambda)
        self.var_edges.fac_to_var_eta = factor_to_var_eta
        self.var_edges.fac_to_var_Lambda = factor_to_var_Lambda


class NonZeroWeightNormFactorMaxNorm(NonLinearFactor):
    def __init__(self,
                 sigma,
                 var_edges,
                 init_lin_point,
                 lengthscale=1.,
                 relin_freq=1,
                 N_rob=None,
                 rob_type='tukey',
                 norm_ord=2):

        super(NonZeroWeightNormFactorMaxNorm, self).__init__(sigma,
                                                      init_lin_point,
                                                      relin_freq,
                                                      N_rob,
                                                      rob_type)
        self.var_edges = var_edges
        self.lengthscale = lengthscale
        self.norm_ord = norm_ord
        self._set_eta_Lambda()
        self.fix_attr_fields()

    def _set_eta_Lambda(self):
        """
        Non-linear factor, so will need current
        values of filters to set variables
        """
        self._eta = None
        self._Lambda = None

    def norm(self, w):
        return tf.norm(w, axis=-1, ord=self.norm_ord)

    def energy(self, conn_vars, robust=None, aggregate=True):
        weight_norm = self.norm(conn_vars[0])
        h = softplus(-weight_norm ** self.norm_ord / self.lengthscale ** self.norm_ord)
        E = ((h) / self.sigma) ** 2.
        # h = tf.sqrt(weight_norm)
        # E = h ** 2 / self.sigma ** 2
        if robust is None:
            robust = self.N_rob is not None
        if robust:
            E = self._robust_correct_energy(E)
        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E

    def get_eta_Lambda(self, conn_vars, **kwargs):
        """
        h(w)=log(1 + exp( - |w|^2 / l^2))
        u := 1 + exp( - |w|^2 / l^2)
        dh / du = 1 / (1 + exp( - |w|^2 / l^2))
        du / d|w|^2 = - exp( - |w|^2 / l^2) / l^2
        d |w| ^ 2 / dw_i = 2w_i
        """
        weights0 = self.var0[0]
        weight_norm = self.norm(weights0)
        dh_du = 1. / (1. + tf.exp(- weight_norm ** self.norm_ord / self.lengthscale ** self.norm_ord))
        du_dwnorm = -tf.exp(- weight_norm ** self.norm_ord / self.lengthscale ** self.norm_ord) / self.lengthscale ** self.norm_ord
        if self.norm_ord == 2:
            dwnorm_dw = 2. * weights0
        elif self.norm_ord == 1:
            dwnorm_dw = tf.sign(weights0) #* 0.5 / tf.sqrt(weight_norm)[..., None]
        J0 = dh_du[..., None] * du_dwnorm[..., None] * dwnorm_dw

        Lambda = J0[..., :, None] * tf.transpose(J0[..., None], (0, 1, 2, 4, 3)) / self.sigma ** 2.

        J0Tw0 = tf.reduce_sum(J0 * weights0, axis=-1)
        weight_norm = self.norm(weights0)
        # h0 = tf.sqrt(weight_norm)
        h0 = softplus(-weight_norm ** self.norm_ord / self.lengthscale ** self.norm_ord)
        eta = J0 * (J0Tw0 - h0)[..., None] / self.sigma ** 2.

        E = self.energy(self.var0, robust=False, aggregate=False)
        if self.N_rob is not None:
            k = self.get_k_robust(E, N_rob=self.N_rob)
        else:
            k = 1.

        return eta * k, Lambda * k

    def _get_incoming_messages(self):
        n_vars = self.var_edges.var_to_fac_eta.shape[0]
        msgs_in_eta_Lambda = []
        for etype in ('eta', 'Lambda'):
            msgs_in_eta_Lambda.append(getattr(self.var_edges, f'var_to_fac_{etype}'))
        return msgs_in_eta_Lambda

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        factor_eta, factor_Lambda = self.get_eta_Lambda(self.var0)
        msg_in_eta, msg_in_Lambda = self._get_incoming_messages()
        eta_fact_plus_msg = factor_eta + msg_in_eta
        Lambda_fact_plus_msg = factor_Lambda + tf.linalg.diag(msg_in_Lambda)
        factor_to_var_eta, factor_to_var_Lambda = \
            self.marginalise(eta_fact_plus_msg,
                             Lambda_fact_plus_msg,
                             factor_eta,
                             factor_Lambda)
        self.var_edges.fac_to_var_eta = factor_to_var_eta
        self.var_edges.fac_to_var_Lambda = factor_to_var_Lambda
