# utf-8
import tensorflow as tf

from core.factors.base_nonlin import NonLinearFactor
from core.utils.utils import flatten_filters


class FilterNormalisationFactor(NonLinearFactor):
    def __init__(self, sigma, var_edges, init_lin_point,
                 relin_freq=1, target_norm=1., N_rob=None,
                 rob_type='tukey'):
        super(FilterNormalisationFactor, self).__init__(sigma,
                                                        init_lin_point,
                                                        relin_freq,
                                                        N_rob,
                                                        rob_type)
        self.target_norm = target_norm
        self.var_edges = var_edges

    def energy(self, conn_vars, robust=None, aggregate=False):
        filters = conn_vars[0]
        filters_flat = flatten_filters(filters)
        filter_norms = tf.norm(filters_flat, axis=-1)
        E = ((filter_norms - self.target_norm) / self.sigma) ** 2
        if robust is None:
            robust = self.N_rob is not None
        if robust:
            E = self._robust_correct_energy(E)
        if aggregate:
            E = tf.reduce_sum(E)
        return E

    def get_eta_Lambda(self, conn_vars, **kwargs):
        filters_flat = flatten_filters(self.var0[0])
        filter_norms = tf.norm(filters_flat, axis=-1)
        J = filters_flat / filter_norms[:, None]
        z = self.target_norm
        eta = J * (tf.reduce_sum(J * filters_flat, axis=-1) + z - filter_norms)[:, None] / self.sigma ** 2.
        Lambda = J[:, None] * J[..., None] / self.sigma ** 2.

        if self.N_rob is not None:
            E = self.energy(conn_vars, robust=False)
            k = self.get_k_robust(E, N_rob=self.N_rob)
        else:
            k = 1.
        return eta * k, Lambda * k

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        eta, Lambda = self.get_eta_Lambda(conn_vars, **kw_msgs_in)
        incoming_eta, incoming_Lambda = self._get_incoming_messages()
        multidim_var = len(incoming_Lambda.shape) == 3
        if multidim_var:
            eta_marg = eta
            Lambda_marg = Lambda
        else:
            eta_plus_mess = eta + incoming_eta
            Lambda_plus_mess = Lambda + tf.linalg.diag(incoming_Lambda)
            eta_marg, Lambda_marg = \
                self.marginalise(eta_plus_mess,
                                 Lambda_plus_mess,
                                 eta,
                                 Lambda)
        self.var_edges.fac_to_var_eta = eta_marg
        self.var_edges.fac_to_var_Lambda = Lambda_marg

    def _get_incoming_messages(self):
        return self.var_edges.var_to_fac_eta, self.var_edges.var_to_fac_Lambda



