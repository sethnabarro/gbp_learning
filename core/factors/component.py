# coding=utf-8
import tensorflow as tf

from core.factors.base_nonlin import NonLinearFactor
from core.utils.utils import flatten_filters


class ComponentFactor(NonLinearFactor):
    N_PARALLEL_PROC_MARG = 1
    def __init__(self,
                 sigma,
                 coeff_var_edges,
                 filter_var_edges,
                 component_var_edges,
                 N_rob=None,
                 rob_type='tukey',
                 relin_freq=1,
                 lin_point=None,
                 kmult=1.):
        super(ComponentFactor, self).__init__(sigma, lin_point, relin_freq, N_rob, rob_type, kmult=kmult)
        self.coeff_var_edges = coeff_var_edges
        self.filter_var_edges = filter_var_edges
        self.component_var_edges = component_var_edges
        self._set_eta_Lambda()
        self.fix_attr_fields()

    def _set_eta_Lambda(self):
        """
        Non-linear factor, so will need current
        values of filters to set variables
        """
        self._eta = None
        self._Lambda = None

    def _get_incoming_messages(self):
        msgs_in_eta_Lambda = []
        edges = (self.filter_var_edges,
                 self.coeff_var_edges,
                 self.component_var_edges)
        for etype in ('eta', 'Lambda'):
            msgs_in = []
            for e in edges:
                msgs_in.append(getattr(e, f'var_to_fac_{etype}'))
            msgs_in = tf.concat([m[..., None] for m in msgs_in], axis=-1)
            msgs_in_eta_Lambda.append(msgs_in)
        return msgs_in_eta_Lambda

    def energy(self, conn_vars, robust=None, aggregate=True):
        filters, weights, components = conn_vars
        filters_flat = flatten_filters(filters)
        weightsTfilters = weights[..., None, :, None] * filters_flat[None, None, None]
        diff = components - weightsTfilters
        E = diff ** 2 / self.sigma ** 2
        if robust is None:
            robust = self.N_rob is not None
        if robust and self.N_rob is not None:
            E = self._robust_correct_energy(E)
        E = E[..., None]

        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E

    def get_eta_Lambda(self, conn_vars, **kwargs):
        eta, J = self.get_eta_J(conn_vars)
        Lambda = J[..., None] * J[..., None, :]

        return eta, Lambda

    def get_eta_J(self, conn_vars, **kwargs):
        filters, weights, components = self.var0
        filters_flat = flatten_filters(filters)
        weightsTfilters = weights[..., None, :, None] * filters_flat[None, None, None]
        h0 = components - weightsTfilters

        filters_flat_bc = tf.broadcast_to(filters_flat[None, None, None], components.shape)
        weights_bc = tf.broadcast_to(weights[..., None, :, None], components.shape)
        Jfilt = -tf.broadcast_to(weights[..., None, :, None], components.shape)
        Jw = -filters_flat_bc
        Jcomp = tf.ones_like(components)
        J = tf.concat([Jfilt[..., None], Jw[..., None], Jcomp[..., None]], axis=-1)

        varstack0 = tf.concat([filters_flat_bc[..., None],
                               weights_bc[..., None],
                               components[..., None]], axis=-1)
        Jx0 = tf.einsum('abcdefg,abcdefg->abcdef', J, varstack0)

        eta = J * (Jx0 - h0)[..., None] / self.sigma ** 2.

        E = self.energy(self.var0, robust=False, aggregate=False)
        if self.N_rob is not None:
            k = self.get_k_robust(E, N_rob=self.N_rob)
            eta *= k
            J *= tf.sqrt(k)
        return eta, J / self.sigma

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        factor_eta, factor_J = self.get_eta_J(conn_vars)
        incoming_eta, incoming_Lambda = self._get_incoming_messages()
        # factor_eta, factor_Lambda = self.get_eta_Lambda(conn_vars)
        # factor_eta_plus_msg = factor_eta + incoming_eta
        # factor_Lambda_plus_msg = factor_Lambda + tf.linalg.diag(incoming_Lambda)
        # fac_to_var_eta, fac_to_var_Lambda = \
        #     self.marginalise(factor_eta_plus_msg, factor_Lambda_plus_msg, factor_eta, factor_Lambda)

        fac_to_var_eta, fac_to_var_Lambda = \
            self.marginalise_sherman_morrison(mess_eta=incoming_eta,
                                              factor_eta=factor_eta,
                                              mess_Lambda=incoming_Lambda,
                                              J=factor_J)

        self.filter_var_edges.fac_to_var_eta = fac_to_var_eta[..., 0]
        self.filter_var_edges.fac_to_var_Lambda = fac_to_var_Lambda[..., 0]

        self.coeff_var_edges.fac_to_var_eta = fac_to_var_eta[..., 1]
        self.coeff_var_edges.fac_to_var_Lambda = fac_to_var_Lambda[..., 1]

        self.component_var_edges.fac_to_var_eta = fac_to_var_eta[..., 2]
        self.component_var_edges.fac_to_var_Lambda = fac_to_var_Lambda[..., 2]

    def get_edge_messages(self):
        edges = [list(self.filter_var_edges),
                 list(self.coeff_var_edges),
                 list(self.component_var_edges)]
        return edges

    def set_edge_messages(self, edges):
        filtmsgs, weightmsgs, compmsgs = edges
        for m, message_type in enumerate(('_var_to_fac_eta', '_var_to_fac_Lambda',
                                          '_fac_to_var_eta', '_fac_to_var_Lambda')):
            setattr(self.filter_var_edges, message_type, filtmsgs[m])
            setattr(self.coeff_var_edges, message_type, weightmsgs[m])
            setattr(self.component_var_edges, message_type, compmsgs[m])
        return edges


