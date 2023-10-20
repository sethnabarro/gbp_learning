# coding=utf-8
import numpy as np
import tensorflow as tf

from core.factors.base import Factor


class UpsampleFactor(Factor):
    def __init__(self, sigma, input_var_edges, output_var_edges, N_rob=None,
                 rob_type='tukey', kmult=1.):
        super(UpsampleFactor, self).__init__(sigma=sigma,
                                             N_rob=N_rob,
                                             rob_type=rob_type,
                                             kmult=kmult)
        self.input_var_edges = input_var_edges
        self.output_var_edges = output_var_edges
        self.ksize = int(np.sqrt(input_var_edges.fac_to_var_eta.shape[-1]))
        self._set_eta_Lambda()
        self.fix_attr_fields()

    def _set_eta_Lambda(self):
        self._eta = tf.zeros([1] * 4 + [2])
        self._J = tf.convert_to_tensor([[1., -1]])[None, None, None]
        self._Lambda = self._J * tf.transpose(self._J) / self.sigma ** 2.

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        # Gather var to factor messages
        var_msg_in_eta, var_msg_in_Lambda = self._get_incoming_messages()

        # Do marginalisation to get outgoing messages
        # TODO: factor only connects to 2 vars, writing out marg explicitly probs faster
        fac_to_var_eta, fac_to_var_Lambda = \
            self.marginalise_sherman_morrison(mess_eta=var_msg_in_eta,
                                              factor_eta=tf.broadcast_to(self._eta, var_msg_in_eta.shape),
                                              mess_Lambda=var_msg_in_Lambda,
                                              J_div_sigma=tf.broadcast_to(self._J / self.sigma, var_msg_in_eta.shape))

        # Update outgoing message state
        self.input_var_edges.fac_to_var_eta = fac_to_var_eta[..., 0]
        self.input_var_edges.fac_to_var_Lambda = fac_to_var_Lambda[..., 0]
        self.output_var_edges.fac_to_var_eta = fac_to_var_eta[..., 1]
        self.output_var_edges.fac_to_var_Lambda = fac_to_var_Lambda[..., 1]

    def _get_incoming_messages(self):
        msgs_combined = []
        for mtype in ('eta', 'Lambda'):
            invartofac = getattr(self.input_var_edges, f'var_to_fac_{mtype}')[..., None]
            outvartofac = getattr(self.output_var_edges, f'var_to_fac_{mtype}')[..., None]
            msgs_combined.append(tf.concat([invartofac, outvartofac], axis=-1))
        return msgs_combined

    def energy(self, conn_vars, robust=None, aggregate=True):
        with tf.name_scope('ups_E'):
            inputs, coeffs = conn_vars
            co_shp = coeffs.shape.as_list()
            coeffs_extra = tf.broadcast_to(coeffs[..., None, None],
                                           co_shp + [self.ksize, self.ksize])
            coeffs_extra = tf.transpose(coeffs_extra, [0, 1, 4, 2, 5, 3])
            coeffs_ups = tf.reshape(coeffs_extra, [co_shp[0],
                                                   co_shp[1] * self.ksize,
                                                   co_shp[2] * self.ksize,
                                                   co_shp[3]])
            E = ((inputs - coeffs_ups) / self.sigma) ** 2.
            if robust is None:
                robust = self.N_rob is not None
            if robust and self.N_rob is not None:
                E = self._robust_correct_energy(E)
            E = E[..., None]

            if aggregate:
                return tf.reduce_sum(E)
            else:
                return E

    def get_edge_messages(self, named=False):
        if named:
            edges = [(str(self.input_var_edges), self.input_var_edges.named_state),
                     (str(self.output_var_edges), self.output_var_edges.named_state)]
        else:
            edges = [self.input_var_edges.state,
                     self.output_var_edges.state]
        return edges

    def set_edge_messages(self, edges):
        self.input_var_edges.state, \
            self.output_var_edges.state = edges
        return edges

    def forward_deterministic(self, coeffs):
        co_shp = coeffs.shape.as_list()
        coeffs_extra = tf.broadcast_to(coeffs[..., None, None],
                                       co_shp + [self.ksize, self.ksize])
        coeffs_extra = tf.transpose(coeffs_extra, [0, 1, 4, 2, 5, 3])
        coeffs_ups = tf.reshape(coeffs_extra, [co_shp[0],
                                               co_shp[1] * self.ksize,
                                               co_shp[2] * self.ksize,
                                               co_shp[3]])
        return coeffs_ups
