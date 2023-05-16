# coding=utf-8
import numpy as np
import tensorflow as tf

from core.factors.base import Factor


class AvgPoolFactor(Factor):
    def __init__(self, sigma, input_var_edges, output_var_edges, N_rob=None,
                 rob_type='tukey', kmult=1.):
        super(AvgPoolFactor, self).__init__(sigma=sigma,
                                            N_rob=N_rob,
                                            rob_type=rob_type,
                                            kmult=kmult)
        self.input_var_edges = input_var_edges
        self.output_var_edges = output_var_edges
        self.ksize = int(np.sqrt(input_var_edges.fac_to_var_eta.shape[-1]))
        self._set_eta_Lambda()
        self.fix_attr_fields()

    def _set_eta_Lambda(self):
        self._eta = tf.zeros([1] * 4 + [self.ksize ** 2 + 1])
        self._J = tf.convert_to_tensor([[1. / self.ksize ** 2.] * self.ksize ** 2 + [-1]])[None, None, None]
        self._Lambda = self._J * tf.transpose(self._J) / self.sigma ** 2.

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        # Gather var to factor messages
        var_msg_in_eta, var_msg_in_Lambda = self._get_incoming_messages()

        # Do marginalisation to get outgoing messages
        fac_to_var_eta, fac_to_var_Lambda = \
            self.marginalise_sherman_morrison(mess_eta=var_msg_in_eta,
                                              factor_eta=tf.broadcast_to(self._eta, var_msg_in_eta.shape),
                                              mess_Lambda=var_msg_in_Lambda,
                                              J_div_sigma=tf.broadcast_to(self._J / self.sigma, var_msg_in_eta.shape))

        # Update outgoing message state
        self.input_var_edges.fac_to_var_eta = fac_to_var_eta[..., :self.ksize ** 2]
        self.input_var_edges.fac_to_var_Lambda = fac_to_var_Lambda[..., :self.ksize ** 2]
        self.output_var_edges.fac_to_var_eta = fac_to_var_eta[..., -1]
        self.output_var_edges.fac_to_var_Lambda = fac_to_var_Lambda[..., -1]

    def _get_incoming_messages(self):
        msgs_combined = []
        for mtype in ('eta', 'Lambda'):
            invartofac = getattr(self.input_var_edges, f'var_to_fac_{mtype}')
            outvartofac = getattr(self.output_var_edges, f'var_to_fac_{mtype}')[..., None]
            msgs_combined.append(tf.concat([invartofac, outvartofac], axis=-1))
        return msgs_combined

    def energy(self, conn_vars, robust=None, aggregate=True):
        inputs, coeffs = conn_vars
        input_pool = tf.nn.avg_pool2d(inputs, ksize=self.ksize, padding='VALID', strides=self.ksize)
        E = ((input_pool - coeffs) / self.sigma) ** 2.
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
