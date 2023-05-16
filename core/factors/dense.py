# coding=utf-8
import tensorflow as tf
from typing import Iterable

from core.factors.base_nonlin import NonLinearFactor
from core.edge import Edge


class DenseFactor(NonLinearFactor):
    def __init__(self,
                 sigma: float,
                 output_var_edges: Edge,
                 weight_var_edges: Edge,
                 input_var_edges: [Edge, None] = None,
                 bias_var_edges: [Edge, None] = None,
                 noiseless_input: bool = False,
                 input_obs: [tf.Tensor, None] = None,
                 N_rob: [float, None] = None,
                 rob_type: [str, None] = None,
                 init_lin_point: [Iterable, None] = None,
                 relin_freq: [int, None] = None,
                 kmult: [float, None] = None,
                 nonlin: [str, None] = None,
                 nonlin_xscale: float = 1.,
                 nonlin_yscale: float = 1.,
                 fac_to_var_chunksize: int = 1):
        super(DenseFactor, self).__init__(sigma,
                                          init_lin_point=init_lin_point,
                                          relin_freq=relin_freq,
                                          N_rob=N_rob,
                                          rob_type=rob_type,
                                          kmult=kmult)
        if noiseless_input:
            assert input_obs is not None
        else:
            assert input_var_edges is not None, \
                "Must provide `input_var_edges` is input is random variable."
        self.is_noiseless_input = noiseless_input
        self.input_obs = input_obs
        self.input_var_edges = input_var_edges
        self.weight_var_edges = weight_var_edges
        self.output_var_edges = output_var_edges
        self.bias_var_edges = bias_var_edges
        self.use_bias = isinstance(bias_var_edges, Edge)
        self.fixed_params = False
        self.fac_to_var_chunksize = fac_to_var_chunksize

        assert len(init_lin_point) == 2 + int(not self.is_noiseless_input) + int(self.use_bias),\
            "Length of `init_lin_point` list should equal number of vars connected to dense factor"

        self.nonlin, self.nonlin_grad = \
            self.build_nonlin_and_grad(nonlin, nonlin_xscale, nonlin_yscale)

    def _get_incoming_messages(self):
        edges = tuple() if self.is_noiseless_input else (('inputs', self.input_var_edges),)
        edges += tuple() if self.fixed_params else (('weights', self.weight_var_edges),)
        edges += tuple() if self.fixed_params or not self.use_bias else (('bias', self.bias_var_edges),)
        edges += (('outputs', self.output_var_edges),)

        msgs_combined = []
        for mtype in ('eta', 'Lambda'):
            msgs = []
            for en, e in edges:
                msg = getattr(e, f'var_to_fac_{mtype}')
                if en in ('outputs', 'bias'):
                    msg = msg[..., None]
                msgs.append(msg)
            msgs_combined.append(tf.concat(msgs, axis=-1))
        return msgs_combined

    def _unpack_vars(self, varlist):
        outputs = varlist[-1]
        bias = varlist[-2] if self.use_bias else None
        weights = varlist[-(2 + int(self.use_bias))]
        inputs = self.input_obs if self.is_noiseless_input else varlist[0]
        return inputs, weights, bias, outputs

    def stack_vars(self, inputs, weights, bias, outputs):
        inputs_bc = tf.broadcast_to(inputs[..., None, :], inputs.shape.as_list()[:1] + weights.shape.as_list()[-2:])
        weights_bc = tf.broadcast_to(weights[None], inputs_bc.shape)
        varstack = outputs[..., None]
        if not self.fixed_params:
            if self.use_bias:
                bias_bc = tf.broadcast_to(bias[None], outputs.shape)[..., None]
                varstack = tf.concat([bias_bc, varstack], axis=-1)
            varstack = tf.concat([weights_bc, varstack], axis=-1)
        if not self.is_noiseless_input:
            varstack = tf.concat([inputs_bc, varstack], axis=-1)
        return varstack

    def get_eta_J(self, conn_vars):
        """
        Compute the factor eta and Jacobian

        Factor energy is
            $ E(x, y, w, b) = (( y - g( w^Tx + b) ) / \sigma )^2 $
                where
                    x is vector input
                    y is vector output
                    w are the weights
                    b is the bias
                    g(.) is optional nonlinearity

        Measurement function
            $ h(x, y, w, b) = y - g( w^Tx + b) $

        Jacobian
            $ J_x := \partial h(x, y, w, b) / \partial x = -(\partial g( w^Tx + b) / \partial (w^Tx + b)) w $
            $ J_w := \partial h(x, y, w, b) / \partial w = -(\partial g( w^Tx + b) / \partial (w^Tx + b) x $
            $ J_b := \partial h(x, y, w, b) / \partial b = -(\partial g( w^Tx + b) / \partial (w^Tx + b)) . 1 $
            $ J_y := \partial h(x, y, w, b) / \partial y = 1 $   (vector of ones)
        """
        # Current linearisation point
        inputs, weights, bias, outputs = self._unpack_vars(self.var0)

        inputs = tf.reshape(inputs, [inputs.shape[0], tf.reduce_prod(inputs.shape[1:])])
        # outputs = tf.transpose(outputs)

        # Compute Jacobian
        weightsTinputs = self.forward(inputs, weights, bias)
        weights = tf.linalg.matrix_transpose(weights)
        J_out = tf.ones_like(outputs)
        J = J_out[..., None]
        if not self.fixed_params:
            if self.use_bias:
                J_b = - tf.broadcast_to(self.nonlin_grad(weightsTinputs), weightsTinputs.shape)[..., None]
                J = tf.concat([J_b, J], axis=-1)
            J_w = - tf.broadcast_to(self.nonlin_grad(weightsTinputs), weightsTinputs.shape)[..., None] * inputs[:, None]
            J = tf.concat([J_w, J], axis=-1)
        if not self.is_noiseless_input:
            J_in = - tf.broadcast_to(self.nonlin_grad(weightsTinputs), weightsTinputs.shape)[..., None] * weights
            J = tf.concat([J_in, J], axis=-1)

        # Calculate info
        varstack = self.stack_vars(inputs, weights, bias, outputs)
        JTx0 = tf.reduce_sum(J * varstack, axis=-1)
        h0 = outputs - self.nonlin(weightsTinputs)
        eta = J * (JTx0 - h0)[..., None] / self.sigma ** 2.

        return eta, J

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        # Get messages from var to factor
        in_msg_eta, in_msg_Lambda = self._get_incoming_messages()

        # Compute the Jacobian and the info of the factor
        fac_eta, fac_J = self.get_eta_J(self.var0)

        # Marginalise
        fac_to_var_eta, fac_to_var_Lambda = \
            self.marginalise_sherman_morrison(mess_eta=in_msg_eta,
                                              mess_Lambda=in_msg_Lambda,
                                              factor_eta=fac_eta,
                                              J_div_sigma=fac_J / self.sigma,
                                              batchsize=self.fac_to_var_chunksize)

        # Update messages along edges
        n_in_var = 0 if self.is_noiseless_input else self.input_var_edges.fac_to_var_eta.shape[-1]
        n_weight_var = 0 if self.fixed_params else self.weight_var_edges.fac_to_var_eta.shape[-1]
        n_bias_var = self.bias_var_edges.fac_to_var_eta.shape[-1] if self.use_bias and not self.fixed_params else 0
        start_id = 0
        if not self.is_noiseless_input:
            self.input_var_edges.fac_to_var_eta = fac_to_var_eta[..., :n_in_var]
            self.input_var_edges.fac_to_var_Lambda = fac_to_var_Lambda[..., :n_in_var]
            start_id += n_in_var
        if not self.fixed_params:
            self.weight_var_edges.fac_to_var_eta = fac_to_var_eta[..., start_id:start_id + n_weight_var]
            self.weight_var_edges.fac_to_var_Lambda = fac_to_var_Lambda[..., start_id:start_id + n_weight_var]
            start_id += n_weight_var
            if self.use_bias:
                self.bias_var_edges.fac_to_var_eta = fac_to_var_eta[..., start_id]
                self.bias_var_edges.fac_to_var_Lambda = fac_to_var_Lambda[..., start_id]
                start_id += n_bias_var
        self.output_var_edges.fac_to_var_eta = fac_to_var_eta[..., -1]
        self.output_var_edges.fac_to_var_Lambda = fac_to_var_Lambda[..., -1]

    def get_edge_messages(self, named=False):
        attr_to_get = 'named_state' if named else 'state'
        enames = []
        if self.is_noiseless_input:
            enames += [str(self.input_var_edges)]
            edges = [getattr(self.input_var_edges, attr_to_get)]
        else:
            edges = []
        edges += [getattr(self.weight_var_edges, attr_to_get)]
        enames += [str(self.weight_var_edges)]
        if self.use_bias:
            edges += [getattr(self.bias_var_edges, attr_to_get)]
            enames += [str(self.bias_var_edges)]
        edges += [getattr(self.output_var_edges, attr_to_get)]
        enames += [str(self.output_var_edges)]
        if named:
            return list(zip(enames, edges))
        else:
            return edges

    def set_edge_messages(self, edges):
        self.output_var_edges.state = edges[-1]
        if self.use_bias:
            self.bias_var_edges.state = edges[-2]
        self.weight_var_edges.state = edges[-(2 + int(self.use_bias))]
        if not self.is_noiseless_input:
            self.input_var_edges.state = edges[0]
        return edges

    def energy(self, conn_vars, robust=None, aggregate=True):
        inputs, weights, bias, outputs = self._unpack_vars(conn_vars)
        inputs = tf.reshape(inputs, [inputs.shape[0], -1])  # Flatten
        weightsTinputs = self.forward(inputs, weights, bias)
        E = ((outputs - self.nonlin(weightsTinputs)) / self.sigma) ** 2.
        if robust is None:
            robust = self.N_rob is not None
        if robust and self.N_rob is not None:
            E = self._robust_correct_energy(E)
        E = E[..., None]

        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E

    def forward_deterministic(self, inputs=None):
        inputs_var, weights, bias, outputs = self._unpack_vars(self.var0)
        inputs = inputs if inputs is not None else inputs_var
        return self.forward(inputs, weights, bias)

    def forward(self, inputs, weights, bias=None):
        weightsTinputs = tf.einsum('ba,cb->ca', weights, inputs)
        if self.use_bias:
            assert bias is not None
            weightsTinputs += bias
        return weightsTinputs


