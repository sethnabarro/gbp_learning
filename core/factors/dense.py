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
                 fac_to_var_chunksize: int = 1,
                 decompose=False,
                 inverted=False):
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
        self.is_noiseless_input = bool(noiseless_input)
        self.input_obs = input_obs
        self.input_var_edges = input_var_edges
        self.weight_var_edges = weight_var_edges
        self.output_var_edges = output_var_edges
        self.bias_var_edges = bias_var_edges
        self.use_bias = isinstance(bias_var_edges, Edge)
        self.fixed_params = False
        self.fac_to_var_chunksize = fac_to_var_chunksize
        self.decompose = decompose
        self.inverted = inverted

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
                if en in ('outputs', 'bias') and self.decompose:
                    msg = msg[..., None]
                elif en in ('weights',) and not self.decompose:
                    # msg = tf.transpose(msg, (0, 2, 1))
                    msg = tf.reshape(msg, [msg.shape[0], -1])
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
        if self.decompose:
            inputs_bc = tf.broadcast_to(inputs[..., None, :], inputs.shape.as_list()[:1] + weights.shape.as_list()[-2:])
            weights_bc = tf.broadcast_to(weights[None], inputs_bc.shape)
        else:
            inputs_bc = inputs
            weights_bc = tf.broadcast_to(weights[None], [inputs_bc.shape[0]] + weights.shape.as_list())
            # weights_bc = tf.transpose(weights_bc, (0, 2, 1))
            weights_bc = tf.reshape(weights_bc, [inputs_bc.shape[0], -1])

        varstack = outputs[..., None] if self.decompose else outputs
        if not self.fixed_params:
            if self.use_bias:
                bias_bc = tf.broadcast_to(bias[None], outputs.shape)
                if self.decompose:
                    bias_bc = bias_bc[..., None]
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

        # Compute Jacobian
        if self.inverted:
            wTx = self.forward(outputs, tf.linalg.matrix_transpose(weights), bias)
        else:
            wTx = self.forward(inputs, weights, bias)
            weights = tf.linalg.matrix_transpose(weights)
        if self.inverted:
            J_out = - tf.broadcast_to(self.nonlin_grad(wTx), wTx.shape)[..., None] * weights
        else:
            J_out = tf.ones_like(outputs)[..., None] if self.decompose else tf.ones_like(outputs)
        J = J_out
        if not self.fixed_params:
            if self.use_bias:
                J_b = - tf.broadcast_to(self.nonlin_grad(wTx), wTx.shape)
                if self.decompose:
                    J_b = J_b[..., None]
                J = tf.concat([J_b, J], axis=-1)
            if self.inverted:
                J_w = - tf.broadcast_to(self.nonlin_grad(wTx), wTx.shape)[..., None] * outputs[:, None]
            else:
                J_w = - tf.broadcast_to(self.nonlin_grad(wTx), wTx.shape)[..., None] * inputs[:, None]
            if not self.decompose:
                J_w = tf.reshape(J_w, [J_w.shape[0], -1])
            J = tf.concat([J_w, J], axis=-1)
        if not self.is_noiseless_input:
            if self.inverted:
                J_in = tf.ones_like(inputs)[..., None]
            else:
                J_in = - tf.broadcast_to(self.nonlin_grad(wTx), wTx.shape)[..., None] * weights
            if not self.decompose:
                J_in = tf.reduce_sum(J_in, axis=-2)
            J = tf.concat([J_in, J], axis=-1)

        if not self.decompose:
            out_dim, in_dim = self.weight_var_edges.shape[1:3]
            n_in_vars = J_in.shape[1] if not self.is_noiseless_input else 0
            n_weight_vars = J_w.shape[1] if not self.fixed_params else 0
            n_bias_vars = J_b.shape[1] if self.use_bias and not self.fixed_params else 0
            Js_masked = []
            for o in range(out_dim):
                cond = tf.range(J.shape[1]) < n_in_vars
                if not self.fixed_params:
                    cond = tf.logical_or(cond, tf.logical_and(tf.range(J.shape[1]) >= n_in_vars + o * in_dim, tf.range(J.shape[1]) < n_in_vars + (o + 1) * in_dim))
                    if self.use_bias:
                        cond = tf.logical_or(cond, tf.range(J.shape[1]) == n_in_vars + n_weight_vars + o)
                cond = tf.logical_or(cond, tf.range(J.shape[1]) == n_in_vars + n_weight_vars + n_bias_vars + o)
                Js_masked.append(tf.where(cond[None], J, tf.zeros_like(J))[..., None])
            J = tf.concat(Js_masked, axis=-1)

        # Calculate info
        varstack = self.stack_vars(inputs, weights, bias, outputs)
        if not self.decompose:
            varstack = varstack[..., None]
        h0 = inputs - self.nonlin(wTx) if self.inverted else outputs - self.nonlin(wTx)
        if self.decompose:
            JTx0 = tf.reduce_sum(J * varstack, axis=-1)
            eta = J * (JTx0 - h0)[..., None] / self.sigma ** 2.
        else:
            JTx0 = tf.reduce_sum(J * varstack, axis=1)
            eta = tf.reduce_sum(J * (JTx0 - h0)[..., None, :] / self.sigma ** 2., axis=-1)

        return eta, J

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        # Get messages from var to factor
        in_msg_eta, in_msg_Lambda = self._get_incoming_messages()

        # Compute the Jacobian and the info of the factor
        fac_eta, fac_J = self.get_eta_J(self.var0)
        shp_orig = fac_eta.shape

        if 'factor_ids' in kw_msgs_in:
            in_msg_eta = tf.gather(in_msg_eta, indices=kw_msgs_in['factor_ids'], axis=1)
            in_msg_Lambda = tf.gather(in_msg_Lambda, indices=kw_msgs_in['factor_ids'], axis=1)
            fac_eta = tf.gather(fac_eta, indices=kw_msgs_in['factor_ids'], axis=1)
            fac_J = tf.gather(fac_J, indices=kw_msgs_in['factor_ids'], axis=1)

        if self.decompose:
            # Marginalise
            fac_to_var_eta, fac_to_var_Lambda = \
                self.marginalise_sherman_morrison(mess_eta=tf.cast(in_msg_eta, tf.float64),
                                                  mess_Lambda=tf.cast(in_msg_Lambda, tf.float64),
                                                  factor_eta=tf.cast(fac_eta, tf.float64),
                                                  J_div_sigma=tf.cast(fac_J / self.sigma, tf.float64))
            fac_to_var_eta = tf.cast(fac_to_var_eta, tf.float32)
            fac_to_var_Lambda = tf.cast(fac_to_var_Lambda, tf.float32)
        else:
            fac_Lambda = tf.reduce_sum(fac_J[..., None, :] * fac_J[..., None, :, :], axis=-1) / self.sigma ** 2.
            fac_to_var_eta, fac_to_var_Lambda = \
                self.marginalise(factor_plus_mess_eta=fac_eta + in_msg_eta,
                                 factor_plus_mess_Lambda=fac_Lambda + tf.linalg.diag(in_msg_Lambda),
                                 factor_eta=fac_eta,
                                 factor_Lambda=fac_Lambda)

        if 'factor_ids' in kw_msgs_in:
            is_updated = tf.cast(tf.reduce_sum(tf.one_hot(kw_msgs_in['factor_ids'], depth=shp_orig[1]), axis=0), dtype=tf.bool)[None]
            fac_to_var_eta = tf.repeat(fac_to_var_eta, shp_orig[1], axis=1)
            fac_to_var_Lambda = tf.repeat(fac_to_var_Lambda, shp_orig[1], axis=1)
        else:
            is_updated = tf.cast(tf.ones((shp_orig[1],)), tf.bool)[None]

        # Update messages along edges
        n_in_var = 0 if self.is_noiseless_input else self.input_var_edges.fac_to_var_eta.shape[-1]
        n_weight_var = 0 if self.fixed_params else (self.weight_var_edges.fac_to_var_eta.shape[-1] if self.decompose else tf.reduce_prod(self.weight_var_edges.fac_to_var_eta.shape[-2:]))
        n_bias_var = self.bias_var_edges.fac_to_var_eta.shape[-1] if self.use_bias and not self.fixed_params else 0
        start_id = 0
        if not self.is_noiseless_input:
            self.input_var_edges.fac_to_var_eta = fac_to_var_eta[..., :n_in_var]
            self.input_var_edges.fac_to_var_Lambda = fac_to_var_Lambda[..., :n_in_var]
            start_id += n_in_var
        if not self.fixed_params:
            if self.decompose:
                fac_to_weight_eta = fac_to_var_eta[..., start_id:start_id + n_weight_var]
                fac_to_weight_Lambda = fac_to_var_Lambda[..., start_id:start_id + n_weight_var]
            else:
                fac_to_weight_eta = tf.reshape(fac_to_var_eta[..., start_id:start_id + n_weight_var], self.weight_var_edges.fac_to_var_eta.shape)
                fac_to_weight_Lambda = tf.reshape(fac_to_var_Lambda[..., start_id:start_id + n_weight_var], self.weight_var_edges.fac_to_var_Lambda.shape)
            self.weight_var_edges.fac_to_var_eta = fac_to_weight_eta
            self.weight_var_edges.fac_to_var_Lambda = fac_to_weight_Lambda
            start_id += n_weight_var
            if self.use_bias:
                if self.decompose:
                    fac_to_bias_eta = fac_to_var_eta[..., start_id]
                    fac_to_bias_Lambda = fac_to_var_Lambda[..., start_id]
                else:
                    fac_to_bias_eta = tf.reshape(fac_to_var_eta[..., start_id:start_id + n_bias_var],
                                                   self.bias_var_edges.fac_to_var_eta.shape)
                    fac_to_bias_Lambda = tf.reshape(fac_to_var_Lambda[..., start_id:start_id + n_bias_var],
                                                      self.bias_var_edges.fac_to_var_Lambda.shape)
                self.bias_var_edges.fac_to_var_eta = fac_to_bias_eta
                self.bias_var_edges.fac_to_var_Lambda = fac_to_bias_Lambda
                start_id += n_bias_var
        self.output_var_edges.fac_to_var_eta = fac_to_var_eta[..., -1] if self.decompose else fac_to_var_eta[..., start_id:]
        self.output_var_edges.fac_to_var_Lambda = fac_to_var_Lambda[..., -1] if self.decompose else fac_to_var_Lambda[..., start_id:]

    def get_edge_messages(self, named=False):
        attr_to_get = 'named_state' if named else 'state'
        enames = []
        if not self.is_noiseless_input:
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
        if self.inverted:
            weightsToutputs = self.forward(outputs, weights, bias)
            E = ((inputs - self.nonlin(weightsToutputs)) / self.sigma) ** 2.
        else:
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
        weights = tf.linalg.matrix_transpose(weights) if self.inverted else weights
        if inputs is not None:
            inputs = tf.reshape(inputs, [inputs.shape[0], -1])   # Flatten
        elif self.inverted:
            inputs = outputs
        inputs = inputs if inputs is not None else inputs_var
        return self.nonlin(self.forward(inputs, weights, bias))

    def forward(self, inputs, weights, bias=None):
        weightsTinputs = tf.einsum('...ba,...cb->...ca', weights, inputs)
        if self.use_bias:
            assert bias is not None
            weightsTinputs += bias
        return weightsTinputs


