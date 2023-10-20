# coding=utf-8
import tensorflow as tf

from core.factors.additive_filter_factor_sum_decomp import AdditiveFilterFactorSumDecomp
from core.utils.utils import flatten_filters, patchify_image


class FeedforwardFilterFactor(AdditiveFilterFactorSumDecomp):
    N_PARALLEL_PROC_MARG = 1

    def __init__(self,
                 *args,
                 **kwargs):
        super(FeedforwardFilterFactor, self).__init__(*args, **kwargs)

    def _make_static_tensors(self):
        pass

    def _get_incoming_messages(self):
        edges = (('inputs', self.input_var_edges),) if not self.fixed_inputs else ()
        if not self.fixed_params:
            edges += (('filters', self.filter_var_edges),)
        edges += (('coeffs', self.coeff_var_edges),)
        if self.use_bias and not self.fixed_params:
            edges += (('biases', self.bias_var_edges),)
        msgs_combined = []
        for mtype in ('eta', 'Lambda'):
            msgs = []
            for en, e in edges:
                msg_in = getattr(e, f'var_to_fac_{mtype}')
                if en in ('filters', 'inputs'):
                    msg_in = tf.transpose(msg_in, (0, 1, 2, 4, 3, 5))
                    msg_in = tf.reshape(msg_in, msg_in.shape.as_list()[:-2] + [-1])
                elif en in ('coeffs', 'biases'):
                    msg_in = msg_in[..., None]
                msgs.append(msg_in)
            msg_combined = tf.concat(msgs, axis=-1)
            msgs_combined.append(msg_combined)
        return msgs_combined

    def energy(self, conn_vars, robust=None, aggregate=True):
        filters, inputs, coeffs = conn_vars[:3]

        conv_pre_act = self.conv2d(inputs, filters)
        if self.use_bias:
            bias = conn_vars[3]
            conv_pre_act += bias
        conv_post_act = self.nonlin(conv_pre_act)
        if coeffs.shape[1] == conv_post_act.shape[1] + 2 * self.coeff_padding:
            coeffs = self.trim_coeffs(coeffs)

        diff = conv_post_act - coeffs
        E = tf.reduce_sum(diff ** 2 / self.sigma ** 2, axis=-1)
        if robust is None:
            robust = self.N_rob is not None
        if robust and self.N_rob is not None:
            E = self._robust_correct_energy(E)
        E = E[..., None]

        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E

    def stack_vars(self, filters, inputs, coeffs, bias=None):
        fshp = filters.shape
        fsize = fshp[0] * fshp[1]
        nfilt = fshp[-1]

        inp_patch = patchify_image(inputs, ksize_x=self.receptive_field[0], stride=self.stride)
        inp_patch_shape = inp_patch.shape.as_list()[:3] + [nfilt] + inp_patch.shape.as_list()[3:]
        inp_patch_stack = tf.broadcast_to(inp_patch[..., None, :, :], inp_patch_shape)
        inp_patch_stack = tf.reshape(inp_patch_stack, inp_patch_stack.shape.as_list()[:-2] + [-1])

        to_stack = [] if self.fixed_inputs else [inp_patch_stack]
        if not self.fixed_params:
            filters_flat = flatten_filters(filters)
            filters_flat_unsqueeze = tf.transpose(filters_flat, (1, 0, 2))[None, None, None]
            filters_flat_stack = tf.reshape(filters_flat_unsqueeze, filters_flat_unsqueeze.shape.as_list()[:-2] + [-1])
            filters_flat_stack = tf.broadcast_to(filters_flat_stack, inp_patch_stack.shape)
            to_stack += [filters_flat_stack]

            # varstack = tf.concat([inp_patch_stack, filters_flat_stack, coeffs_stack], axis=-1)
        if not self.fixed_coeffs:
            coeff_shp = inp_patch_stack.shape[:-1].as_list()
            coeffs_stack = tf.broadcast_to(coeffs, coeff_shp)[..., None]
            to_stack += [coeffs_stack]
        if self.use_bias and not self.fixed_params:
            bias_patch = tf.broadcast_to(bias[None, None, None, :], to_stack[-1].shape[:-1])[..., None]
            to_stack += [bias_patch]
            # varstack = tf.concat([varstack, bias_patch], axis=-1)
        return tf.concat(to_stack, axis=-1)

    def conv2d(self, inputs, filters):
        """Using tf.nn.conv2d() raises warning with static graph mode, write out conv"""
        ksize = filters.shape[0]
        filters_flat = flatten_filters(filters)
        img_patches = patchify_image(inputs, ksize_x=ksize, stride=self.stride)
        conv_pre_act = tf.einsum('...abc,...efgac->...efgb', filters_flat, img_patches)
        return conv_pre_act

    def get_eta_J(self, conn_vars):
        filters, inputs_mu, coeffs = self.var0[:3]   # Covers case both with and without bias
        if self.use_bias:
            bias = self.var0[3]
        else:
            bias = None
        ksize = filters.shape[0]
        filters_flat = flatten_filters(filters)
        N_rob = self.N_rob

        conv_pre_act = self.conv2d(inputs_mu, filters)

            # tf.nn.conv2d(input=inputs_mu,
            #                         filters=filters,
            #                         strides=[self.stride, self.stride],
            #                         padding='VALID')
        if self.use_bias:
            conv_pre_act += bias

        conv_grad = self.nonlin_grad(conv_pre_act)
        filters_flat_unsqueeze = tf.transpose(filters_flat, (1, 0, 2))[None, None, None]

        Jin = conv_grad[..., None, None] * filters_flat_unsqueeze
        J_to_concat = [] if self.fixed_inputs else [tf.reshape(Jin, Jin.shape.as_list()[:-2] + [-1])]
        if not self.fixed_params:
            img_patches = patchify_image(inputs_mu, ksize_x=ksize, stride=self.stride)
            Jfilt = conv_grad[..., None, None] * img_patches[..., None, :, :]
            Jfilt = tf.reshape(Jfilt, Jfilt.shape.as_list()[:-2] + [-1])
            J_to_concat += [Jfilt]
        if not self.fixed_coeffs:
            J_to_concat += [-tf.ones_like(coeffs)[..., None]]
        if not self.fixed_params and self.use_bias:
            Jbias = [conv_grad[..., None]]
            J_to_concat += Jbias
        J = tf.concat(J_to_concat, axis=-1)
        varstack0 = self.stack_vars(filters=filters, inputs=inputs_mu, coeffs=coeffs, bias=bias)
        JTx0 = tf.reduce_sum(J * varstack0, -1) # tf.einsum('abcdef,abcdef->abcde', J, varstack0, optimize='optimal')
        h0 = self.nonlin(conv_pre_act) - coeffs

        eta = J * (JTx0 - h0)[..., None] / self.sigma ** 2.

        if self.N_rob is not None:
            E = self.energy(self.var0,
                            robust=False,
                            aggregate=False)
            k = self.get_k_robust(E, N_rob=N_rob)[..., None, :]
            eta *= k
            J *= tf.sqrt(k)

        return eta, J

    def update_outgoing_messages(self, conn_vars, **kwargs):
        var_msg_in_eta, var_msg_in_Lambda = self._get_incoming_messages()

        factor_eta, factor_J = self.get_eta_J(conn_vars)

        # Marginalisation
        factor_to_var_eta, factor_to_var_Lambda = \
            self.marginalise_sherman_morrison(tf.cast(var_msg_in_eta, tf.float64),
                                              tf.cast(factor_eta, tf.float64),
                                              tf.cast(var_msg_in_Lambda, tf.float64),
                                              tf.cast(factor_J / self.sigma, tf.float64))
        factor_to_var_eta = tf.cast(factor_to_var_eta, tf.float32)
        factor_to_var_Lambda = tf.cast(factor_to_var_Lambda, tf.float32)
        self.update_outgoing_edge_messages(factor_to_var_eta, factor_to_var_Lambda)

    def update_outgoing_edge_messages(self, factor_to_var_eta, factor_to_var_Lambda):
        ks2 = self.filter_var_edges.shape[-1]
        nin_chan = self.filter_var_edges.shape[-3]
        ninps = int(not self.fixed_inputs) * ks2 * nin_chan  # Number of input vars connected to factor
        nfilt = int(not self.fixed_params) * nin_chan * ks2  # Number of filter variables (used to reconstruct 1 pixel)
        ncoeff = self.n_filt
        nbias = int(self.use_bias and not self.fixed_params)

        def reshape(x):
            x = tf.reshape(x, x.shape.as_list()[:-1] + [nin_chan, ks2])
            return tf.transpose(x, (0, 1, 2, 4, 3, 5))

        if not self.fixed_inputs:
            self.input_var_edges.fac_to_var_eta = reshape(factor_to_var_eta[..., :ninps])
            self.input_var_edges.fac_to_var_Lambda = reshape(factor_to_var_Lambda[..., :ninps])

        if not self.fixed_params:
            self.filter_var_edges.fac_to_var_eta = reshape(factor_to_var_eta[..., ninps:ninps + nfilt])
            self.filter_var_edges.fac_to_var_Lambda = reshape(factor_to_var_Lambda[..., ninps:ninps + nfilt])
            if self.use_bias:
                self.bias_var_edges.fac_to_var_eta = factor_to_var_eta[..., -1]
                self.bias_var_edges.fac_to_var_Lambda = factor_to_var_Lambda[..., -1]
        if not self.fixed_coeffs:
            self.coeff_var_edges.fac_to_var_eta = factor_to_var_eta[..., -(1 + nbias)]
            self.coeff_var_edges.fac_to_var_Lambda = factor_to_var_Lambda[..., -(1 + nbias)]

    def forward_deterministic(self, inputs=None):
        filters, inputs_var, _ = self.var0[:3]
        if self.use_bias:
            bias = self.var0[3]
        else:
            bias = 0.
        inputs = inputs if inputs is not None else inputs_var
        return self.nonlin(self.conv2d(inputs, filters) + bias)


