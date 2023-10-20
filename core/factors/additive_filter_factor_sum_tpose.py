# coding=utf-8
import tensorflow as tf

from core.factors.additive_filter_factor_sum_decomp import AdditiveFilterFactorSumDecomp
from core.utils.utils import flatten_filters, patchify_image

LAMBDA_MSG_PAD = 1e-6   # Enforced prec of msgs incoming to padded coeffs


class AdditiveFilterFactorSumTpose(AdditiveFilterFactorSumDecomp):
    """Convolutional reconstruction factor, differs from the other conv reconstruction
    factors as recon signals to an input from all neighbourhoods are summed before the
    constraint is enforced. In the other convolutional reconstruction factors, a constraint
    is instead enforced on each neighbourhood. Equivalent `tf.nn.conv2d_transpose()`.
    Implemented as a FF convolution where the coefficient variables are inputs and the
    input variables are the outputs."""

    def __init__(self,
                 *args,
                 **kwargs):
        super(AdditiveFilterFactorSumTpose, self).__init__(*args, **kwargs)

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
                if en in ('filters', 'coeffs'):
                    if mtype == 'Lambda':
                        # Coeff and filter prec messages in padded region may be 0.
                        # Add small positive constant to ensure diagonal matrix of
                        # messages is invertible.
                        # Note this doesn't impact the outgoing messages as the
                        # Jacobian elements corresponding to filters and coefficients
                        # are forced to zero in padded region
                        mask_Lam = (1. - self.get_mask_patches(self.var0[2].shape)) * LAMBDA_MSG_PAD
                        msg_in += mask_Lam
                    msg_in = tf.reshape(msg_in, msg_in.shape.as_list()[:-2] + [-1])
                elif en in ('inputs', 'biases'):
                    msg_in = msg_in[..., None]
                msgs.append(msg_in)
            msg_combined = tf.concat(msgs, axis=-1)
            msgs_combined.append(msg_combined)
        return msgs_combined

    def pad_coeffs(self, coeffs, padval=0.):
        ksize = self.receptive_field[0]
        cpad = ksize // 2 * 2
        return tf.pad(coeffs, [[0, 0], [cpad, cpad], [cpad, cpad], [0, 0]], constant_values=padval)

    def energy(self, conn_vars, robust=None, aggregate=True):
        with tf.name_scope('perpixE'):
            filters, inputs, coeffs = conn_vars[:3]
            conv_pre_act = self.conv2dtranspose(coeffs, filters)
            if self.use_bias:
                bias = conn_vars[3]
                conv_pre_act += bias
            conv_post_act = self.nonlin(conv_pre_act)

            diff = conv_post_act - inputs
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

    def get_mask_patches(self, coeffs_shp, collapse_last_dim=False, padval=0.):
        mask_pad = self.pad_coeffs(tf.ones(coeffs_shp), padval=padval)
        mask_pad = patchify_image(mask_pad, ksize_x=self.receptive_field[0], stride=self.stride)[..., None, :, :]
        if collapse_last_dim:
            mask_pad = tf.reshape(mask_pad, mask_pad.shape.as_list()[:-2] + [-1])
        return mask_pad

    def stack_vars(self, filters, inputs, coeffs, bias=None):
        with tf.name_scope('tpose_recon_stack_vars'):
            fshp = filters.shape
            nfilt = fshp[-1]
            recfield = self.receptive_field[0] * self.receptive_field[1]

            to_stack = []
            if not self.fixed_inputs:
                to_stack += [inputs[..., None]]

            if not self.fixed_params:
                filters_flat = flatten_filters(filters)
                filters_flat_unsqueeze = filters_flat[None, None, None]
                filts_shape = inputs.shape.as_list() + [nfilt * recfield]
                filters_flat_stack = tf.reshape(filters_flat_unsqueeze, filts_shape[3:])
                filters_flat_stack = tf.broadcast_to(filters_flat_stack, filts_shape)
                filter_mask = self.get_mask_patches(coeffs.shape, collapse_last_dim=True)
                to_stack += [filters_flat_stack * filter_mask]

            if not self.fixed_coeffs:
                coeffs_pad = self.pad_coeffs(coeffs)
                coeff_patch = patchify_image(coeffs_pad, ksize_x=self.receptive_field[0], stride=self.stride)
                coeff_patch_shape = inputs.shape.as_list() + [nfilt, recfield]
                coeff_patch_stack = tf.broadcast_to(coeff_patch[..., None, :, :], coeff_patch_shape)
                coeff_patch_stack = tf.reshape(coeff_patch_stack, coeff_patch_stack.shape.as_list()[:-2] + [-1])
                coeffs_mask = self.get_mask_patches(coeffs.shape, collapse_last_dim=True)
                to_stack += [coeff_patch_stack * coeffs_mask]

            if self.use_bias and not self.fixed_params:
                bias_patch = tf.broadcast_to(bias[None, None, None, :], to_stack[-1].shape[:-1])[..., None]
                to_stack += [bias_patch]
                # varstack = tf.concat([varstack, bias_patch], axis=-1)
            return tf.concat(to_stack, axis=-1)

    def conv2dtranspose(self, inputs, filters):
        """Using tf.nn.conv2d() raises warning with static graph mode, write out conv"""
        ksize = filters.shape[0]
        filters_flat = flatten_filters(filters)
        with tf.name_scope('pad'):
            inputs_pad = self.pad_coeffs(inputs)
        with tf.name_scope('patfy'):
            inputs_patches = patchify_image(inputs_pad, ksize_x=ksize, stride=self.stride)
        conv_pre_act = tf.einsum('...abc,...efgbc->...efga', filters_flat, inputs_patches)
        return conv_pre_act

    def get_eta_J(self, conn_vars):
        filters, inputs_mu, coeffs = self.var0[:3]   # Covers case both with and without bias

        if self.use_bias:
            bias = self.var0[3]
        else:
            bias = None
        filters_flat = flatten_filters(filters)
        N_rob = self.N_rob
        with tf.name_scope('per_pix_recon_J_eta_prep'):
            conv_pre_act = self.conv2dtranspose(coeffs, filters)
            if self.use_bias:
                conv_pre_act += bias

            conv_grad = self.nonlin_grad(conv_pre_act)
            Jin = -tf.ones_like(inputs_mu)[..., None]
            J_to_concat = [] if self.fixed_inputs else [Jin]
            coeffs_pad = self.pad_coeffs(coeffs)
        with tf.name_scope('per_pix_recon_J'):
            if not self.fixed_params:
                coeff_patches = patchify_image(coeffs_pad,
                                               ksize_x=self.receptive_field[0],
                                               stride=self.stride)[..., None, :, :]

                # Set coeffs in padding region to be zero.
                # This means corr. filter elements in Jacobian are 0 and
                # ensures outgoing message from partially padded regions are not
                # impacted by coeff/filter messages incoming to padded vars
                coeff_patches *= self.get_mask_patches(coeffs.shape)
                Jfilt = conv_grad[..., None, None] * coeff_patches
                Jfilt = tf.reshape(Jfilt, Jfilt.shape.as_list()[:-2] + [-1])
                J_to_concat += [Jfilt]
            if not self.fixed_coeffs:
                filters_flat_unsqueeze = filters_flat[None, None, None]
                Jcoeff = conv_grad[..., None, None] * filters_flat_unsqueeze
                Jcoeff = tf.reshape(Jcoeff, Jcoeff.shape.as_list()[:-2] + [-1])

                # Set filter elems in padding region to be zero.
                # This means corr. coeff elements in Jacobian are 0 and
                # ensures outgoing message from partially padded regions are not
                # impacted by coeff/filter messages incoming to padded vars
                Jcoeff *= self.get_mask_patches(coeffs.shape, collapse_last_dim=True)
                J_to_concat += [Jcoeff]
            if not self.fixed_params and self.use_bias:
                Jbias = conv_grad[..., None]
                J_to_concat += [Jbias]
            J = tf.concat(J_to_concat, axis=-1)
        varstack0 = self.stack_vars(filters=filters, inputs=inputs_mu, coeffs=coeffs, bias=bias)
        with tf.name_scope('per_pix_recon_eta'):
            JTx0 = tf.reduce_sum(J * varstack0, -1)
            h0 = self.nonlin(conv_pre_act) - inputs_mu
            eta = J * (JTx0 - h0)[..., None] / self.sigma ** 2.

        if self.N_rob is not None:
            with tf.name_scope('per_pix_recon_eta_J_rob_corr'):
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
            self.marginalise_sherman_morrison(var_msg_in_eta,
                                              factor_eta,
                                              var_msg_in_Lambda,
                                              factor_J / self.sigma,)
        with tf.name_scope('per_pix_recon_outmsg_format'):
            self.update_outgoing_edge_messages(factor_to_var_eta, factor_to_var_Lambda)

    def update_outgoing_edge_messages(self, factor_to_var_eta, factor_to_var_Lambda):
        ks2 = self.filter_var_edges.shape[-1]
        nout_chan = self.filter_var_edges.shape[-2]
        ninps = int(not self.fixed_inputs) * 1  # Number of input vars connected to factor
        nfilt = int(not self.fixed_params) * nout_chan * ks2  # Number of filter variables (used to reconstruct 1 pixel)
        ncoeff = int(not self.fixed_coeffs) * nout_chan * ks2
        nbias = int(self.use_bias and not self.fixed_params)

        def reshape(x):
            # Un-collapse the output channel and patch dims
            return tf.reshape(x, x.shape.as_list()[:-1] + [nout_chan, ks2])

        if not self.fixed_inputs:
            self.input_var_edges.fac_to_var_eta = factor_to_var_eta[..., 0]
            self.input_var_edges.fac_to_var_Lambda = factor_to_var_Lambda[..., 0]

        if not self.fixed_params:
            self.filter_var_edges.fac_to_var_eta = reshape(factor_to_var_eta[..., ninps:ninps + nfilt])
            self.filter_var_edges.fac_to_var_Lambda = reshape(factor_to_var_Lambda[..., ninps:ninps + nfilt])
            if self.use_bias:
                self.bias_var_edges.fac_to_var_eta = factor_to_var_eta[..., -1]
                self.bias_var_edges.fac_to_var_Lambda = factor_to_var_Lambda[..., -1]
        if not self.fixed_coeffs:
            end = -1 if self.use_bias and not self.fixed_params else factor_to_var_eta.shape[-1]
            self.coeff_var_edges.fac_to_var_eta = reshape(factor_to_var_eta[..., -(nbias + ncoeff):end])
            self.coeff_var_edges.fac_to_var_Lambda = reshape(factor_to_var_Lambda[..., -(nbias + ncoeff):end])

    def forward_deterministic(self, coeffs=None):
        filters, _, coeffs_var = self.var0[:3]
        if self.use_bias:
            bias = self.var0[3]
        else:
            bias = 0.
        coeffs = coeffs if coeffs is not None else coeffs_var
        return self.nonlin(self.conv2dtranspose(coeffs, filters) + bias)
