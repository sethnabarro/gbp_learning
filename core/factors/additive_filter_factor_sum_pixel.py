# coding=utf-8
import numpy as np
import tensorflow as tf

from core.factors.additive_filter_factor_sum import AdditiveFilterFactorSum
from core.utils.utils import flatten_filters, patchify_image


class AdditiveFilterFactorSumPixel(AdditiveFilterFactorSum):
    N_PARALLEL_PROC_MARG = 1

    def __init__(self,
                 sigma,
                 n_filters,
                 input_var_edges,
                 filter_var_edges=None,
                 coeff_var_edges=None,
                 bias_var_edges=None,
                 N_rob=None,
                 rob_type='tukey',
                 init_lin_point=None,
                 relin_freq=1,
                 relative_to_centre=True,
                 kmult=1.,
                 dynamic_robust_mixture_weight=False,
                 pass_n_low_energy_filter_messages=False,
                 compute_low_energy_filter_message_only=False,
                 rec_field=(3, 3),
                 stride=1,
                 component_var_edges=None,
                 multidim_filter_vars=False,
                 use_component_vars=None,
                 nonlin=None,
                 nonlin_yscale=None,
                 nonlin_xscale=None,
                 depatch_fn=None):
        assert not pass_n_low_energy_filter_messages, "not yet supported"
        assert not multidim_filter_vars, "not yet supported"
        self.n_filt = n_filters
        super(AdditiveFilterFactorSum, self).__init__(
              sigma=sigma,
              init_lin_point=init_lin_point,
              relin_freq=relin_freq,
              N_rob=N_rob,
              rob_type=rob_type,
              kmult=kmult)
        self.stride = stride
        self.receptive_field = rec_field
        self.multidim_filter_vars = multidim_filter_vars
        self.input_var_edges = input_var_edges
        self.filter_var_edges = filter_var_edges
        self.coeff_var_edges = coeff_var_edges
        self.bias_var_edges = bias_var_edges
        self.component_var_edges = component_var_edges
        self.use_component_vars = use_component_vars or (component_var_edges is not None)
        self.use_learnt_bias = bias_var_edges is not None
        self.depatch_fn = depatch_fn

        self.nonlin, self.nonlin_grad = \
            self.build_nonlin_and_grad(nonlin, nonlin_xscale, nonlin_yscale)

        self._make_static_tensors()

        assert relative_to_centre
        self.relative_to_centre = True
        self.fix_attr_fields()

    def _make_static_tensors(self):
        if self.use_component_vars:
            components, inputs_mu = self.var0
            nfilt, fsize2 = components.shape[-2:]
            componentsT = tf.transpose(components, [0, 1, 2, 3, 5, 4])
            Jinp = tf.convert_to_tensor([1., -1.])
            Jinp = tf.broadcast_to(Jinp[None, None, None], componentsT.shape.as_list()[:-1] + [2])
            Jcomp = -tf.ones(componentsT.shape)
            self.J = tf.concat([Jinp, Jcomp], axis=-1)
        else:
            filters, inputs_mu, coeffs = self.var0[:3]
            filters_flat = flatten_filters(filters)
            nchannelin, nfilt, fsize2 = filters_flat.shape

            Jinp = tf.convert_to_tensor([1.] * nchannelin)
            self.Jinp = tf.broadcast_to(Jinp[None, None, None],
                                   inputs_mu.shape.as_list()[:-1] + [nchannelin])

        self.coeff_mask = self._make_mask(inputs_mu.shape, fsize2, dtype=inputs_mu.dtype, filt_or_coeff='coeff')
        self.filter_mask = self._make_mask(inputs_mu.shape, fsize2, dtype=inputs_mu.dtype, filt_or_coeff='filter')

    def _make_mask(self, inshp, ksize2, dtype=tf.float32, filt_or_coeff=None):
        filt_or_coeff = filt_or_coeff or 'filter'
        ksize = int(np.sqrt(ksize2))
        mask = np.ones(list(inshp) + [ksize2])
        recfield_iter = range(ksize)[::-1] #if filt_or_coeff == 'coeff' else range(ksize)
        for x in range(inshp[1]):
            for y in range(inshp[2]):
                if ksize <= x < inshp[1] - ksize - 1 and ksize <= y < inshp[1] - ksize - 1:
                    # Middle of image, away from borders
                    continue

                # Mask LH and top
                mask[:, x, y] *= [float((x >= kx) and (y >= ky)) for kx in recfield_iter
                                  for ky in recfield_iter]
                # Mask RH and bottom
                mask[:, x, y] *= [float((x <= inshp[1] - kx - 1) and (y <= inshp[2] - ky - 1))
                                  for kx in recfield_iter[::-1] for ky in recfield_iter[::-1]]
        return tf.convert_to_tensor(mask[..., None], dtype=dtype)

    def _get_incoming_messages(self):
        edges = (('inputs', self.input_var_edges),)
        if self.use_learnt_bias:
            edges += (('biases', self.bias_var_edges),)
        if self.use_component_vars:
            edges += (('components', self.component_var_edges),)
        else:
            edges += (('filters', self.filter_var_edges),
                      ('coeffs', self.coeff_var_edges))
        msgs_combined = []
        for mtype in ('eta', 'Lambda'):
            msgs = []
            for en, e in edges:
                msg_in = getattr(e, f'var_to_fac_{mtype}')
                if en in ('filters', 'biases', 'coeffs', 'components'):
                    s = msg_in.shape.as_list()
                    msg_in = tf.reshape(msg_in, s[:-2] + [s[-2] * s[-1]])
                if en in ('inputs',):
                    msg_in = msg_in[..., None]
                msgs.append(msg_in)
            msg_combined = tf.concat(msgs, axis=-1)
            msgs_combined.append(msg_combined)
        return msgs_combined

    # def energy(self, conn_vars, robust=None, aggregate=True):
    #     filters, pixels, weights = conn_vars
    #     fs = filters.shape[0]
    #     filters_flat = flatten_filters(filters)
    #     weightsTfilters = tf.linalg.einsum('abcd,de->abce', weights, filters_flat)
    #
    #     x_patch = patchify_image(image=pixels,
    #                              ksize_y=self.receptive_field[0],
    #                              ksize_x=self.receptive_field[1],
    #                              stride=self.stride)
    #     diff = x_patch - x_patch[..., int(fs ** 2 / 2)][..., None] - weightsTfilters
    #     E = tf.reduce_sum(diff ** 2 / self.sigma ** 2, axis=-1)
    #     if robust is None:
    #         robust = self.N_rob is not None
    #     if robust and self.N_rob is not None:
    #         E = self._robust_correct_energy(E)
    #     E = E[..., None]
    #
    #     if aggregate:
    #         return tf.reduce_sum(E)
    #     else:
    #         return E
    def get_eta_J(self, conn_vars):
        if self.use_component_vars:
            return self._get_eta_J_components(conn_vars)
        else:
            return self._get_eta_J_filter_coeffs(conn_vars)

    def _get_eta_J_components(self, conn_vars):
        components, inputs_mu = self.var0
        nfilt, fsize = components.shape[-2:]
        N_rob = self.N_rob

        componentsT = tf.transpose(components, [0, 1, 2, 3, 5, 4])

        inp_patch = patchify_image(inputs_mu, ksize_x=self.receptive_field[0], stride=self.stride)
        inp_patch_stack = tf.concat([inp_patch[..., None], tf.repeat(inp_patch[..., int(fsize / 2)][..., None], fsize, axis=-1)[..., None]], axis=-1)
        varstack0 = tf.concat([inp_patch_stack, componentsT], axis=-1)
        JTx0 = tf.einsum('abcdef,abcdef->abcde', self.J, varstack0, optimize='optimal')

        h0 = inp_patch - inp_patch[..., int(fsize / 2)][..., None] - self.nonlin(tf.reduce_sum(componentsT, axis=-1))
        eta = self.J * (JTx0 - h0)[..., None] / self.sigma ** 2.
        J = self.J
        if self.N_rob is not None:
            E = self.energy(self.var0,
                            robust=False,
                            aggregate=False,
                            sum_over_nhood=False,
                            input_patches=inp_patch)
            k = self.get_k_robust(E, N_rob=N_rob)
            eta *= k
            J *= tf.sqrt(k)

        return eta, J / self.sigma

    def _get_eta_J_filter_coeffs(self, conn_vars):

        filters, inputs_mu, coeffs = self.var0[:3]   # Covers case both with and without bias
        if self.use_learnt_bias:
            bias = self.var0[3]
        else:
            bias = None
        filters_flat = flatten_filters(filters)
        nchannelin, nfilt, fsize = filters_flat.shape
        ksize = int(np.sqrt(fsize))
        wpad = ksize // 2 * 2
        N_rob = self.N_rob
        E = self.energy(self.var0, robust=False, aggregate=False, sum_over_nhood=False)

        filters_flat_unsqueeze = tf.transpose(filters_flat, (0, 2, 1))[None, None, None]
        coeffs_pad = tf.pad(coeffs, [[0, 0], [wpad, wpad], [wpad, wpad], [0, 0]])
        coeffs_pad_patch = patchify_image(coeffs_pad, ksize_x=ksize, stride=self.stride)
        coeffs_pad_patch = tf.transpose(coeffs_pad_patch, (0, 1, 2, 4, 3))[..., None, :, :]
        Jfilt = tf.broadcast_to(-coeffs_pad_patch, inputs_mu.shape.as_list() + [ksize ** 2, nfilt])
        Jfilt = tf.reverse(Jfilt, axis=[-2]) * self.filter_mask  # To account for factor being from pixel perspective
        Jfilt = tf.reshape(tf.transpose(Jfilt, (0, 1, 2, 3, 5, 4)), inputs_mu.shape.as_list() + [ksize ** 2 * nfilt])
        Jcoeff = tf.broadcast_to(-filters_flat_unsqueeze, inputs_mu.shape.as_list() + [ksize ** 2, nfilt])
        Jcoeff = tf.reverse(Jcoeff, axis=[-2])  # To account for factor being from pixel perspective

        # Zero out coeff jacobian from outside image border
        Jcoeff *= self.coeff_mask
        Jcoeff = tf.reshape(tf.transpose(Jcoeff, (0, 1, 2, 3, 5, 4)), inputs_mu.shape.as_list() + [ksize ** 2 * nfilt])

        coeffsTfilters = coeffs[..., None, :, None] * filters_flat[None, None, None]
        coeffsTfilters = self.depatch_fn(tf.reduce_sum(coeffsTfilters, axis=-2))
        dh_dwTf = self.nonlin_grad(coeffsTfilters)[..., None]
        J = tf.concat([self.Jinp[..., None],  Jfilt * dh_dwTf, Jcoeff * dh_dwTf], axis=-1)

        # inp_patch = patchify_image(inputs_mu, ksize_x=self.receptive_field[0], stride=self.stride)
        # if self.use_learnt_bias:
        #     bias_patch = tf.repeat(bias[..., None], fsize, axis=-1)
        # else:
        #     bias_patch = tf.repeat(inp_patch[..., int(fsize / 2)][..., None], fsize, axis=-1)
        # inp_patch_stack = tf.concat([inp_patch[..., None], bias_patch[..., None]], axis=-1)
        imshp = self.Jinp.shape.as_list()
        filters_flat_stack = tf.reshape(tf.transpose(tf.broadcast_to(filters_flat_unsqueeze, imshp + [ksize ** 2, nfilt]), (0, 1,2,3,5,4)), imshp + [ksize ** 2 * nfilt])
        coeffs_stack = tf.reshape(tf.transpose(tf.broadcast_to(coeffs_pad_patch, imshp + [ksize ** 2, nfilt]), (0, 1,2 ,3,5,4)), imshp + [ksize ** 2 * nfilt])
        varstack0 = tf.concat([inputs_mu[..., None], filters_flat_stack, coeffs_stack], axis=-1)
        JTx0 = tf.reduce_sum(J * varstack0, -1) # tf.einsum('abcdef,abcdef->abcde', J, varstack0, optimize='optimal')
        # weightsTfilters = tf.reduce_sum(weights[..., None, :, None] * filters_flat[None, None, None], axis=-2) #tf.einsum('abcd,edf->abcef', weights, filters_flat, optimize='optimal')
        # h0 = inp_patch - self.nonlin(weightsTfilters + bias_patch)
        h0 = inputs_mu - self.nonlin(coeffsTfilters)
        eta = J * (JTx0 - h0)[..., None] / self.sigma ** 2.


        # Lambda = J[0, 2, 6, 0][None] * J[0, 2, 6, 0][:, None] / self.sigma ** 2.
        # print(tf.reduce_sum(tf.linalg.inv(Lambda) * eta[0, 2, 6, 0], axis=1))
        # print(weights_pad_patch[0, 2, 6], filters_flat)


        if self.N_rob is not None:
            E = self.energy(self.var0,
                            robust=False,
                            aggregate=False,
                            sum_over_nhood=True)
            k = self.get_k_robust(E, N_rob=N_rob)[..., None, :]
            eta *= k
            J *= tf.sqrt(k)

        return eta, J

    def energy(self, conn_vars, robust=None, aggregate=True, sum_over_nhood=True, input_patches=None):
        filters, inputs, coeffs = conn_vars[:3]
        # fs = filters.shape[0]
        filters_flat = flatten_filters(filters)
        coeffsTfilters = coeffs[..., None, :, None] * filters_flat[None, None, None]
        coeffsTfilters = self.depatch_fn(tf.reduce_sum(coeffsTfilters, axis=-2))
        # if input_patches is None:
        #     input_patches = patchify_image(image=inputs,
        #                                    ksize_y=self.receptive_field[0],
        #                                    ksize_x=self.receptive_field[1],
        #                                    stride=self.stride)
        #
        # if self.use_learnt_bias:
        #     bias_patches = conn_vars[3][..., None]
        # else:
        #     bias_patches = input_patches[..., int(fs ** 2 / 2)][..., None]
        diff = inputs - self.nonlin(coeffsTfilters)
        E = diff ** 2 / self.sigma ** 2

        if sum_over_nhood:
            E = tf.reduce_sum(E, axis=-1)
        if robust is None:
            robust = self.N_rob is not None
        if robust and self.N_rob is not None:
            E = self._robust_correct_energy(E)
        E = E[..., None]

        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E

    def update_outgoing_messages(self, conn_vars, **kwargs):
        var_msg_in_eta, var_msg_in_Lambda = self._get_incoming_messages()
        factor_eta, factor_J = self.get_eta_J(conn_vars)

        # factor_plus_mess_eta = var_msg_in_eta + factor_eta
        # factor_Lambda = factor_J[..., None, :] * factor_J[..., None]
        # factor_plus_mess_Lambda = factor_Lambda + tf.linalg.diag(var_msg_in_Lambda)
        # factor_to_var_eta, factor_to_var_Lambda = self.marginalise(factor_plus_mess_eta, factor_plus_mess_Lambda, factor_eta, factor_Lambda)

        tf.debugging.check_numerics(var_msg_in_eta, f'ineta')
        tf.debugging.check_numerics(var_msg_in_Lambda, f'inLam')
        tf.debugging.check_numerics(factor_eta, f'feta')
        tf.debugging.check_numerics(factor_J, f'fJ')
        mask = tf.concat([tf.ones_like(self.filter_mask[..., :1, 0]),
                          tf.concat([self.filter_mask[..., 0]] * self.n_filt, axis=-1),
                          tf.concat([self.coeff_mask[..., 0]] * self.n_filt, axis=-1)], axis=-1)
        factor_Lambda = factor_J[..., None] * factor_J[..., None, :] / self.sigma ** 2.
        # print(var_msg_in_Lambda[0, 0, 0], factor_Lambda[0, 0, 0], mask[0, 0, 0], "ED")

        # Marginalisation
        factor_to_var_eta, factor_to_var_Lambda = \
            self.marginalise_sherman_morrison(var_msg_in_eta,
                                              factor_eta,
                                              var_msg_in_Lambda,
                                              factor_J,
                                              mask=mask)

        tf.debugging.check_numerics(factor_to_var_eta, f'outeta')
        tf.debugging.check_numerics(factor_to_var_Lambda, f'outLambda')
        # print(np.where(factor_to_var_Lambda < 0.))
        # tf.assert_greater(factor_to_var_Lambda, tf.ones_like(factor_to_var_Lambda) * -1e-3)
        # asdf
        self.update_outgoing_edge_messages(factor_to_var_eta, factor_to_var_Lambda)

    def update_outgoing_edge_messages(self, factor_to_var_eta, factor_to_var_Lambda):
        ninps = 1  # Number of input vars connected to factor (inc learnt bias if using)
        nfiltvars = self.n_filt * self.filter_var_edges.fac_to_var_eta.shape[-1]
        ksize2 = self.filter_var_edges.fac_to_var_eta.shape[-1]
        # factor_to_var_eta = tf.where(tf.math.is_nan(factor_to_var_eta), 0., factor_to_var_eta)
        # factor_to_var_Lambda = tf.where(tf.math.is_nan(factor_to_var_Lambda), 0., factor_to_var_Lambda)
        self.input_var_edges.fac_to_var_eta = factor_to_var_eta[..., 0]
        self.input_var_edges.fac_to_var_Lambda = factor_to_var_Lambda[..., 0]

        if self.use_component_vars:
            self.component_var_edges.fac_to_var_eta = tf.transpose(factor_to_var_eta[..., ninps:],
                                                                   (0, 1, 2, 3, 5, 4))
            self.component_var_edges.fac_to_var_Lambda = tf.transpose(factor_to_var_Lambda[..., ninps:],
                                                                      (0, 1, 2, 3, 5, 4))
        else:
            def rstp(x):
                return tf.transpose(tf.reshape(x, x.shape.as_list()[:-1] + [ksize2, self.n_filt]), (0, 1, 2, 3, 5, 4))
            self.filter_var_edges.fac_to_var_eta = rstp(factor_to_var_eta[..., ninps:ninps + nfiltvars])
            self.coeff_var_edges.fac_to_var_eta = rstp(factor_to_var_eta[..., ninps + nfiltvars:])

            self.filter_var_edges.fac_to_var_Lambda = rstp(factor_to_var_Lambda[..., ninps:ninps + nfiltvars])
            self.coeff_var_edges.fac_to_var_Lambda = rstp(factor_to_var_Lambda[..., ninps + nfiltvars:])
