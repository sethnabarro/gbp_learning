# coding=utf-8
import tensorflow as tf

from core.factors.additive_filter_factor_sum import AdditiveFilterFactorSum
from core.utils.utils import flatten_filters, patchify_image


class AdditiveFilterFactorSumDecomp(AdditiveFilterFactorSum):
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
                 sum_before_nonlin=True,
                 fac_to_var_chunksize=1,
                 coeff_padding=None):
        assert not pass_n_low_energy_filter_messages, "not yet supported"
        assert not multidim_filter_vars, "not yet supported"
        self.n_filt = n_filters
        self.coeff_padding = coeff_padding or 0
        init_lin_point[2] = self.trim_coeffs(init_lin_point[2])
        super(AdditiveFilterFactorSum, self).__init__(
              sigma=sigma,
              init_lin_point=init_lin_point,
              relin_freq=relin_freq,
              N_rob=N_rob,
              rob_type=rob_type,
              kmult=kmult)
        self.fac_to_var_chunksize = fac_to_var_chunksize
        self.stride = stride
        self.receptive_field = rec_field
        self.multidim_filter_vars = multidim_filter_vars
        self.input_var_edges = input_var_edges
        self.filter_var_edges = filter_var_edges
        self.coeff_var_edges = coeff_var_edges
        self.bias_var_edges = bias_var_edges
        self.component_var_edges = component_var_edges
        self.use_component_vars = use_component_vars or (component_var_edges is not None)
        self.use_bias = bias_var_edges is not None

        self.nonlin, self.nonlin_grad = \
            self.build_nonlin_and_grad(nonlin, nonlin_xscale, nonlin_yscale)
        self.sum_before_nonlin = sum_before_nonlin
        self.relative_to_centre = relative_to_centre
        self.fixed_first_n_channels = 0
        self.coeff_padding = coeff_padding or 0

        self._make_static_tensors()
        self.fixed_params = False
        self.fixed_coeffs = False
        self.fixed_inputs = False

        self.fix_attr_fields()

    def _make_static_tensors(self):
        if self.use_component_vars:
            components, inputs_mu = self.var0
            componentsT = tf.transpose(components, [0, 1, 2, 3, 5, 4])
            Jinp = tf.convert_to_tensor([1., -1.])
            Jinp = tf.broadcast_to(Jinp[None, None, None], componentsT.shape.as_list()[:-1] + [2])
            Jcomp = -tf.ones(componentsT.shape)
            self.J = tf.concat([Jinp, Jcomp], axis=-1)
        else:
            filters, inputs_mu, coeffs = self.var0[:3]
            filters_flat = flatten_filters(filters)
            nchannelin, nfilt, fsize = filters_flat.shape

            Jinp = tf.convert_to_tensor([[1., -1.] if self.relative_to_centre else [1.]] * nchannelin)
            self.Jinp = tf.broadcast_to(Jinp[None, None, None, :, None],
                                        coeffs.shape.as_list()[:-1] + [nchannelin, fsize, 1 + int(self.relative_to_centre)])

    def _get_incoming_messages(self):
        edges = (('inputs', self.input_var_edges),) if not self.fixed_inputs else ()
        if self.use_component_vars:
            edges += (('components', self.component_var_edges),)
        elif not self.fixed_params:
            edges += (('filters', self.filter_var_edges),)
        if not self.fixed_coeffs:
            edges += (('coeffs', self.coeff_var_edges),)
        if self.use_bias and not self.fixed_params:
            edges += (('biases', self.bias_var_edges),)
        msgs_combined = []
        for mtype in ('eta', 'Lambda'):
            msgs = []
            for en, e in edges:
                msg_in = getattr(e, f'var_to_fac_{mtype}')
                if en in ('filters', 'coeffs', 'components'):
                    msg_in = tf.transpose(msg_in, (0, 1, 2, 3, 5, 4))
                elif en == 'biases':
                    msg_in = msg_in[..., None]
                msgs.append(msg_in)
                # print(mtype, msg_in[0,0,0], en)
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
    def stack_vars(self, filters, inputs, coeffs, bias=None):
        fshp = filters.shape
        fsize = fshp[0] * fshp[1]
        nfilt = fshp[-1]

        inp_patch = patchify_image(inputs, ksize_x=self.receptive_field[0], stride=self.stride)
        inp_patch_stack = inp_patch[..., None]
        if self.relative_to_centre:
            centpix_patch = tf.repeat(inp_patch[..., int(fsize / 2)][..., None], fsize, axis=-1)
            inp_patch_stack = tf.concat([inp_patch_stack, centpix_patch[..., None]], axis=-1)

        coeff_shp = inp_patch_stack.shape[:-1].as_list() + [nfilt]
        coeffs_stack = tf.broadcast_to(coeffs[..., None, None, :], coeff_shp)
        to_stack = [] if self.fixed_inputs else [inp_patch_stack]
        # if self.fixed_params:
        #     varstack = tf.concat([inp_patch_stack, coeffs_stack], axis=-1)
        # else:
        if not self.fixed_params:
            filters_flat = flatten_filters(filters)
            filters_flat_unsqueeze = tf.transpose(filters_flat, (0, 2, 1))[None, None, None]
            filters_flat_stack = tf.broadcast_to(filters_flat_unsqueeze, inp_patch_stack.shape[:-1].as_list() + [nfilt])
            to_stack += [filters_flat_stack]
        if not self.fixed_coeffs:
            to_stack += [coeffs_stack]
            # varstack = tf.concat(to_concat, axis=-1)
        if self.use_bias and not self.fixed_params:
            bias_patch = tf.broadcast_to(bias[None, None, None, :, None], inp_patch_stack.shape[:-1])[..., None]
            to_stack += [bias_patch]
        return tf.concat(to_stack, axis=-1)

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
        if self.use_bias:
            bias = self.var0[3]
        else:
            bias = None
        filters_flat = flatten_filters(filters)
        nchannelin, nfilt, fsize = filters_flat.shape
        N_rob = self.N_rob
        E = self.energy(self.var0, robust=False, aggregate=False, sum_over_nhood=False)
        if N_rob is not None:
            k = self.get_k_robust(E, N_rob=N_rob)
        else:
            k = 1.
        filters_flat_unsqueeze = tf.transpose(filters_flat, (0, 2, 1))[None, None, None]

        Jcoeff = tf.broadcast_to(-filters_flat_unsqueeze, self.Jinp.shape.as_list()[:-1] + [nfilt])
        coeffsTfilters = coeffs[..., None, :, None] * filters_flat[None, None, None]
        if self.sum_before_nonlin:
            coeffsTfilters = tf.reduce_sum(coeffsTfilters, axis=-2)
            if self.use_bias:
                coeffsTfilters += bias[None, None, None, :, None]
            dh_dwTf = self.nonlin_grad(coeffsTfilters)[..., None]
        else:
            coeffsTfilters = tf.transpose(coeffsTfilters, (0, 1, 2, 3, 5, 4))
            dh_dwTf = self.nonlin_grad(coeffsTfilters)
        Jcoeff = Jcoeff * dh_dwTf
        J_to_concat = [self.Jinp] if not self.fixed_inputs else []
        # if self.fixed_params:
        #     J = self.Jinp if self.fixed_coeffs else tf.concat([self.Jinp, Jcoeff], axis=-1)
        # else:
        if not self.fixed_params:
            Jfilt = tf.broadcast_to(-coeffs[..., None, None, :], self.Jinp.shape.as_list()[:-1] + [nfilt]) * dh_dwTf
            J_to_concat += [Jfilt]
        if not self.fixed_coeffs:
            J_to_concat += [Jcoeff]
            # J = tf.concat([self.Jinp,  Jfilt] + ([] if self.fixed_coeffs else [Jcoeff]), axis=-1)
        if self.use_bias and not self.fixed_params:
            J_bias = -dh_dwTf[..., 0] * tf.ones_like(coeffsTfilters) if self.sum_before_nonlin else -tf.ones_like(coeffsTfilters[..., 0])
            J_to_concat += [J_bias[..., None]]
        J = tf.concat(J_to_concat, axis=-1)

        varstack0 = self.stack_vars(filters=filters, inputs=inputs_mu, coeffs=coeffs, bias=bias)
        inp_patch = patchify_image(inputs_mu, ksize_x=self.receptive_field[0], stride=self.stride)
        if self.relative_to_centre:
            centpix_patch = tf.repeat(inp_patch[..., int(fsize / 2)][..., None], fsize, axis=-1)
        else:
            centpix_patch = 0.
        JTx0 = tf.reduce_sum(J * varstack0, -1)  # tf.einsum('abcdef,abcdef->abcde', J, varstack0, optimize='optimal')
        if self.sum_before_nonlin:
            h0 = inp_patch - centpix_patch - self.nonlin(coeffsTfilters)
        else:
            h0 = inp_patch - centpix_patch - (tf.reduce_sum(self.nonlin(coeffsTfilters), axis=-1) + bias[None, None, None, :, None])

        eta = J * (JTx0 - h0)[..., None] / self.sigma ** 2.

        if self.N_rob is not None:
            E = self.energy(self.var0,
                            robust=False,
                            aggregate=False,
                            sum_over_nhood=True,
                            input_patches=inp_patch)
            k = self.get_k_robust(E, N_rob=N_rob)[..., None, :]
            eta *= k
            J *= tf.sqrt(k)

        return eta, J

    def update_outgoing_messages(self, conn_vars, **kwargs):
        var_msg_in_eta, var_msg_in_Lambda = self._get_incoming_messages()

        factor_eta, factor_J = self.get_eta_J(conn_vars)

        if self.fixed_first_n_channels > 0:
            factor_eta = factor_eta[..., self.fixed_first_n_channels:, :, :]
            factor_J = factor_J[..., self.fixed_first_n_channels:, :, :]
            var_msg_in_eta = var_msg_in_eta[..., self.fixed_first_n_channels:, :, :]
            var_msg_in_Lambda = var_msg_in_Lambda[..., self.fixed_first_n_channels:, :, :]

        # Marginalisation
        factor_to_var_eta, factor_to_var_Lambda = \
            self.marginalise_sherman_morrison(var_msg_in_eta,
                                              factor_eta,
                                              var_msg_in_Lambda,
                                              factor_J / self.sigma,
                                              batchsize=self.fac_to_var_chunksize)

        self.update_outgoing_edge_messages(factor_to_var_eta, factor_to_var_Lambda)

    def update_outgoing_edge_messages(self, factor_to_var_eta, factor_to_var_Lambda):
        ninps = (1 + int(self.relative_to_centre)) * int(not self.fixed_inputs)  # Number of input vars connected to factor
        nfilt = int(not self.fixed_params) * self.n_filt  # Number of filter variables (used to reconstruct 1 pixel)
        ncoeff = int(not self.fixed_coeffs) * self.n_filt
        nchanin = self.filter_var_edges.shape[-3]

        if self.fixed_first_n_channels > 0 and not self.fixed_inputs:
            # Need to pad messages s.t. shape consistent with edge shape for all channels
            shp = factor_to_var_eta.shape.as_list()
            pad_shp = shp.copy()
            pad_shp[-3] = nchanin - shp[-3]
            factor_to_var_eta = tf.concat([tf.zeros(pad_shp), factor_to_var_eta], axis=-3)
            factor_to_var_Lambda = tf.concat([tf.zeros(pad_shp), factor_to_var_Lambda], axis=-3)

        def _update_last_channels(edge, new_msg_eta, new_msg_Lambda):
            cond = (tf.range(nchanin) >= self.fixed_first_n_channels)[:, None, None]
            if len(new_msg_Lambda.shape) == 5:
                cond = cond[..., 0]
            edge.fac_to_var_eta = tf.where(cond,
                                           new_msg_eta,
                                           edge.fac_to_var_eta)
            edge.fac_to_var_Lambda = tf.where(cond,
                                              new_msg_Lambda,
                                              edge.fac_to_var_Lambda)

        _update_last_channels(self.input_var_edges,
                              factor_to_var_eta[..., :ninps],
                              factor_to_var_Lambda[..., :ninps])

        if self.use_component_vars:
            self.component_var_edges.fac_to_var_eta = tf.transpose(factor_to_var_eta[..., ninps:],
                                                                   (0, 1, 2, 3, 5, 4))
            self.component_var_edges.fac_to_var_Lambda = tf.transpose(factor_to_var_Lambda[..., ninps:],
                                                                      (0, 1, 2, 3, 5, 4))
        else:
            if not self.fixed_params:
                _update_last_channels(self.filter_var_edges,
                                      tf.transpose(factor_to_var_eta[..., ninps:ninps + nfilt], (0, 1, 2, 3, 5, 4)),
                                      tf.transpose(factor_to_var_Lambda[..., ninps:ninps + nfilt], (0, 1, 2, 3, 5, 4)))
                if self.use_bias:
                    _update_last_channels(self.bias_var_edges,
                                          factor_to_var_eta[..., -1],
                                          factor_to_var_Lambda[..., -1])
            if not self.fixed_coeffs:
                self.coeff_var_edges.fac_to_var_eta = \
                    tf.transpose(factor_to_var_eta[..., ninps + nfilt:ninps + nfilt + ncoeff], (0, 1, 2, 3, 5, 4))
                self.coeff_var_edges.fac_to_var_Lambda = \
                    tf.transpose(factor_to_var_Lambda[..., ninps + nfilt:ninps + nfilt + ncoeff], (0, 1, 2, 3, 5, 4))

