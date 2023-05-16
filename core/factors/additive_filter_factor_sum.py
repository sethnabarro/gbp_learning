# coding=utf-8
import tensorflow as tf

from core.factors.base_nonlin import NonLinearFactor
from core.utils.utils import flatten_filters, patchify_image


class AdditiveFilterFactorSum(NonLinearFactor):
    def __init__(self,
                 sigma,
                 input_var_edges,
                 n_filters,
                 filter_var_edges=None,
                 coeff_var_edges=None,
                 component_var_edges=None,
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
                 multidim_filter_vars=False,
                 use_component_vars=None,
                 nonlin=None,
                 nonlin_yscale=1.,
                 nonlin_xscale=1.,
                 coeff_padding=None
                 ):
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

        assert relative_to_centre
        self.relative_to_centre = True
        self.fix_attr_fields()

    def _get_incoming_messages(self):

        def _collapse_last_dim(x):
            shp = x.shape.as_list()
            new_shp = shp[:-2] + [shp[-2] * shp[-1]]
            return tf.reshape(x, new_shp)

        msgs_combined = []
        edges = (self.input_var_edges, self.filter_var_edges, self.coeff_var_edges)
        for mtype in ('eta', 'Lambda'):
            msgs = []
            for e in edges:
                msg_in = getattr(e, f'var_to_fac_{mtype}')
                msgs.append(msg_in)
            msg_pc = [msgs[0],
                      _collapse_last_dim(msgs[1]),
                      msgs[2][..., 0]]
            msg_combined = tf.concat(msg_pc, axis=-1)
            msgs_combined.append(msg_combined)
        return msgs_combined

    def _energy_filter_coeffs(self, conn_vars, robust=None, aggregate=True, sum_over_nhood=True, input_patches=None):
        filters, inputs, coeffs = conn_vars[:3]

        if coeffs.shape[1] * self.stride == inputs.shape[1]:
            coeffs = self.trim_coeffs(coeffs)
        fs = filters.shape[0]
        filters_flat = flatten_filters(filters)
        coeffsTfilters = tf.linalg.einsum('abcd,edf->abcef', coeffs, filters_flat, optimize='optimal')
        if input_patches is None:
            input_patches = patchify_image(image=inputs,
                                           ksize_y=self.receptive_field[0],
                                           ksize_x=self.receptive_field[1],
                                           stride=self.stride)

        bias_patches = conn_vars[3][None, None, None, :, None] if self.use_bias else 0.
        centpix_patches = input_patches[..., int(fs ** 2 / 2)][..., None] if self.relative_to_centre else 0.
        diff = input_patches - centpix_patches - self.nonlin(coeffsTfilters + bias_patches)
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

    def _energy_components(self, conn_vars, robust=None, aggregate=True, sum_over_nhood=True, input_patches=None):
        components, inputs = conn_vars
        fs = self.receptive_field[0] * self.receptive_field[1]
        if input_patches is None:
            input_patches = patchify_image(image=inputs,
                                           ksize_y=self.receptive_field[0],
                                           ksize_x=self.receptive_field[1],
                                           stride=self.stride)
        diff = input_patches - input_patches[..., int(fs / 2)][..., None] - self.nonlin(tf.reduce_sum(components, axis=-2))
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

    def energy(self, *args, **kwargs):
        if self.use_component_vars:
            return self._energy_components(*args, **kwargs)
        else:
            return self._energy_filter_coeffs(*args, **kwargs)

    def get_eta_Lambda(self, conn_vars):
        filters, inputs_mu, coeffs = self.var0
        filters_flat = flatten_filters(filters)
        ks2 = filters_flat.shape[-1]
        N_rob = self.N_rob
        E = self.energy(self.var0, robust=False, aggregate=False)
        if N_rob is not None:
            k = self.get_k_robust(E, N_rob=N_rob)
        else:
            k = 1.
        fsize = self.receptive_field[0] * self.receptive_field[1]
        Jinps = tf.eye(fsize)
        u = tf.ones((fsize,))
        Jinps = tf.tensor_scatter_nd_sub(Jinps, [[int(fsize / 2), r] for r in range(fsize)], u)
        Jfilt = -tf.linalg.diag(tf.broadcast_to(coeffs[..., None], coeffs.shape.as_list() + [fsize]))
        Jfilt = tf.concat(tf.unstack(Jfilt, axis=-3), axis=-2)
        Jinps = tf.broadcast_to(Jinps[None, None, None], Jfilt.shape.as_list()[:-2] + [fsize, fsize])
        coeffsTfilters = tf.linalg.einsum('abcd,de->abce', coeffs, filters_flat)
        dh_dwTf = self.nonlin_grad(coeffsTfilters)[..., None]
        Jw = -flatten_filters(filters)
        Jw = tf.broadcast_to(Jw[None, None, None], Jfilt.shape.as_list()[:3] + Jw.shape.as_list())
        J = tf.concat([Jinps, Jfilt * dh_dwTf, Jw * dh_dwTf], axis=-2)

        inputs_mu_patch = patchify_image(inputs_mu,
                                         ksize_x=self.receptive_field[0],
                                         stride=self.stride)
        filters_vec = tf.reshape(flatten_filters(filters), (-1))
        filters_flat_rs = tf.broadcast_to(filters_vec[None, None, None],
                                          inputs_mu_patch.shape.as_list()[:-1] + filters_vec.shape.as_list())
        var0stack = tf.concat([inputs_mu_patch, filters_flat_rs, coeffs], axis=-1)

        Jx = tf.einsum('abcde,abcd->abce', J, var0stack)

        h = inputs_mu_patch - inputs_mu_patch[..., int(fsize / 2)][..., None] - self.nonlin(coeffsTfilters)

        eta = tf.linalg.einsum('abcde,abce->abcd', J, Jx - h) / self.sigma ** 2.
        Lambda = tf.linalg.einsum('abcde,abcfe->abcdf', J, J) / self.sigma ** 2.

        if self.N_rob is not None:
            eta *= k
            Lambda *= k[..., None]

        return eta, Lambda

    def update_outgoing_messages(self, conn_vars, **kwargs):
        factor_eta, factor_Lambda = self.get_eta_Lambda(conn_vars)

        original_shp = self.input_var_edges.var_to_fac_eta.shape
        var_msg_in_eta, var_msg_in_Lambda = self._get_incoming_messages()

        factor_plus_mess_eta = var_msg_in_eta + factor_eta

        factor_plus_mess_Lambda = factor_Lambda + tf.linalg.diag(var_msg_in_Lambda)

        # Marginalisation
        factor_to_var_eta, factor_to_var_Lambda = \
            self.marginalise(factor_plus_mess_eta,
                             factor_plus_mess_Lambda,
                             factor_eta, factor_Lambda)

        def _add_filter_dim(x):
            shp = x.shape.as_list()
            assert (shp[-1] / self.n_filt).is_integer()
            new_shp = shp[:-1] + [self.n_filt, int(shp[-1] / self.n_filt)]
            return tf.reshape(x, new_shp)

        ninps = self.receptive_field[0] * self.receptive_field[1]
        nfilt = self.receptive_field[0] * self.receptive_field[1] * self.n_filt

        self.input_var_edges.fac_to_var_eta = factor_to_var_eta[..., :ninps]
        self.filter_var_edges.fac_to_var_eta = _add_filter_dim(factor_to_var_eta[..., ninps:ninps + nfilt])
        self.coeff_var_edges.fac_to_var_eta = factor_to_var_eta[..., ninps + nfilt:]

        self.input_var_edges.fac_to_var_Lambda = factor_to_var_Lambda[..., :ninps]
        self.filter_var_edges.fac_to_var_Lambda = _add_filter_dim(factor_to_var_Lambda[..., ninps:ninps + nfilt])
        self.coeff_var_edges.fac_to_var_Lambda = factor_to_var_Lambda[..., ninps + nfilt:]

    def get_edge_messages(self, named=False):
        attr_to_get = 'named_state' if named else 'state'
        enames = []
        if self.use_component_vars:
            edges = [getattr(self.input_var_edges, attr_to_get),
                     getattr(self.component_var_edges, attr_to_get)]
            enames += [str(self.input_var_edges), str(self.component_var_edges)]
        else:
            edges = [getattr(self.input_var_edges, attr_to_get),
                     getattr(self.filter_var_edges, attr_to_get),
                     getattr(self.coeff_var_edges, attr_to_get)]
            enames += [str(self.input_var_edges),
                       str(self.filter_var_edges),
                       str(self.coeff_var_edges)]
        if self.use_bias:
            edges.append(getattr(self.bias_var_edges, attr_to_get))
            enames.append(str(self.bias_var_edges))
        if named:
            edges = list(zip(enames, edges))
        return edges

    def set_edge_messages(self, edges):
        if self.use_component_vars:
            self.input_var_edges.state, self.component_var_edges.state = edges
        elif self.use_bias:
            self.input_var_edges.state,\
                self.filter_var_edges.state, \
                self.coeff_var_edges.state, \
                self.bias_var_edges.state = edges
        else:
            self.input_var_edges.state, \
                self.filter_var_edges.state, \
                self.coeff_var_edges.state = edges
        return edges

    def trim_coeffs(self, coeffs):
        if self.coeff_padding > 0:
            coeffs = coeffs[:, self.coeff_padding:-self.coeff_padding, self.coeff_padding:-self.coeff_padding]
        return coeffs

    def relinearise(self, itr, current_vars):
        current_vars[2] = self.trim_coeffs(current_vars[2])   # [2] is for coeff vars
        super().relinearise(itr, current_vars)
