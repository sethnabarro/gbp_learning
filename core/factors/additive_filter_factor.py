# coding=utf-8
import tensorflow as tf

from core.factors.base_nonlin import NonLinearFactor
from core.utils.utils import patchify_image, flatten_filters


class AdditiveFilterFactor(NonLinearFactor):
    N_PARALLEL_PROC_MARG = 1

    def __init__(self,
                 sigma,
                 input_var_edges,
                 filter_var_edges,
                 init_lin_point,
                 N_rob=None,
                 rob_type='tukey',
                 rec_field=(3, 3),
                 stride=1,
                 coeff_var_edges=None,
                 relative_to_centre=True,
                 relin_freq=1,
                 kmult=1.,
                 nonlin=None,
                 nonlin_xscale=None,
                 nonlin_yscale=None):
        self.relative_to_centre = relative_to_centre
        super(AdditiveFilterFactor, self).__init__(sigma, init_lin_point, relin_freq, N_rob, rob_type, kmult=kmult)
        self.receptive_field = rec_field
        self.use_filter_coeffs = coeff_var_edges is not None
        self._set_eta_Lambda()
        self.input_var_edges = input_var_edges
        self.filter_var_edges = filter_var_edges
        self.coeff_var_edges = coeff_var_edges
        self.lowest_energy_filter_ids = None
        self.stride = stride
        assert nonlin is None, "Nonlinearity only supported in weighted sum reconstruction factors"
        self.fix_attr_fields()

    @property
    def eta(self):
        return self._eta

    @property
    def Lambda(self):
        return self._Lambda

    def _set_eta_Lambda(self):
        fsize = self.receptive_field[0] * self.receptive_field[1]
        J = tf.eye(fsize)
        u = tf.ones((fsize,))
        if self.relative_to_centre:
            J = tf.tensor_scatter_nd_sub(J, [[int(fsize / 2), r] for r in range(fsize)], u)
        self.J = tf.concat((J, -tf.eye(fsize)), axis=0)
        if self.use_filter_coeffs:
            self.J = tf.concat((self.J, -tf.ones((1, fsize))), axis=0)
        self._eta = tf.zeros((fsize * 2 + int(self.use_filter_coeffs),)) / self.sigma ** 2.
        self._Lambda = tf.matmul(self.J, tf.transpose(self.J)) / self.sigma ** 2.

    def get_eta_Lambda(self, conn_vars, **kwargs):
        filters, inputs_mu = self.var0[0], self.var0[1]
        if self.use_filter_coeffs:
            coeffs = self.var0[2]
            fsize = self.receptive_field[0] * self.receptive_field[1]
            J = tf.eye(fsize)
            u = tf.ones((fsize,))
            if self.relative_to_centre:
                J = tf.tensor_scatter_nd_sub(J, [[int(fsize / 2), r] for r in range(fsize)], u)
            J = tf.broadcast_to(J[None, None, None, None], coeffs.shape.as_list() + J.shape.as_list())
            J = tf.concat((J, -tf.eye(fsize, batch_shape=coeffs.shape) * coeffs[..., None, None]), axis=-2)
            # J = tf.concat((J, flatten_filters(filters)))

            filters_flat = flatten_filters(filters)
            filters_flat_bc = tf.broadcast_to(filters_flat[None, None, None], J.shape.as_list()[:-2] + [fsize])
            J = tf.concat((J, -filters_flat_bc[..., None, :]), axis=-2)
            inputs_patch = patchify_image(inputs_mu, ksize_x=self.receptive_field[0], stride=self.stride)
            inputs_bc = tf.broadcast_to(inputs_patch[..., None, :], filters_flat_bc.shape)
            varstack = tf.concat((inputs_bc, filters_flat_bc, coeffs[..., None]), axis=-1)
            # JTx = tf.einsum('abcdef,abcde->abcdf', J, varstack)

            # h = pixels_bc - pixels_bc[..., int(fsize / 2)][..., None] - coeffs[..., None] * filters_flat_bc
            eta = tf.einsum('abcdef,abcdf->abcde', J, - coeffs[..., None] * filters_flat_bc) / self.sigma ** 2.

            Lambda = tf.einsum('abcdef,abcdgf->abcdeg', J, J) / self.sigma ** 2.
        else:
            eta, Lambda = self._eta, self._Lambda
        N_rob = self.N_rob

        E = self.energy(self.var0, robust=False, aggregate=False)
        if N_rob is not None:

            k = self.get_k_robust(E, N_rob=N_rob)
            eta = eta * k[..., None]
            Lambda = Lambda * k[..., None, None]

        elif not self.use_filter_coeffs:
            fsize, _, _, n_filt = filters.shape
            _, H, W, _ = inputs_mu.shape
            facsize = self._eta.shape[0]
            eta = tf.broadcast_to(eta[None, None, None, None], [1, H - fsize + 1, W - fsize + 1, n_filt, facsize])
            Lambda = tf.broadcast_to(Lambda[None, None, None, None], [1, H - fsize + 1, W - fsize + 1, n_filt, facsize, facsize])

        if 'return_k' in kwargs:
            if kwargs['return_k'] and N_rob is not None:
                return eta, Lambda, (k if self.N_rob is not None else 1.)
        return eta, Lambda

    def marginalise(self,
                    factor_plus_mess_eta,
                    factor_plus_mess_Lambda,
                    # var_msg_in_eta,
                    # var_msg_in_Lambda):
                    factor_eta,
                    factor_Lambda,
                    nvar_marg=None):
        fsize = self.receptive_field[0] * self.receptive_field[1]
        nvar_marg = fsize * 2 + int(self.use_filter_coeffs)
        factor_to_var_eta, factor_to_var_Lambda = \
            super(AdditiveFilterFactor, self).marginalise(factor_plus_mess_eta,
                                                          factor_plus_mess_Lambda,
                                                          factor_eta,
                                                          factor_Lambda,
                                                          nvar_to_marg=nvar_marg)

        factor_to_input_eta = factor_to_var_eta[..., :fsize]
        factor_to_filter_eta = factor_to_var_eta[..., fsize:2 * fsize]
        factor_to_input_Lambda = factor_to_var_Lambda[..., :fsize]
        factor_to_filter_Lambda = factor_to_var_Lambda[..., fsize:2 * fsize]
        out = (factor_to_input_eta, factor_to_input_Lambda, factor_to_filter_eta, factor_to_filter_Lambda)
        if self.use_filter_coeffs:
            factor_to_coeff_eta = factor_to_var_eta[..., -1][..., None]
            factor_to_coeff_Lambda = factor_to_var_Lambda[..., -1][..., None]
            out += (factor_to_coeff_eta, factor_to_coeff_Lambda)
        return out

    def _broadcast_to_all_filters(self, n_filters, *msgs):
        msgs_bc = []
        for msg in msgs:
            shp = list(msg.shape)
            shp[3] = n_filters
            msgs_bc.append(tf.broadcast_to(msg, shp))
        return msgs_bc

    def _get_incoming_messages(self):
        msgs_combined = []
        edges = (self.input_var_edges, self.filter_var_edges)
        if self.use_filter_coeffs:
            edges += (self.coeff_var_edges,)
        for mtype in ('eta', 'Lambda'):
            msgs = []
            for e in edges:
                msg_in = getattr(e, f'var_to_fac_{mtype}')
                msgs.append(msg_in)

            msg_combined = tf.concat(msgs, axis=-1)
            msgs_combined.append(msg_combined)
        return msgs_combined

    def update_outgoing_messages(self, conn_vars, **kwargs):
        n_filters = conn_vars[0].shape[-1]

        # Get factor params
        factor_eta, factor_Lambda = self.get_eta_Lambda(conn_vars)

        # Collect incoming messages, from both pixels and filters
        var_msg_in_eta, var_msg_in_Lambda = \
            self._get_incoming_messages()

        # Sum incoming messages with factor params
        factor_plus_mess_eta = var_msg_in_eta + factor_eta
        factor_plus_mess_Lambda = tf.linalg.diag(var_msg_in_Lambda) + factor_Lambda

        # Marginalisation
        marg_outs = \
            self.marginalise(factor_plus_mess_eta,
                             factor_plus_mess_Lambda,
                             factor_eta,
                             factor_Lambda)

        self.input_var_edges.fac_to_var_eta, \
            self.input_var_edges.fac_to_var_Lambda,\
            self.filter_var_edges.fac_to_var_eta, \
            self.filter_var_edges.fac_to_var_Lambda = marg_outs[:4]
        if self.use_filter_coeffs:
            self.coeff_var_edges.fac_to_var_eta, self.coeff_var_edges.fac_to_var_Lambda = marg_outs[4:]

    def patchify_image(self, inputs):
        return patchify_image(image=inputs,
                              ksize_y=self.receptive_field[0],
                              ksize_x=self.receptive_field[1],
                              stride=self.stride)

    def energy(self, conn_vars, robust=None, aggregate=True):
        if self.use_filter_coeffs:
            filters, inputs, coeffs = conn_vars
            coeffs = coeffs[..., None]
        else:
            filters, inputs = conn_vars
            coeffs = 1.
        fs = filters.shape[0]

        x_patch = self.patchify_image(inputs)
        filters_rs = tf.transpose(tf.reshape(filters, [fs ** 2, 1, -1]), [1, 2, 0])[:, None, None, :]
        x_patch = tf.cast(x_patch[..., None, :], tf.float32)
        if self.relative_to_centre:
            # print(x_patch.shape)
            centre_pix_mask = 1. #- tf.one_hot(int(fs ** 2 / 2), fs ** 2)
            diff = x_patch - x_patch[..., int(fs ** 2 / 2)][..., None] * centre_pix_mask - coeffs * filters_rs
        else:
            diff = x_patch - coeffs * filters_rs
        E = tf.reduce_sum(diff ** 2 / self.sigma ** 2, axis=-1)
        if robust is None:
            robust = self.N_rob is not None
        if robust and self.N_rob is not None:
            E = self._robust_correct_energy(E)
        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E

    def get_edge_messages(self):
        edges = [self.input_var_edges.state,
                 self.filter_var_edges.state]
        if self.use_filter_coeffs:
            edges += [self.coeff_var_edges.state]
        return edges

    def set_edge_messages(self, edges):
        self.input_var_edges.state = edges[0]
        self.filter_var_edges.state = edges[1]
        if self.use_filter_coeffs:
            self.coeff_var_edges.state = edges[2]
        return edges
