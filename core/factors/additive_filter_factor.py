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
                 dynamic_robust_mixture_weight=False,
                 pass_n_low_energy_filter_messages=False,
                 compute_low_energy_filter_message_only=False,
                 rec_field=(3, 3),
                 stride=1,
                 multidim_filter_vars=False,
                 coeff_var_edges=None,
                 component_var_edges=None,
                 relative_to_centre=True,
                 relin_freq=1,
                 kmult=1.,
                 nonlin=None,
                 nonlin_xscale=None,
                 nonlin_yscale=None):
        self.relative_to_centre = relative_to_centre
        super(AdditiveFilterFactor, self).__init__(sigma, init_lin_point, relin_freq, N_rob, rob_type, kmult=kmult)
        self.dynamic_robust_mixture_weight = dynamic_robust_mixture_weight
        self.compute_low_energy_filter_message_only = compute_low_energy_filter_message_only
        self.pass_n_low_energy_filter_messages = pass_n_low_energy_filter_messages
        self.receptive_field = rec_field
        self.use_filter_coeffs = coeff_var_edges is not None
        self._set_eta_Lambda()
        self.multidim_filter_vars = multidim_filter_vars
        self.input_var_edges = input_var_edges
        self.filter_var_edges = filter_var_edges
        self.coeff_var_edges = coeff_var_edges
        self.use_component_vars = component_var_edges is not None
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
        if self.dynamic_robust_mixture_weight:
            E = self.energy(conn_vars, robust=False, aggregate=False)
            E_min = tf.reduce_min(E, axis=-1)  # Minimum energies over factors connected to each neighbourhood
            E_diff = E - E_min[..., None]
            N_rob = tf.clip_by_value(N_rob - tf.sqrt(E_diff), 0.1, 1e10)

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
        if self.pass_n_low_energy_filter_messages:
            E = self.energy(conn_vars, robust=False, aggregate=False)
            self.lowest_energy_filter_ids = tf.argsort(E,
                                                       axis=-1,
                                                       direction='ASCENDING')[..., :self.pass_n_low_energy_filter_messages]
            eta = tf.gather(eta, self.lowest_energy_filter_ids, batch_dims=3, axis=-2)
            Lambda = tf.gather(Lambda, self.lowest_energy_filter_ids, batch_dims=3, axis=-3)
        if 'return_k' in kwargs:
            if kwargs['return_k']:
                return eta, Lambda, (k if self.N_rob is not None else 1.)
        return eta, Lambda

    def _marg_iter_multidim_filter_var(self,
                                       factor_plus_mess_eta,
                                       factor_plus_mess_Lambda,
                                       factor_eta,
                                       factor_Lambda):
        ksize = self.receptive_field[0] * self.receptive_field[1]

        # Marginalise out the pixel vars
        # TODO: souble check this is correct
        Lambda_b_sub = factor_plus_mess_Lambda[..., :ksize, :]
        Lambda_ba, Lambda_bb = Lambda_b_sub[..., ksize:], Lambda_b_sub[..., :ksize]
        Lambda_aa = factor_Lambda[..., ksize:, ksize:]
        eta_a, eta_b = factor_eta[..., ksize:], factor_plus_mess_eta[..., :ksize]
        Lambda_bb_inv = tf.linalg.inv(Lambda_bb)
        factor_to_var_Lambda_k = \
            Lambda_aa - tf.einsum('abcdef,abcdeh,abcdhl->abcdfl', Lambda_ba, Lambda_bb_inv, Lambda_ba)
        factor_to_var_eta_k = eta_a - tf.einsum('abcdel,abcdef,abcdf->abcdl', Lambda_ba, Lambda_bb_inv, eta_b)
        return factor_to_var_eta_k, factor_to_var_Lambda_k

    def marginalise(self,
                    factor_plus_mess_eta,
                    factor_plus_mess_Lambda,
                    # var_msg_in_eta,
                    # var_msg_in_Lambda):
                    factor_eta,
                    factor_Lambda,
                    nvar_marg=None):
        fsize = self.receptive_field[0] * self.receptive_field[1]
        nvar_marg = fsize if self.multidim_filter_vars else fsize * 2 + int(self.use_filter_coeffs)
        factor_to_var_eta, factor_to_var_Lambda = \
            super(AdditiveFilterFactor, self).marginalise(factor_plus_mess_eta,
                                                          factor_plus_mess_Lambda,
                                                          factor_eta,
                                                          factor_Lambda,
                                                          nvar_to_marg=nvar_marg)

        if self.multidim_filter_vars:
            # Compute message to filter vars
            factor_to_filter_eta, factor_to_filter_Lambda = \
                self._marg_iter_multidim_filter_var(factor_plus_mess_eta=factor_plus_mess_eta,
                                                    factor_plus_mess_Lambda=factor_plus_mess_Lambda,
                                                    factor_eta=factor_eta,
                                                    factor_Lambda=factor_Lambda)
            factor_to_input_eta, factor_to_input_Lambda = factor_to_var_eta, factor_to_var_Lambda

        else:
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

    def zero_higher_energy_filter_messages(self, *mess):
        m_zerod = []
        for m in mess:
            n_filters = m.shape[3]
            is_low_E = tf.fill(self.lowest_energy_filter_ids.shape, False)
            for l in range(self.lowest_energy_filter_ids.shape[-1]):
                is_low_E = tf.logical_or(is_low_E,
                                              tf.range(n_filters) == self.lowest_energy_filter_ids[..., l][..., None])

            mrank = len(tf.shape(m))
            elow_rank = len(tf.shape(is_low_E))
            is_low_E_unsqueeze = tf.reshape(is_low_E, list(is_low_E.shape) + [1] * (mrank - elow_rank))
            m_zerod.append(tf.where(is_low_E_unsqueeze, m, m * 0.))

        return m_zerod

    def _select_low_energy_msgs(self, msg):
        return tf.gather(msg,
                         indices=self.lowest_energy_filter_ids,
                         axis=3, batch_dims=3)

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

                if self.pass_n_low_energy_filter_messages:
                    msgs.append(self._select_low_energy_msgs(msg_in))
                else:
                    msgs.append(msg_in)

            if mtype == 'Lambda' and self.multidim_filter_vars:
                msg_combined = tf.linalg.LinearOperatorBlockDiag(
                    [tf.linalg.LinearOperatorDiag(msgs[0]),
                     tf.linalg.LinearOperatorFullMatrix(msgs[1])]).to_dense()

            else:
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
        if self.multidim_filter_vars:
            factor_plus_mess_Lambda = var_msg_in_Lambda + factor_Lambda
        else:
            factor_plus_mess_Lambda = tf.linalg.diag(var_msg_in_Lambda) + factor_Lambda

        # Marginalisation
        marg_outs = \
            self.marginalise(factor_plus_mess_eta,
                             factor_plus_mess_Lambda,
                             factor_eta,
                             factor_Lambda)

        if self.pass_n_low_energy_filter_messages:
            # Broadcast messages up to same shape as messages to all filters
            # Zero out high energy messages
            marg_outs = self._broadcast_to_all_filters(n_filters, *marg_outs)
            marg_outs = self.zero_higher_energy_filter_messages(*marg_outs)

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

    def zero_out_high_energy_variable_to_factor_messages(self):
        for e in (self.input_var_edges, self.filter_var_edges):
            for mtype in ('var_to_fac_eta', 'var_to_fac_Lambda'):
                mess = getattr(e, mtype)
                setattr(e, mtype, self.zero_higher_energy_filter_messages(mess)[0])

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
