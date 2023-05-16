# coding=utf-8
import tensorflow as tf

from core.factors.additive_filter_factor import AdditiveFilterFactor
from core.utils.utils import flatten_filters


class AdditiveFilterFactorDecomp(AdditiveFilterFactor):
    def _set_eta_Lambda(self):
        if self.relative_to_centre:
            self.J = tf.convert_to_tensor([1., -1., -1., -1.] if self.use_filter_coeffs else [1., -1., -1.])
        else:
            self.J = tf.convert_to_tensor([1., -1., -1.] if self.use_filter_coeffs else [1., -1.])
        self._Lambda = self.J * self.J[:, None] / self.sigma ** 2.
        if self.use_filter_coeffs:
            self._eta = self.J / self.sigma ** 2.
        else:
            self._eta = tf.zeros((3,)) if self.relative_to_centre else tf.zeros((2,))

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
                mats = [tf.linalg.LinearOperatorDiag(msgs[0]),
                        tf.linalg.LinearOperatorFullMatrix(msgs[1])]
                if self.use_filter_coeffs:
                    mats.append(tf.linalg.LinearOperatorDiag(msgs[2]))
                msg_combined = tf.linalg.LinearOperatorBlockDiag(mats).to_dense()

            else:
                if self.use_filter_coeffs:
                    msg_pc = [msgs[0], msgs[1][..., None], msgs[2][..., None]]
                else:
                    msg_pc = [msgs[0], msgs[1][..., None]]
                msg_combined = tf.concat(msg_pc, axis=-1)
            msgs_combined.append(msg_combined)
        return msgs_combined

    def _get_eta_Lambda_with_filter_coeffs(self, conn_vars):
        # TODO: clean up and comment
        assert not self.dynamic_robust_mixture_weight, "not yet supported"
        # tf.reduce_sum(self.J * )
        filters, pixels, coeffs = self.var0

        filters_flat = flatten_filters(filters)
        ks2 = filters_flat.shape[-1]

        N_rob = self.N_rob
        E = self.energy(self.var0, robust=False, aggregate=False)
        if N_rob is not None:
            k = self.get_k_robust(E, N_rob=N_rob)
        else:
            k = 1.

        coeffs_bc = coeffs[..., None, None]
        filters_flat_bc = filters_flat[None, None, None, ..., None]
        if self.relative_to_centre:
            eta = tf.concat([[self._eta] * int(ks2 / 2),
                             tf.convert_to_tensor([[0., 0., -1., -1.]]) / self.sigma ** 2,
                             [self._eta] * int(ks2 / 2)], axis=0)
        else:
            eta = tf.concat([[self._eta] * ks2], axis=0)
        eta_shp = list(coeffs.shape) + [filters_flat.shape[-1], self._eta.shape[0]]
        eta = tf.broadcast_to(eta, eta_shp)
        eta *= - filters_flat_bc * coeffs_bc
        npix_dim = 2 if self.relative_to_centre else 1
        eta = tf.concat([eta[..., :npix_dim],
                        (eta[..., npix_dim] * coeffs_bc[..., 0])[..., None],
                        (eta[..., npix_dim + 1] * filters_flat_bc[..., 0])[..., None]], axis=-1)
        if self.relative_to_centre:
            central_pixel_Lambda = \
                tf.convert_to_tensor([[0., 0., 0., 0.],
                                      [0., 0., 0., 0.],
                                      [0., 0., 1., -1.],
                                      [0., 0., -1., 1.]]) / self.sigma ** 2
            Lambda = tf.stack([self._Lambda] * int(ks2 / 2) + \
                                     [central_pixel_Lambda] + \
                                     [self._Lambda] * int(ks2 / 2),
                                     axis=-3)
        else:
            Lambda = tf.stack([self._Lambda] * ks2, axis=-3)
        Lambda = tf.broadcast_to(Lambda, eta_shp + [Lambda.shape[-1]])
        Lambda = tf.concat([Lambda[..., :npix_dim, :],
                                   (Lambda[..., npix_dim, :] * coeffs_bc)[..., None, :],
                                   (Lambda[..., npix_dim + 1, :] * filters_flat_bc)[..., None, :]], axis=-2)
        Lambda = tf.concat([Lambda[..., :npix_dim],
                                   (Lambda[..., npix_dim] * coeffs_bc)[..., None],
                                   (Lambda[..., npix_dim + 1] * filters_flat_bc)[..., None]], axis=-1)
        if self.N_rob is not None:
            eta *= k[..., None, None]
            Lambda *= k[..., None, None, None]
        if self.pass_n_low_energy_filter_messages:
            n_lowE = self.pass_n_low_energy_filter_messages
            self.lowest_energy_filter_ids = tf.argsort(E,
                                                       axis=-1,
                                                       direction='ASCENDING')[..., :n_lowE]
            eta = tf.gather(eta, self.lowest_energy_filter_ids, batch_dims=3, axis=-3)
            Lambda = tf.gather(Lambda, self.lowest_energy_filter_ids, batch_dims=3, axis=-4)
        return eta, Lambda

    def get_eta_Lambda(self, conn_vars):
        if self.use_filter_coeffs:
            return self._get_eta_Lambda_with_filter_coeffs(conn_vars)
        else:
            ks2 = self.var0[0].shape[0] * self.var0[0].shape[1]
            eta, Lambda, k = \
                super(AdditiveFilterFactorDecomp, self).get_eta_Lambda(conn_vars, return_k=True)
            central_pixel_Lambda = \
                tf.convert_to_tensor([[0., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 1.]]) / self.sigma ** 2
            central_pixel_Lambda = tf.broadcast_to(central_pixel_Lambda, Lambda.shape)
            if self.pass_n_low_energy_filter_messages:
                k = tf.gather(k, self.lowest_energy_filter_ids, axis=-1, batch_dims=3)
            if self.N_rob is not None:
                central_pixel_Lambda *= k[..., None, None]
            return eta[..., None, :],\
                   tf.stack([Lambda] * int(ks2 / 2) +
                            [central_pixel_Lambda] +
                            [Lambda] * int(ks2 / 2), axis=-3)

    def update_outgoing_messages(self, conn_vars, **kwargs):
        factor_eta, factor_Lambda = self.get_eta_Lambda(conn_vars)

        original_shp = self.input_var_edges.var_to_fac_eta.shape
        var_msg_in_eta, var_msg_in_Lambda = self._get_incoming_messages()
        factor_plus_mess_eta = var_msg_in_eta + factor_eta

        factor_plus_mess_Lambda = tf.linalg.set_diag(factor_Lambda,
                                                     tf.linalg.diag_part(factor_Lambda) + var_msg_in_Lambda,
                                                     name='Lambda_fac_plus_mess_set_diag')

        # Marginalisation
        marg_outs = self.marginalise(factor_plus_mess_eta,
                                     factor_plus_mess_Lambda,
                                     factor_eta,
                                     factor_Lambda)

        if self.pass_n_low_energy_filter_messages:
            # Broadcast messages up to same shape as messages to all filters
            # Zero out high energy messages
            n_filters = original_shp[3]
            marg_outs = self._broadcast_to_all_filters(n_filters, *marg_outs)
            marg_outs = self.zero_higher_energy_filter_messages(*marg_outs)

        self.input_var_edges.fac_to_var_eta, \
            self.input_var_edges.fac_to_var_Lambda,\
            self.filter_var_edges.fac_to_var_eta, \
            self.filter_var_edges.fac_to_var_Lambda = marg_outs[:4]

        if self.use_filter_coeffs:
            self.coeff_var_edges.fac_to_var_eta, \
                    self.coeff_var_edges.fac_to_var_Lambda = marg_outs[4:]

    def marginalise(self,
                    factor_plus_mess_eta,
                    factor_plus_mess_Lambda,
                    factor_eta,
                    factor_Lambda,
                    nvar_marg=None):

        factor_to_var_eta, factor_to_var_Lambda = \
            super(AdditiveFilterFactor, self).marginalise(factor_plus_mess_eta,
                                                          factor_plus_mess_Lambda,
                                                          factor_eta,
                                                          factor_Lambda)
        npix_dim = 2 if self.relative_to_centre else 1
        factor_to_pixel_eta = factor_to_var_eta[..., :npix_dim]
        factor_to_filter_eta = factor_to_var_eta[..., npix_dim]
        factor_to_pixel_Lambda = factor_to_var_Lambda[..., :npix_dim]
        factor_to_filter_Lambda = factor_to_var_Lambda[..., npix_dim]

        out = factor_to_pixel_eta,\
            factor_to_pixel_Lambda,\
            factor_to_filter_eta,\
            factor_to_filter_Lambda
        if self.use_filter_coeffs:
            factor_to_coeff_eta = factor_to_var_eta[..., npix_dim + 1]
            factor_to_coeff_Lambda = factor_to_var_Lambda[..., npix_dim + 1]
            out += (factor_to_coeff_eta, factor_to_coeff_Lambda)
        return out

    def _marg_iter(self,
                   k,
                   eta_fact_plus_mess,
                   Lambda_fact_plus_mess,
                   factor_eta,
                   factor_Lambda,
                   eta_array,
                   Lambda_array):
        factor_size = Lambda_fact_plus_mess.shape[-1]

        def _except_k(x, axis=-2):
            return tf.gather(x,
                             tf.sort(tf.math.mod(tf.range(1, factor_size) + k, factor_size)),
                             axis=axis,
                             name='excpt_k_gather')

        def _except_kk_inv(x_neqk):
            x_neqk_neqk = _except_k(x_neqk, axis=-1)
            if self.use_filter_coeffs:
                return tf.linalg.inv(x_neqk_neqk)
            det = tf.identity(
                x_neqk_neqk[..., 0, 0] * x_neqk_neqk[..., -1, -1] - x_neqk_neqk[..., 1, 0] * x_neqk_neqk[..., 0, 1],
                name='Lambda_bb_det')
            inv = tf.transpose(tf.stack([[x_neqk_neqk[..., 1, 1], -x_neqk_neqk[..., 1, 0]],
                                         [-x_neqk_neqk[..., 0, 1], x_neqk_neqk[..., 0, 0]]], axis=-1),
                               (1, 2, 3, 4, 5, 6, 0, 7))
            return tf.identity(inv / det[..., None, None], name='Lambda_bb_inv')

        Lambda_b_sub = _except_k(Lambda_fact_plus_mess, axis=-2)
        Lambda_bb_inv = _except_kk_inv(Lambda_b_sub)

        def _custom_einsum(Lab, Lbbi, Labeb):
            idn = tf.identity
            out = [idn((Lab[..., 0] * Lbbi[..., 0, 0] + Lab[..., 1] * Lbbi[..., 1, 0]) * Labeb[..., 0, 0],
                       name='ein1') +
                   idn((Lab[..., 0] * Lbbi[..., 0, 1] + Lab[..., 1] * Lbbi[..., 1, 1]) * Labeb[..., 1, 0],
                       name='ein2'),
                   idn((Lab[..., 0] * Lbbi[..., 0, 0] + Lab[..., 1] * Lbbi[..., 1, 0]) * Labeb[..., 0, 1],
                       name='ein3') +
                   idn((Lab[..., 0] * Lbbi[..., 0, 1] + Lab[..., 1] * Lbbi[..., 1, 1]) * Labeb[..., 1, 1],
                       name='ein4')]
            return tf.stack(out, axis=-1, name='custom_einsum_stack')

        Lambda_ab_eta_b = tf.concat(
            [Lambda_b_sub[..., k][..., None], _except_k(eta_fact_plus_mess, axis=-1)[..., None]], axis=-1,
            name='concat_Lambdaab_etab')
        if self.use_filter_coeffs:
            Lambda_ab_bb_inv_ba_eta_b = tf.einsum('abcdef,abcdefg,abcdegh->abcdeh', Lambda_b_sub[..., k], Lambda_bb_inv, Lambda_ab_eta_b)
        else:
            Lambda_ab_bb_inv_ba_eta_b = _custom_einsum(Lambda_b_sub[..., k], Lambda_bb_inv,
                                                       Lambda_ab_eta_b)

        factor_to_var_Lambda_k = \
            factor_Lambda[..., k, k] - \
            Lambda_ab_bb_inv_ba_eta_b[..., 0]
        factor_to_var_eta_k = \
            factor_eta[..., k] - \
            Lambda_ab_bb_inv_ba_eta_b[..., 1]
        return [k + 1,
                eta_fact_plus_mess,
                Lambda_fact_plus_mess,
                factor_eta,
                factor_Lambda,
                eta_array.write(k, factor_to_var_eta_k),
                Lambda_array.write(k, factor_to_var_Lambda_k)]
