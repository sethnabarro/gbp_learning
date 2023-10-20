# coding=utf-8
import numpy as np
import tensorflow as tf

from core.utils.utils import FixedAttributeFieldsObject
from core.factors.utils import get_nonlin_fns, get_nonlin_fn_grads


class Factor(FixedAttributeFieldsObject):
    N_PARALLEL_PROC_MARG = 1
    REPULSIVE_N_MULT = 1.5
    REPULSIVE_SIGMA_ROB_MULT = 1.
    REPULSIVE_TAIL_OFFSET = 0.

    def __init__(self,
                 sigma: float,
                 N_rob: [float, None] = None,
                 rob_type: [str, None] = 'tukey',
                 kmult: float = 1.):
        self.sigma = sigma
        self.N_rob = N_rob
        self.rob_type = rob_type
        self.kmult = kmult

    def energy(self, conn_vars, robust=None, aggregate=True):
        raise NotImplementedError('Implemented in child class')

    def get_eta_Lambda(self, conn_vars, **kwargs):
        raise NotImplementedError('Implemented in child class')

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        raise NotImplementedError('Implemented in child class')

    def get_edge_messages(self, named=False):
        raise NotImplementedError('Implemented in child class')

    def set_edge_messages(self, edge_msgs):
        raise NotImplementedError('Implemented in child class')

    def get_k_robust(self, E, N_rob=None):
        N_rob = N_rob or self.N_rob
        if N_rob is None or self.rob_type is None:
            k = tf.ones_like(E)
        else:
            M_mnobis = tf.sqrt(E)
            N_div_M = N_rob / M_mnobis
            if self.rob_type == 'huber':
                k = 2. * N_div_M - N_div_M ** 2
            elif self.rob_type == 'tukey':
                k = N_div_M ** 2
            elif self.rob_type == 'repulsive':
                N_mult = self.REPULSIVE_N_MULT
                tail_offset = self.REPULSIVE_TAIL_OFFSET
                sig_rob = self.sigma * self.REPULSIVE_SIGMA_ROB_MULT
                diff = M_mnobis * self.sigma
                new_sigma = tf.sqrt(diff ** 2. / (tf.exp(
                    -(tf.abs(diff) - N_mult * N_rob * self.sigma) ** 2. / (N_rob * sig_rob) ** 2.) * N_rob ** 2. / tf.exp(
                    -((N_mult * N_rob * self.sigma - N_rob * self.sigma) / (sig_rob * N_rob)) ** 2.) + tail_offset))
                k = self.sigma / new_sigma
            else:
                raise ValueError(f'robust_type {self.rob_type} not supported. '
                                 f'Must be "huber", "repulsive" or "tukey"')
            if self.kmult is not None:
                k *= self.kmult
            k = tf.where(M_mnobis > N_rob, k, 1.)
        return k

    def _robust_correct_energy(self, E):
        if self.rob_type is None or self.N_rob is None:
            return E
        elif self.rob_type == 'tukey':
            return tf.where(E > self.N_rob ** 2, self.N_rob ** 2, E)
        elif self.rob_type == 'huber':
            residual = tf.sqrt(E)
            M_rob = self.N_rob * (2. * residual - self.N_rob)
            return tf.where(E > self.N_rob ** 2, M_rob ** 2, E)
        elif self.rob_type == 'repulsive':
            N_mult = 1.5
            tail_offset = 0.
            sig_rob = self.sigma * 1.
            diff = tf.sqrt(E) * self.sigma
            N_rob = self.N_rob
            rescale = self.N_rob ** 2 / tf.exp(-((N_mult * N_rob * self.sigma - N_rob * self.sigma) / (sig_rob * N_rob)) ** 2.)
            E_rob = tf.exp(-((tf.abs(diff) - (-N_mult * self.N_rob * self.sigma)) / (self.N_rob * sig_rob)) ** 2) * rescale + tail_offset
            return tf.where(E > self.N_rob ** 2, E_rob, E)
        else:
            raise ValueError("`rob_type` must be 'tukey', 'huber', 'repulsive' or None")

    def marginalise(self,
                    factor_plus_mess_eta,
                    factor_plus_mess_Lambda,
                    factor_eta,
                    factor_Lambda,
                    nvar_to_marg=None):
        n_vars = nvar_to_marg or factor_eta.shape[-1]
        rank = len(factor_eta.shape)
        eta_arr = tf.TensorArray(factor_plus_mess_eta.dtype, size=n_vars)
        Lambda_arr = tf.TensorArray(factor_plus_mess_Lambda.dtype, size=n_vars)

        loop_vars = [0,
                     factor_plus_mess_eta,
                     factor_plus_mess_Lambda,
                     factor_eta,
                     factor_Lambda,
                     eta_arr,
                     Lambda_arr]
        _, _, _, _, _, eta_arr, Lambda_arr = \
            tf.while_loop(cond=lambda k, *_: k < n_vars,
                          body=self._marg_iter,
                          loop_vars=loop_vars,
                          back_prop=False,
                          parallel_iterations=self.N_PARALLEL_PROC_MARG,
                          name='srlised_marg_loops')

        factor_to_var_eta = tf.transpose(eta_arr.stack(), perm=[r for r in range(1, rank)] + [0])
        factor_to_var_Lambda = tf.transpose(Lambda_arr.stack(), perm=[r for r in range(1, rank)] + [0])
        return factor_to_var_eta, factor_to_var_Lambda

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
            return tf.linalg.inv(x_neqk_neqk)

        Lambda_b_sub = _except_k(Lambda_fact_plus_mess, axis=-2)
        Lambda_bb_inv = _except_kk_inv(Lambda_b_sub)

        Lambda_ab_eta_b = tf.concat(
            [Lambda_b_sub[..., k][..., None], _except_k(eta_fact_plus_mess, axis=-1)[..., None]], axis=-1,
            name='concat_Lambdaab_etab')
        n_batch_dims = len(Lambda_bb_inv.shape) - 2
        bdims = 'abcdefghijklmn'[:n_batch_dims]
        Lambda_ab_bb_inv_ba_eta_b = tf.einsum(f'{bdims}x,{bdims}xy,{bdims}yz->{bdims}z',
                                              Lambda_b_sub[..., k],
                                              Lambda_bb_inv,
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

    def _marg_iter_vector(self,
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
            return tf.linalg.inv(x_neqk_neqk)

        Lambda_b_sub = _except_k(Lambda_fact_plus_mess, axis=-2)
        Lambda_bb_inv = _except_kk_inv(Lambda_b_sub)

        Lambda_ab_eta_b = tf.concat(
            [Lambda_b_sub[..., k][..., None], _except_k(eta_fact_plus_mess, axis=-1)[..., None]], axis=-1,
            name='concat_Lambdaab_etab')
        n_batch_dims = len(Lambda_bb_inv.shape) - 2
        bdims = 'abcdefghijklmn'[:n_batch_dims]
        Lambda_ab_bb_inv_ba_eta_b = tf.einsum(f'{bdims}x,{bdims}xy,{bdims}yz->{bdims}z',
                                              Lambda_b_sub[..., k],
                                              Lambda_bb_inv,
                                              Lambda_ab_eta_b)

        factor_to_var_Lambda_k = \
            factor_Lambda[..., k, k] - \
            Lambda_ab_bb_inv_ba_eta_b[..., 0]
        factor_to_var_eta_k = \
            factor_eta[..., k] - \
            Lambda_ab_bb_inv_ba_eta_b[..., 1]
        return factor_to_var_eta_k, factor_to_var_Lambda_k

    @staticmethod
    def marginalise_sherman_morrison_older(
                    mess_eta,
                    factor_eta,
                    mess_Lambda,
                    J_div_sigma,
                    n_parallel_proc=1,
                    mask=None):
        n_vars = factor_eta.shape[-1]
        rank = len(factor_eta.shape)
        eta_arr = tf.TensorArray(factor_eta.dtype, size=n_vars)
        Lambda_arr = tf.TensorArray(factor_eta.dtype, size=n_vars)

        loop_vars = [0,
                     mess_eta,
                     factor_eta,
                     mess_Lambda,
                     J_div_sigma,
                     eta_arr,
                     Lambda_arr]
        _, _, _, _, _, eta_arr, Lambda_arr = \
            tf.while_loop(cond=lambda k, *_: k < n_vars,
                          body=lambda *args, **kwargs: Factor._marg_iter_sherman_morrison(*args, **kwargs, mask=mask),
                          loop_vars=loop_vars,
                          back_prop=False,
                          parallel_iterations=n_parallel_proc,
                          maximum_iterations=n_vars,
                          name='srlised_marg_loops_sherm_morr')

        factor_to_var_eta = tf.transpose(eta_arr.stack(), perm=[r for r in range(1, rank)] + [0])
        factor_to_var_Lambda = tf.transpose(Lambda_arr.stack(), perm=[r for r in range(1, rank)] + [0])
        if mask is not None:
            factor_to_var_eta = tf.math.multiply_no_nan(factor_to_var_eta, mask)
            factor_to_var_Lambda = tf.math.multiply_no_nan(factor_to_var_Lambda, mask)
        return factor_to_var_eta, factor_to_var_Lambda

    @staticmethod
    def marginalise_sherman_morrison(
                    mess_eta,
                    factor_eta,
                    mess_Lambda,
                    J_div_sigma,
                    n_parallel_proc=1,
                    mask=None,
                    batchsize=1):

        mess_Lambda_inv = 1. / mess_Lambda
        if mask is not None:
            J_div_sigma *= mask
            mess_Lambda_inv = tf.math.multiply_no_nan(mess_Lambda_inv, mask)
        J_div_sigma2 = J_div_sigma ** 2.
        J_mess_Lambda_inv_Jeta_elem = tf.stack([J_div_sigma * mess_Lambda_inv * (factor_eta + mess_eta),
                                                J_div_sigma2 * mess_Lambda_inv], axis=-1)
        J_mess_Lambda_inv_Jeta = tf.reduce_sum(J_mess_Lambda_inv_Jeta_elem, axis=-2, keepdims=True)
        J_mess_Lambda_inv_Jeta_sub_elemcorr = J_mess_Lambda_inv_Jeta - J_mess_Lambda_inv_Jeta_elem
        J_mess_Lambda_inv_J = J_mess_Lambda_inv_Jeta_sub_elemcorr[..., 1:]
        second = J_mess_Lambda_inv_J * J_mess_Lambda_inv_Jeta_sub_elemcorr / (1. + J_mess_Lambda_inv_J)
        adjustment = \
            tf.stack([J_div_sigma, J_div_sigma2], axis=-1) * (J_mess_Lambda_inv_Jeta_sub_elemcorr - second)
        marg = tf.stack([factor_eta, J_div_sigma2], axis=-1)
        outmsg = marg - adjustment
        factor_to_var_eta, factor_to_var_Lambda = outmsg[..., 0], outmsg[..., 1]
        if mask is not None:
            factor_to_var_eta = tf.math.multiply_no_nan(factor_to_var_eta, mask)
            factor_to_var_Lambda = tf.math.multiply_no_nan(factor_to_var_Lambda, mask)
        return factor_to_var_eta, factor_to_var_Lambda

    @staticmethod
    def marginalise_sherman_morrison_old(
                    mess_eta,
                    factor_eta,
                    mess_Lambda,
                    J_div_sigma,
                    n_parallel_proc=1,
                    mask=None,
                    batchsize=1):
        n_vars = factor_eta.shape[-1]
        n_iters = int(np.ceil(n_vars / batchsize))
        rank = len(factor_eta.shape)
        eta_arr = tf.TensorArray(factor_eta.dtype, size=n_iters)
        Lambda_arr = tf.TensorArray(factor_eta.dtype, size=n_iters)

        mess_Lambda_inv = 1. / mess_Lambda
        if mask is not None:
            J_div_sigma *= mask
            mess_Lambda_inv = tf.math.multiply_no_nan(mess_Lambda_inv, mask)

        J_mess_tensor = tf.concat([(J_div_sigma ** 2. * mess_Lambda_inv)[..., None],
                                   (J_div_sigma * mess_Lambda_inv * (factor_eta + mess_eta))[..., None],
                                   ], axis=-1)

        if n_vars % batchsize != 0:
            # Pad the tensors, so all loops operate on objects with same shape
            n_extra_rows = batchsize - n_vars % batchsize
            J_mess_tensor_extra = tf.zeros_like(J_mess_tensor[..., :n_extra_rows, :])
            factor_eta_extra = tf.zeros_like(factor_eta[..., :n_extra_rows])
            J_div_sigma_extra = factor_eta_extra
            J_mess_tensor = tf.concat([J_mess_tensor, J_mess_tensor_extra], axis=-2)
            factor_eta = tf.concat([factor_eta, factor_eta_extra], axis=-1)
            J_div_sigma = tf.concat([J_div_sigma, J_div_sigma_extra], axis=-1)
        loop_vars = [0,
                     eta_arr,
                     Lambda_arr]
        _, eta_arr, Lambda_arr = \
            tf.while_loop(cond=lambda k, *_: k < n_iters,
                          body=lambda *args, **kwargs: Factor._marg_iter_sherman_morrison_opt(*args, **kwargs,
                                                                                              factor_eta=factor_eta,
                                                                                              J_div_sigma=J_div_sigma,
                                                                                              J_mess_tensor=J_mess_tensor,
                                                                                              mask=mask,
                                                                                              batchsize=batchsize),
                          loop_vars=loop_vars,
                          back_prop=False,
                          parallel_iterations=n_parallel_proc,
                          maximum_iterations=n_iters,
                          name='srlised_marg_loops_sherm_morr_opt')

        factor_to_var_eta = tf.transpose(eta_arr.stack(), perm=[r for r in range(1, rank)] + [0, rank])
        factor_to_var_Lambda = tf.transpose(Lambda_arr.stack(), perm=[r for r in range(1, rank)] + [0, rank])

        # Combine batch and var dims
        eta_shp = factor_to_var_eta.shape.as_list()
        target_shp = eta_shp[:-2] + [eta_shp[-2] * batchsize]
        factor_to_var_eta = tf.reshape(factor_to_var_eta, target_shp)[..., :n_vars]
        factor_to_var_Lambda = tf.reshape(factor_to_var_Lambda, target_shp)[..., :n_vars]

        if mask is not None:
            factor_to_var_eta = tf.math.multiply_no_nan(factor_to_var_eta, mask)
            factor_to_var_Lambda = tf.math.multiply_no_nan(factor_to_var_Lambda, mask)
        return factor_to_var_eta, factor_to_var_Lambda

    @staticmethod
    def _sherman_morrison_inv(u, A_diag):
        """https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula"""
        A_diag_inv = 1. / A_diag
        A_diag_inv_u = u * A_diag_inv
        denom = (1. + tf.reduce_sum(A_diag_inv_u * u, axis=-1))[..., None, None]
        numerator = A_diag_inv_u[..., None, :] * A_diag_inv_u[..., None]
        return tf.linalg.diag(A_diag_inv) - numerator / denom

    def _marg_iter_decomp_(self,
                          k,
                    mess_eta,
                    factor_eta,
                    mess_Lambda,
                    J_div_sigma,
                          eta_array,
                          Lambda_array):
        # TODO: NOT IN USE. delete?
        factor_size = factor_eta.shape[-1]

        def _except_k(x, axis=-2):
            return tf.gather(x,
                             tf.sort(tf.math.mod(tf.range(1, factor_size) + k, factor_size)),
                             axis=axis,
                             name='excpt_k_gather')

        J_b = _except_k(J_div_sigma, axis=-1)
        mess_b = _except_k(mess_Lambda, axis=-1)
        Lambda_bb_inv = self._sherman_morrison_inv(J_b, mess_b)
        Lambda_ab = J_b * J_div_sigma[..., k:k + 1]

        eta_plus_mess = factor_eta + mess_eta
        Lambda_ab_eta_b = tf.concat(
            [Lambda_ab[..., None], _except_k(eta_plus_mess, axis=-1)[..., None]], axis=-1,
            name='concat_Lambdaab_etab')
        n_batch_dims = len(Lambda_bb_inv.shape) - 2
        bdims = 'abcdefghijklmn'[:n_batch_dims]
        Lambda_ab_bb_inv_ba_eta_b = tf.einsum(f'{bdims}x,{bdims}xy,{bdims}yz->{bdims}z',
                                              Lambda_ab,
                                              Lambda_bb_inv,
                                              Lambda_ab_eta_b)

        factor_to_var_Lambda_k = \
            J_div_sigma[..., k] ** 2. - \
            Lambda_ab_bb_inv_ba_eta_b[..., 0]
        factor_to_var_eta_k = \
            factor_eta[..., k] - \
            Lambda_ab_bb_inv_ba_eta_b[..., 1]
        return [k + 1,
                mess_eta,
                factor_eta,
                mess_Lambda,
                J_div_sigma,
                eta_array.write(k, factor_to_var_eta_k),
                Lambda_array.write(k, factor_to_var_Lambda_k)]

    @staticmethod
    def _marg_iter_sherman_morrison_opt(
                            k,
                            eta_array,
                            Lambda_array,
                            factor_eta,
                            J_mess_tensor,
                            J_div_sigma,
                            mask=None,
                            batchsize=3):
        factor_size = factor_eta.shape[-1]
        batch_idx_start = k * batchsize
        batch_idx_end = batch_idx_start + batchsize

        def _except_k(x, axis=-2):
            except_idx = tf.math.mod(tf.range(1, factor_size)[None] + tf.range(batch_idx_start, batch_idx_end)[..., None], factor_size)
            except_idx_sort = tf.sort(except_idx)
            return tf.gather(x,
                             except_idx_sort,
                             axis=axis,
                             name='excpt_k_gather')

        J_mess_subk = _except_k(J_mess_tensor, axis=-2)
        J_eta_Lambda_inv_J = tf.reduce_sum(J_mess_subk, axis=-2)
        J2_Lambda_inv_sum = J_eta_Lambda_inv_J[..., 0][..., None]
        second = J2_Lambda_inv_sum * J_eta_Lambda_inv_J / (1. + J2_Lambda_inv_sum)
        corr = J_eta_Lambda_inv_J - second
        J_k = J_div_sigma[..., batch_idx_start:batch_idx_end]
        factor_to_var_Lambda_k = \
            J_k ** 2. * (1. - corr[..., 0])
        factor_to_var_eta_k = \
            factor_eta[..., batch_idx_start:batch_idx_end] - J_k * corr[..., 1]

        return [k + 1,
                eta_array.write(k, factor_to_var_eta_k),
                Lambda_array.write(k, factor_to_var_Lambda_k)]

    @staticmethod
    def _marg_iter_sherman_morrison(
                            k,
                            mess_eta,
                            factor_eta,
                            mess_Lambda,
                            J_div_sigma,
                            eta_array,
                            Lambda_array,
                            mask=None):
        factor_size = factor_eta.shape[-1]

        def _except_k(x, axis=-2):
            return tf.gather(x,
                             tf.sort(tf.math.mod(tf.range(1, factor_size) + k, factor_size)),
                             axis=axis,
                             name='excpt_k_gather')
        if mask is not None:
            mask_b = _except_k(mask, axis=-1)
        else:
            mask_b = 1.

        def _sherman_morrison_components(J_sub, mess_sub):   # TODO: should be able to precompute all of this
            mess_sub_inv = 1. / mess_sub   # TODO: precompute reciprocal of messages
            if mask is not None:
                mess_sub_inv = tf.math.multiply_no_nan(mess_sub_inv, mask_b)
                J_sub *= mask_b
            JTmess_sub_inv = J_sub * mess_sub_inv   # TODO: precompute J * message_reciprocal
            # TODO: precompute J * message_reciprocal * J
            return mess_sub_inv, JTmess_sub_inv / tf.sqrt(1. + tf.reduce_sum(JTmess_sub_inv * J_sub, axis=-1)[..., None])

        J_b = _except_k(J_div_sigma, axis=-1)
        mess_b = _except_k(mess_Lambda, axis=-1)
        mess_sub_inv_vec, u = _sherman_morrison_components(J_b, mess_b)

        Lambda_ab = J_b * J_div_sigma[..., k:k + 1]

        eta_plus_mess = factor_eta + mess_eta  # TODO: precompute, and precomp eta_plus_mess
        Lambda_ab_eta_b = tf.concat(
            [Lambda_ab[..., None], _except_k(eta_plus_mess, axis=-1)[..., None]], axis=-1,
            name='concat_Lambdaab_etab')

        Lambda_ab_mess_inv = Lambda_ab * mess_sub_inv_vec
        Lambda_ab_bb_mess_inv_ba_eta_b = tf.reduce_sum(Lambda_ab_mess_inv[..., None] * Lambda_ab_eta_b, axis=-2)

        Lambda_ab_eta_b_mess_inv_u = tf.reduce_sum(u[..., None] * Lambda_ab_eta_b, axis=-2)

        Lambda_ab_mess_inv_u = tf.reduce_sum(Lambda_ab * u, axis=-1)

        Lambda_ab_bb_u_inv_ba_eta_b = Lambda_ab_eta_b_mess_inv_u * Lambda_ab_mess_inv_u[..., None]
        # print(Lambda_ab_eta_b_mess_inv_u.shape, Lambda_ab_bb_u_inv_ba_eta_b.shape, 'nnnn')
        Lambda_ab_bb_inv_ba_eta_b = Lambda_ab_bb_mess_inv_ba_eta_b - Lambda_ab_bb_u_inv_ba_eta_b

        factor_to_var_Lambda_k = \
            J_div_sigma[..., k] ** 2. - \
            Lambda_ab_bb_inv_ba_eta_b[..., 0]
        factor_to_var_eta_k = \
            factor_eta[..., k] - \
            Lambda_ab_bb_inv_ba_eta_b[..., 1]

        return [k + 1,
                mess_eta,
                factor_eta,
                mess_Lambda,
                J_div_sigma,
                eta_array.write(k, factor_to_var_eta_k),
                Lambda_array.write(k, factor_to_var_Lambda_k)]

    @staticmethod
    def build_nonlin_and_grad(nonlin_name, xscale, yscale):
        nonlin_fns = get_nonlin_fns()
        nonlin_grads = get_nonlin_fn_grads()

        if nonlin_name is None:
            return lambda x: x, lambda _: tf.convert_to_tensor([1.])

        def nonlin_fn(x):
            return yscale * nonlin_fns[nonlin_name](xscale * x)

        def nonlin_gradx(x):
            return yscale * nonlin_grads[nonlin_name](xscale * x)
        return nonlin_fn, nonlin_gradx

    @property
    def state(self):
        st = [self.get_edge_messages()]
        return st

    @state.setter
    def state(self, new_state):
        self.set_edge_messages(new_state[0])

    @property
    def named_state(self):
        st = [('edges', self.get_edge_messages(named=True))]
        return st
