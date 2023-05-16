# coding=utf-8
import tensorflow as tf

from core.inference.base import Learner
from core.utils.utils import make_tf_filter_shp


class BatchLearner(Learner):
    def __init__(self, *args, **kwargs):
        self.momentum = kwargs.pop('momentum')
        super(BatchLearner, self).__init__(*args, **kwargs)
        assert not self.use_decomp_filter_factors, \
            "Decomposed filter factors not supported with batch method"

    def _run_inference(self, n_iters, img_new, callback_fn, filter_factor_recon_only=False):
        return super()._run_inference(n_iters, img_new, callback_fn, filter_factor_recon_only=False)

    def _inference_iter(self, itr, *state):
        self.state = state
        self.relinarise_factors(itr)

        self.eta_global, self.Lambda_global = self._get_global_eta_Lambda()
        L = tf.linalg.cholesky(self.Lambda_global)
        self.Sigma2_global = tf.linalg.cholesky_solve(L, tf.eye(self.n_vars))
        self.mu_global = tf.matmul(self.Sigma2_global, self.eta_global[:, None])

        self.update_marginals(itr)
        return [itr + 1] + self.state

    def _get_global_eta_Lambda(self):
        rfsize = self.filter_vars.shape[0] * self.filter_vars.shape[1]
        filtfac_args = [self.filter_vars.mu, self.pixel_vars.mu]
        if self.use_filter_weights:
            filtfac_args += [self.weight_vars.mu]
        filtfac_eta, filtfac_Lambda = self.filter_factor.get_eta_Lambda(filtfac_args)
        eta_pix = filtfac_eta[..., :rfsize]
        Lambda_pixpix = filtfac_Lambda[..., :rfsize, :rfsize]
        if not self.use_sum_filter_factors:
            filt_axis = 3
            Lambda_pixpix = tf.reduce_sum(Lambda_pixpix, filt_axis)
            eta_pix = tf.reduce_sum(eta_pix, filt_axis)

        eta_pix = self.depatchify(eta_pix)
        Lambda_pixpix = self.depatchify_Lambda(Lambda_pixpix)
        Lambda_pixpix = tf.reshape(Lambda_pixpix, (self.n_pixel_vars, self.n_pixel_vars))
        eta_pix = tf.reshape(eta_pix, (self.n_pixel_vars, 1))

        eta_pix_obs, Lambda_pix_obs = self.pixel_obs_factor.get_eta_Lambda([self.pixel_vars.mu])
        eta_pix = eta_pix + tf.reshape(eta_pix_obs, (self.n_pixel_vars, 1))
        Lambda_pixpix = Lambda_pixpix + tf.linalg.diag(tf.reshape(Lambda_pix_obs, (self.n_pixel_vars,)))

        filtmaxid = rfsize + self.n_filter_vars if self.use_sum_filter_factors else 2 * rfsize
        Lambda_filtpix = self.format_Lambdafiltpix(filtfac_Lambda[..., rfsize:filtmaxid, :rfsize])
        Lambda_filtpix = tf.transpose(tf.reshape(Lambda_filtpix, (self.n_filter_vars, self.n_pixel_vars)))

        Lambda_filtfilt = tf.reduce_sum(filtfac_Lambda[..., rfsize:filtmaxid, rfsize:filtmaxid], (0, 1, 2))

        if not self.use_sum_filter_factors:
            Lambda_filtfilt = tf.linalg.LinearOperatorBlockDiag([tf.linalg.LinearOperatorFullMatrix(t) for t in tf.unstack(Lambda_filtfilt)]).to_dense()
        eta_filt = tf.reshape(tf.reduce_sum(filtfac_eta[..., rfsize:filtmaxid], (0, 1, 2)), (self.n_filter_vars, 1))
        eta_filtprior, Lambda_filtprior = self.filter_prior_factor.get_eta_Lambda([self.filter_vars.mu])
        eta_filtprior, Lambda_filtprior = [tf.reshape(f, (self.n_filter_vars,)) for f in (eta_filtprior, Lambda_filtprior)]
        eta_filt = eta_filtprior[..., None] + eta_filt
        Lambda_filtfilt = Lambda_filtfilt + tf.linalg.diag(Lambda_filtprior)

        if self.use_filter_weights:
            Lambda_pixweight = filtfac_Lambda[..., :rfsize, filtmaxid:]
            if self.use_sum_filter_factors:
                Lambda_pixweight = tf.transpose(Lambda_pixweight, (0, 1, 2, 4, 3))
            Lambda_pixweight = self.format_Lambdapixweight(Lambda_pixweight)
            Lambda_pixweight = tf.reshape(Lambda_pixweight, (self.n_pixel_vars, self.n_weight_vars))
            Lambda_weightfilt = filtfac_Lambda[..., rfsize:filtmaxid, filtmaxid:]

            tpose_perm = (0, 1, 2, 4, 3) if self.use_sum_filter_factors else (0, 1, 2, 3, 5, 4)
            Lambda_weightfilt = tf.transpose(Lambda_weightfilt, tpose_perm)
            if not self.use_sum_filter_factors:
                Lambda_weightfilt = tf.broadcast_to(Lambda_weightfilt, Lambda_weightfilt.shape.as_list()[:-2] + [self.filter_vars.n_filters] + Lambda_weightfilt.shape.as_list()[-1:])
                Lambda_weightfilt = tf.eye(self.filter_vars.n_filters)[None, None, None, ..., None] * Lambda_weightfilt
            Lambda_weightfilt = tf.reshape(Lambda_weightfilt, (self.n_weight_vars, self.n_filter_vars))
            Lambda_weightweight = filtfac_Lambda[..., filtmaxid:, filtmaxid:]

            eta_weight = tf.reshape(filtfac_eta[..., filtmaxid:], (self.n_weight_vars, 1))
            eta_weight_prior, Lambda_weight_prior = self.weight_prior_factor.get_eta_Lambda([self.weight_vars.mu])
            eta_weight = tf.reshape(eta_weight_prior, (self.n_weight_vars, 1)) + eta_weight

            if not self.use_sum_filter_factors:
                Lambda_weightweight = Lambda_weightweight[..., 0, 0]
                Lambda_weightweight = Lambda_weightweight + Lambda_weight_prior
                Lambda_weightweight = tf.reshape(Lambda_weightweight, (self.n_weight_vars,))
                Lambda_weightweight = tf.linalg.diag(Lambda_weightweight)
            else:
                Lambda_weightweight = tf.reshape(Lambda_weightweight,
                                                 [-1, self.filter_vars.n_filters,
                                                  self.filter_vars.n_filters])
                Lambda_weightweight = tf.linalg.LinearOperatorBlockDiag(
                    [tf.linalg.LinearOperatorFullMatrix(t) for t in tf.unstack(Lambda_weightweight)]).to_dense()
                Lambda_weightweight = Lambda_weightweight + tf.linalg.diag(tf.reshape(Lambda_weight_prior, self.n_weight_vars))

            if self.use_weight_norm_factor:
                weight_reg_eta, weight_reg_Lambda = \
                    self.non_zero_weight_norm_factor.get_eta_Lambda([self.weight_vars.mu])
                weight_reg_eta = tf.reshape(weight_reg_eta, (self.n_weight_vars, 1))
                weight_reg_Lambda = tf.reshape(weight_reg_Lambda,
                                               (int(self.n_weight_vars / self.filter_vars.n_filters),
                                                self.filter_vars.n_filters,
                                                self.filter_vars.n_filters))
                weight_reg_Lambda = tf.linalg.LinearOperatorBlockDiag([tf.linalg.LinearOperatorFullMatrix(t) for t in tf.unstack(weight_reg_Lambda)]).to_dense()
                Lambda_weightweight = weight_reg_Lambda + Lambda_weightweight
                eta_weight = weight_reg_eta + eta_weight

        Lambda_global = tf.zeros((self.n_vars, self.n_vars))

        def _get_update_ids(ids_a, ids_b):
            return tf.transpose(tf.meshgrid(ids_a, ids_b), [1, 2, 0])

        def _add_to_Lambda(current_Lambda, update_Lambda, ids_a, ids_b=None, also_tpose=False):
            update_ids = _get_update_ids(ids_a, ids_b or ids_a)
            current_Lambda = tf.tensor_scatter_nd_add(current_Lambda, update_ids, update_Lambda)
            if also_tpose:
                assert ids_b is not None
                update_ids = _get_update_ids(ids_b, ids_a)
                current_Lambda = tf.tensor_scatter_nd_add(current_Lambda, update_ids, tf.transpose(update_Lambda))
            return current_Lambda

        pixel_var_ids = range(self.n_pixel_vars)
        filter_var_ids = range(self.n_pixel_vars, self.n_pixel_vars + self.n_filter_vars)
        Lambda_global = _add_to_Lambda(Lambda_global, Lambda_pixpix, pixel_var_ids)
        Lambda_global = _add_to_Lambda(Lambda_global, Lambda_filtpix, filter_var_ids, ids_b=pixel_var_ids, also_tpose=True)
        Lambda_global = _add_to_Lambda(Lambda_global, Lambda_filtfilt, filter_var_ids)

        if self.use_filter_weights:
            weight_var_ids = range(self.n_pixel_vars + self.n_filter_vars, self.n_pixel_vars + self.n_filter_vars + self.n_weight_vars)
            Lambda_global = _add_to_Lambda(Lambda_global, Lambda_weightweight, weight_var_ids)
            Lambda_global = _add_to_Lambda(Lambda_global, Lambda_pixweight, weight_var_ids, ids_b=pixel_var_ids, also_tpose=True)
            Lambda_global = _add_to_Lambda(Lambda_global, Lambda_weightfilt, filter_var_ids, ids_b=weight_var_ids, also_tpose=True)
            eta_global = tf.concat([eta_pix, eta_filt, eta_weight], axis=0)
        else:
            eta_global = tf.concat([eta_pix, eta_filt], axis=0)
        return eta_global[..., 0], Lambda_global

    def format_Lambdafiltpix(self, Lambdafiltpix):
        paramshp = [-1,
                    self.n_strides_y,
                    self.n_strides_x,
                    self.filter_vars.shape[-1],
                    self.filter_vars.k_size ** 2,
                    self.filter_vars.k_size ** 2]
        Lambdafiltpix = tf.reshape(Lambdafiltpix, paramshp)

        recfield = self.filter_vars.k_size ** 2
        out_shp = [self.filter_vars.shape[-1],
                   self.filter_vars.k_size ** 2, self.img_in.shape[1], self.img_in.shape[2]]
        Lfp_out = tf.zeros(out_shp)
        stride_ids = \
            tf.transpose(tf.meshgrid(tf.range(self.n_strides_y * self.stride, delta=self.stride),
                                     tf.range(self.n_strides_x * self.stride, delta=self.stride)), (2, 1, 0))
        for f in range(self.filter_vars.shape[-1]):
            for vpix in range(recfield):
                for vfilt in range(recfield):
                    vy, vx = vpix // self.filter_vars.k_size, vpix % self.filter_vars.k_size
                    idx = tf.concat((tf.broadcast_to([[f, vfilt]], stride_ids.shape), stride_ids + [[vy, vx]]), axis=-1)
                    Lfp_out = tf.tensor_scatter_nd_add(Lfp_out,
                                                     idx,
                                                     Lambdafiltpix[0, :, :, f, vfilt, vpix])
        return Lfp_out  # TODO: allow for batched input

    def format_Lambdapixweight(self, Lambdapixweight):
        paramshp = [-1,
                    self.n_strides_y,
                    self.n_strides_x,
                    self.filter_vars.shape[-1],
                    self.filter_vars.k_size ** 2,
                    1]
        Lambdapixweight = tf.reshape(Lambdapixweight, paramshp)

        recfield = self.filter_vars.k_size ** 2
        out_shp = [self.img_in.shape[1], self.img_in.shape[2],
                   self.n_strides_y, self.n_strides_x,
                   self.filter_vars.shape[-1]]
        Lpw_out = tf.zeros(out_shp)
        stride_ids = \
            tf.transpose(tf.meshgrid(tf.range(self.n_strides_y * self.stride, delta=self.stride),
                                     tf.range(self.n_strides_x * self.stride, delta=self.stride)), (2, 1, 0))
        # for f in range(self.filter_vars.shape[-1]):
        for vpix in range(recfield):
            vx, vy = vpix // self.filter_vars.k_size, vpix % self.filter_vars.k_size
            idx = tf.concat((stride_ids + [[vx, vy]], tf.cast(stride_ids / self.stride, tf.int32)), axis=-1)
            Lpw_out = tf.tensor_scatter_nd_add(Lpw_out,
                                             idx,
                                             Lambdapixweight[0, :, :, :, vpix, 0])
        return Lpw_out  # TODO: allow for batched input

    def update_marginals(self, itr):
        Sigma2_diag = tf.linalg.diag_part(self.Sigma2_global)[..., None]

        def _set_marginal(var_ids, var_obj, reshp_fn=None):
            mu_marg = tf.gather(self.mu_global, var_ids)
            Sigma2_marg = tf.gather(Sigma2_diag, var_ids)
            if reshp_fn is None:
                reshp_fn = lambda x: tf.reshape(x, var_obj.shape)

            setattr(var_obj, 'eta', self.momentum * var_obj.eta + (1. - self.momentum) * reshp_fn(mu_marg / Sigma2_marg))
            setattr(var_obj, 'Lambda', self.momentum * var_obj.Lambda + (1. - self.momentum) * reshp_fn(1. / Sigma2_marg))

        pixel_var_ids = range(self.n_pixel_vars)
        _set_marginal(pixel_var_ids, self.pixel_vars)

        if not self.use_filter_weights or itr // 5 % 1 == 0:
            filter_var_ids = range(self.n_pixel_vars, self.n_pixel_vars + self.n_filter_vars)
            _set_marginal(filter_var_ids, self.filter_vars, reshp_fn=lambda x: make_tf_filter_shp(tf.reshape(x, (self.filter_vars.n_filters, self.filter_vars.k_size ** 2)), self.filter_vars.k_size))
            # print(self.momentum)
            # afsd
        if self.use_filter_weights and itr // 5 % 1 == 0:
            weight_var_ids = range(self.n_pixel_vars + self.n_filter_vars,
                                   self.n_pixel_vars + self.n_filter_vars + self.n_weight_vars)
            _set_marginal(weight_var_ids, self.weight_vars)
