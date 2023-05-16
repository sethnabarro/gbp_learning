# coding=utf-8
import tensorflow as tf

from core.factors.pixel_obs import PixelObservationFactor


class SegmentationObservationFactor(PixelObservationFactor):
    def __init__(self,
                 sigma,
                 sparse_labels,
                 dense_labels,
                 var_edges,
                 N_rob=None,
                 rob_type='tukey',
                 kmult=1.,
                 relin_freq=1,
                 two_class=False):

        super(SegmentationObservationFactor, self).__init__(sigma,
                                                            obs=dense_labels,
                                                            var_edges=var_edges,
                                                            init_lin_point=[dense_labels],
                                                            N_rob=N_rob,
                                                            rob_type=rob_type,
                                                            kmult=kmult,
                                                            relin_freq=relin_freq,
                                                            mask=None)
        self.mask = self.make_mask(sparse_labels, dense_labels)

        # Reformat labels: for obs pixels 1. for observed class, -1. for other classes
        # 0 for unobs pixels
        fmask = tf.cast(self.mask, dense_labels.dtype)
        self._obs = fmask * (dense_labels * 2. - 1.) + (1. - fmask) * tf.zeros_like(dense_labels)
        if two_class:
            self._obs = self._obs[..., :-1]
            self.mask = self.mask[..., :-1]
        self._set_eta_Lambda()

    def make_mask(self, sparse, dense):
        n_classes = dense.shape[-1]
        mask = tf.zeros(dense.shape)
        mask = tf.tensor_scatter_nd_update(mask,
                                           sparse[..., :-1],
                                           tf.broadcast_to([[1.]], (sparse.shape[0], n_classes)))
        return tf.cast(mask, tf.bool)


class SoftmaxSegmentationObservationFactor(SegmentationObservationFactor):
    def __init__(self,
                 sigma,
                 sparse_labels,
                 dense_labels,
                 var_edges,
                 N_rob=None,
                 rob_type='tukey',
                 kmult=1.,
                 relin_freq=1):
        super(SoftmaxSegmentationObservationFactor, self).__init__(sigma,
                                                                   sparse_labels,
                                                                   dense_labels,
                                                                   var_edges=var_edges,
                                                                   N_rob=N_rob,
                                                                   rob_type=rob_type,
                                                                   kmult=kmult,
                                                                   relin_freq=relin_freq)
        self._obs = self.make_one_hot_obs(sparse_labels)

    def make_one_hot_obs(self, sparse_labels):
        dense_labels = tf.zeros_like(self._obs)
        dense_labels = tf.tensor_scatter_nd_update(dense_labels,
                                                   sparse_labels,
                                                   tf.broadcast_to([1.], [sparse_labels.shape[0]]))
        return dense_labels

    @staticmethod
    def compute_J(segment_vars):
        # Compute Jacobian
        sm = tf.nn.softmax(segment_vars)
        corr = sm[..., None] * sm[..., None, :]
        J = tf.linalg.diag(sm) - corr
        return J

    @staticmethod
    def h(segment_vars):
        return tf.math.softmax(segment_vars, axis=-1)

    def get_eta_Lambda(self, conn_vars=None, **kwargs):
        segment_vars = self.var0[0]
        J = self.compute_J(segment_vars)
        eta = tf.reduce_sum(J * (tf.reduce_sum(J * segment_vars[..., None, :], axis=-1) +
                                 self.obs - self.h(segment_vars))[..., None, :], axis=-1) / self.sigma ** 2.
        Lambda = tf.matmul(J, J) / self.sigma ** 2.

        if self.N_rob is not None:
            E = self.energy(self.var0, robust=False, aggregate=False)
            k = self.get_k_robust(E)
            return eta * k, Lambda * k[..., None]
        else:
            return eta, Lambda

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        # Collect incoming messages
        var_to_fac_eta = self.var_edges.var_to_fac_eta
        var_to_fac_Lambda = self.var_edges.var_to_fac_Lambda

        # Computer factor eta and Lambda
        factor_eta, factor_Lambda = self.get_eta_Lambda(conn_vars)

        # # Lambda + mess may not be invertible for pixels without seg obs
        # # As we'll zero out these messages anyway - replace their Lambdas with something def invertible
        fmask = tf.cast(self.mask, factor_Lambda.dtype)[..., None]
        masked_Lambda = tf.eye(factor_Lambda.shape[-1])[None, None, None] - tf.linalg.diag(var_to_fac_Lambda)
        factor_Lambda = fmask * factor_Lambda + (1. - fmask) * masked_Lambda

        fac_to_var_eta, fac_to_var_Lambda =\
            self.marginalise(factor_plus_mess_eta=var_to_fac_eta + factor_eta,
                             factor_plus_mess_Lambda=tf.linalg.diag(var_to_fac_Lambda) + factor_Lambda,
                             factor_eta=factor_eta,
                             factor_Lambda=factor_Lambda)

        if self.mask is not None:
            # Set messages from pixels without segmentation observations to zero
            fac_to_var_eta *= tf.cast(self.mask, tf.float32)
            fac_to_var_Lambda *= tf.cast(self.mask, tf.float32)

        self.var_edges.fac_to_var_eta = fac_to_var_eta
        self.var_edges.fac_to_var_Lambda = fac_to_var_Lambda

    def energy(self, conn_vars, robust=None, aggregate=True):
        segment_vars = conn_vars[0] if conn_vars is not None else self.var0[0]
        E = (self.obs - self.h(segment_vars)) ** 2. / self.sigma ** 2.
        if robust:
            E = self._robust_correct_energy(E)
        if self.mask is not None:
            # Zero-out energies in inpainting region
            E *= tf.cast(self.mask, tf.float32)
        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E


class BinarySegmentationObservationFactor(SoftmaxSegmentationObservationFactor):
    def __init__(self,
                 sigma,
                 sparse_labels,
                 dense_labels,
                 var_edges,
                 N_rob=None,
                 rob_type='tukey',
                 kmult=1.,
                 relin_freq=1):
        super(BinarySegmentationObservationFactor, self).__init__(sigma,
                                                                   sparse_labels,
                                                                   dense_labels,
                                                                   var_edges=var_edges,
                                                                   N_rob=N_rob,
                                                                   rob_type=rob_type,
                                                                   kmult=kmult,
                                                                   relin_freq=relin_freq)
        self._obs = self._obs[..., 0][..., None]
        self.var0[0] = self.var0[0][..., 0][..., None]
        self.mask = self.mask[..., 0]

    @staticmethod
    def compute_J(segment_vars):
        # Compute Jacobian
        sm = tf.nn.sigmoid(segment_vars)
        J = sm * (1. - sm)
        return J

    @staticmethod
    def h(segment_vars):
        return tf.math.sigmoid(segment_vars)

    def get_eta_Lambda(self, conn_vars=None, **kwargs):
        segment_vars = self.var0[0]
        J = self.compute_J(segment_vars)
        eta = J * (J * segment_vars + self.obs - self.h(segment_vars)) / self.sigma ** 2.

        Lambda = J ** 2. / self.sigma ** 2.
        if self.N_rob is not None:
            E = self.energy(self.var0, robust=False, aggregate=False)
            k = self.get_k_robust(E)
            return eta * k, Lambda * k
        else:
            return eta, Lambda

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):

        # Computer factor eta and Lambda
        factor_eta, factor_Lambda = self.get_eta_Lambda(conn_vars)

        # Unary factor so no marginalisation needed
        fac_to_var_eta, fac_to_var_Lambda = factor_eta, factor_Lambda
            # self.marginalise(factor_plus_mess_eta=var_to_fac_eta + factor_eta,
            #                  factor_plus_mess_Lambda=tf.linalg.diag(var_to_fac_Lambda) + factor_Lambda,
            #                  factor_eta=factor_eta,
            #                  factor_Lambda=factor_Lambda)

        if self.mask is not None:
            # Set messages from pixels without segmentation observations to zero
            fac_to_var_eta *= tf.cast(self.mask, tf.float32)[..., None]
            fac_to_var_Lambda *= tf.cast(self.mask, tf.float32)[..., None]

        self.var_edges.fac_to_var_eta = fac_to_var_eta
        self.var_edges.fac_to_var_Lambda = fac_to_var_Lambda

    def energy(self, conn_vars, robust=None, aggregate=True):
        segment_vars = conn_vars[0] if conn_vars is not None else self.var0[0]
        E = (self.obs - self.h(segment_vars)) ** 2. / self.sigma ** 2.
        if robust:
            E = self._robust_correct_energy(E)
        if self.mask is not None:
            # Zero-out energies in inpainting region
            E *= tf.cast(self.mask, tf.float32)[..., None]

        if aggregate:
            E = tf.reduce_sum(E)
            return E
        else:
            return E
