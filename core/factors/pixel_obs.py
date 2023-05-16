# coding=utf-8
import tensorflow as tf

from core.factors.unary_factor import NonLinearUnaryFactor


class PixelObservationFactor(NonLinearUnaryFactor):
    def __init__(self,
                 sigma,
                 obs,
                 var_edges,
                 init_lin_point,
                 N_rob=None,
                 rob_type='tukey',
                 kmult=1.,
                 relin_freq=1,
                 mask=None,
                 mask_prec=None):
        # TODO: write method to assert that `mask_prec` is zero (or close) if
        # TODO: masked region is in centre of image (not at border).
        # TODO: In this case, likely to be doing inpainting rather than padding
        # TODO: and setting non-zero `mask_prec` could be cheating!
        self.mask = mask
        self.mask_prec = mask_prec or 0.
        super(PixelObservationFactor, self).__init__(sigma,
                                                     obs,
                                                     var_edges,
                                                     init_lin_point,
                                                     N_rob=N_rob,
                                                     rob_type=rob_type,
                                                     kmult=kmult,
                                                     relin_freq=relin_freq)

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        """Unary, so don't need to deal with incoming messages"""
        eta, Lambda = self.get_eta_Lambda(conn_vars)
        if self.mask is not None:
            # Weaken messages from mask region. WIll be zero if self.mask_prec == 0.
            bmask = tf.cast(self.mask, tf.bool)
            Lambda = tf.where(bmask, Lambda, self.mask_prec)
            eta = tf.where(bmask, eta, self.obs * self.mask_prec)
        self.var_edges.fac_to_var_eta = eta
        self.var_edges.fac_to_var_Lambda = Lambda

    def energy(self, conn_vars, robust=None, aggregate=True):
        E = super().energy(conn_vars, aggregate=False, robust=robust)
        if self.mask is not None:
            fmask = tf.cast(self.mask, tf.float32)
            if self.mask_prec > 0.:
                E_mask = (conn_vars[0] - self.obs) ** 2 * self.mask_prec
                if robust:
                    E_mask = self._robust_correct_energy(E_mask)
                E = E * fmask + (1. - fmask) * E_mask
            else:
                E *= fmask
        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E
