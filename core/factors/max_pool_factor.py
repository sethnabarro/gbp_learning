# coding=utf-8
import numpy as np
import tensorflow as tf

from core.factors.base_nonlin import NonLinearFactor
from core.factors.avg_pool_factor import AvgPoolFactor
from core.utils.utils import patchify_image


class MaxPoolFactor(NonLinearFactor, AvgPoolFactor):
    def __init__(self, sigma, input_var_edges, output_var_edges, init_lin_point, N_rob=None,
                 rob_type='tukey', relin_freq=1, kmult=1.):
        NonLinearFactor.__init__(self,
                                 sigma=sigma,
                                 init_lin_point=init_lin_point,
                                 N_rob=N_rob,
                                 rob_type=rob_type,
                                 relin_freq=relin_freq,
                                 kmult=kmult)
        AvgPoolFactor.__init__(self,
                               sigma=sigma,
                               input_var_edges=input_var_edges,
                               output_var_edges=output_var_edges,
                               N_rob=N_rob,
                               rob_type=rob_type,
                               kmult=kmult)
        self.fix_attr_fields()

    def get_eta_J(self):
        """
        Factor energy:
            E(x, y) = ((max(x) - y) / σ)^2
            where
                x are input variables within receptive field
                y is the output variable for that field

        Measurement function:
            h(x, y) = max(x) - y

        Pseudomeasurement:
            z = 0

        Jacobian:
            \partial h / \partial x = \mathbb{1}_{m}
            \partial h / \partial y = -1
            where
                m = argmax x
        """
        input_vars, output_vars = self.var0

        # Jacobian for inputs is 1 for max elem in recep field, else 0
        # Note the method below will take first max along -1 axis - ties ignored
        input_patches = patchify_image(input_vars, ksize_x=self.ksize, stride=self.ksize)
        argmax_patches = tf.argmax(input_patches, axis=-1)
        argmax_bool = \
            tf.range(self.ksize ** 2, dtype=argmax_patches.dtype)[None, None, None, None] == \
            argmax_patches[..., None]
        J_in = tf.where(argmax_bool, 1., 0.)

        # Jacobian for outputs is -1.
        J_out = -tf.ones_like(output_vars)[..., None]

        J = tf.concat([J_in, J_out], axis=-1)

        # Eta is 0 as (max_pool(ins) - outs) should be zero
        eta = tf.zeros_like(J)
        return eta, J

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        factor_eta, factor_J = self.get_eta_J()

        # Gather var to factor messages
        var_msg_in_eta, var_msg_in_Lambda = self._get_incoming_messages()

        # Do marginalisation to get outgoing messages
        fac_to_var_eta, fac_to_var_Lambda = \
            self.marginalise_sherman_morrison(mess_eta=tf.cast(var_msg_in_eta, tf.float64),
                                              factor_eta=tf.cast(factor_eta, tf.float64),
                                              mess_Lambda=tf.cast(var_msg_in_Lambda, tf.float64),
                                              J_div_sigma=tf.cast(factor_J / self.sigma, tf.float64))

        # Update outgoing message state
        self.input_var_edges.fac_to_var_eta = tf.cast(fac_to_var_eta[..., :self.ksize ** 2], tf.float32)
        self.input_var_edges.fac_to_var_Lambda = tf.cast(fac_to_var_Lambda[..., :self.ksize ** 2], tf.float32)
        self.output_var_edges.fac_to_var_eta = tf.cast(fac_to_var_eta[..., -1], tf.float32)
        self.output_var_edges.fac_to_var_Lambda = tf.cast(fac_to_var_Lambda[..., -1], tf.float32)

    def energy(self, conn_vars, robust=None, aggregate=True):
        inputs, coeffs = conn_vars
        input_pool = tf.nn.max_pool2d(inputs, ksize=self.ksize, padding='VALID', strides=self.ksize)
        E = ((input_pool - coeffs) / self.sigma) ** 2.
        if robust is None:
            robust = self.N_rob is not None
        if robust and self.N_rob is not None:
            E = self._robust_correct_energy(E)
        E = E[..., None]

        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E
