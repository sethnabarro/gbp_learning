# coding=utf-8
import tensorflow as tf

from core.factors.base_nonlin import NonLinearFactor
from core.utils.utils import patchify_image


class PairwiseSmoothingFactorDecomp(NonLinearFactor):
    def __init__(self,   # Most of these inputs not use but allow polymorphism with the more complex factor types
                 sigma,
                 input_var_edges,
                 n_filters,
                 coeff_var_edges=None,
                 bias_var_edges=None,
                 filter_var_edges=None,
                 weight_var_edges=None,
                 component_var_edges=None,
                 sum_before_nonlin=False,
                 fac_to_var_chunksize=None,
                 coeff_padding=None,
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
                 nonlin_xscale=None,
                 nonlin_yscale=None):
        super(PairwiseSmoothingFactorDecomp, self).__init__(
            sigma=sigma,
            N_rob=N_rob,
            rob_type=rob_type,
            kmult=kmult,
            init_lin_point=init_lin_point,
            relin_freq=relin_freq)
        self.input_var_edges = input_var_edges
        self.relative_to_centre = True
        self.receptive_field = rec_field
        self.stride = stride
        self.use_component_vars = False

    def get_eta_J(self, conn_vars, **kwargs):
        J = tf.convert_to_tensor([1., -1.])
        eta = tf.zeros_like(J[0])

        pixels_mu = self.var0[1]
        input_patch_shp = patchify_image(image=pixels_mu,
                                         ksize_y=self.receptive_field[0],
                                         ksize_x=self.receptive_field[1],
                                         stride=self.stride).shape
        eta = tf.broadcast_to(eta[None, None, None, None, None], input_patch_shp.as_list()[:-1] + [4, 2])
        J = tf.broadcast_to(J[None, None, None, None, None], input_patch_shp.as_list()[:-1] + [4, 2])

        if self.N_rob is not None:
            E = self.energy(self.var0,
                            robust=False,
                            aggregate=False)
            k = self.get_k_robust(E, N_rob=self.N_rob)
            eta *= k
            J *= tf.sqrt(k)
        return eta, J

    def energy(self, conn_vars, robust=None, aggregate=True):
        pixels_mu = conn_vars[1]
        ksize = self.receptive_field[0] * self.receptive_field[1]
        input_patches = patchify_image(image=pixels_mu,
                                       ksize_y=self.receptive_field[0],
                                       ksize_x=self.receptive_field[1],
                                       stride=self.stride)
        sqdiffs = (input_patches[..., int(ksize / 2):int(ksize / 2) + 1] - input_patches[..., 1::2]) ** 2.
        E = sqdiffs / self.sigma ** 2
        if robust is None:
            robust = self.N_rob is not None
        if robust and self.N_rob is not None:
            E = self._robust_correct_energy(E)
        E = E[..., None]

        if aggregate:
            return tf.reduce_sum(E)
        else:
            return E

    def _get_incoming_messages(self):
        msgs_combined = []
        ks2 = self.input_var_edges.fac_to_var_eta.shape[-2]
        for mtype in ('eta', 'Lambda'):
            msg_in = getattr(self.input_var_edges, f'var_to_fac_{mtype}')
            msg_in *= tf.cast(tf.math.mod(tf.range(ks2), 2), msg_in.dtype)[None, None,
                                 None, None, :, None]

            msgs_combined.append(msg_in)
        return msgs_combined

    def update_outgoing_messages(self, conn_vars, **kwargs):
        var_msg_in_eta, var_msg_in_Lambda = self._get_incoming_messages()

        factor_eta, factor_J = self.get_eta_J(conn_vars)
        original_fac_shp = var_msg_in_eta.shape[1:]

        var_msg_in_eta = var_msg_in_eta[..., 1::2, :]
        var_msg_in_Lambda = var_msg_in_Lambda[..., 1::2, :]

        factor_to_var_eta, factor_to_var_Lambda = \
            self.marginalise_sherman_morrison(var_msg_in_eta, factor_eta, var_msg_in_Lambda, factor_J / self.sigma)

        factor_to_var_eta_sparse = tf.zeros(original_fac_shp)
        factor_to_var_eta = tf.where((tf.math.mod(tf.range(original_fac_shp[-2]), 2) == 1)[..., None],
                                     tf.concat([tf.repeat(factor_to_var_eta, 2, axis=-2), tf.zeros_like(factor_to_var_eta[..., :1, :])], axis=-2),
                                     factor_to_var_eta_sparse)
        factor_to_var_Lambda_sparse = tf.zeros(original_fac_shp)
        factor_to_var_Lambda = tf.where((tf.math.mod(tf.range(original_fac_shp[-2]), 2) == 1)[..., None],
                                     tf.concat([tf.repeat(factor_to_var_Lambda, 2, axis=-2), tf.zeros_like(factor_to_var_Lambda[..., :1, :])], axis=-2),
                                     factor_to_var_Lambda_sparse)

        self.update_outgoing_edge_messages(factor_to_var_eta, factor_to_var_Lambda)

    def update_outgoing_edge_messages(self, factor_to_var_eta, factor_to_var_Lambda):
        ninps = 2  # Number of input vars connected to factor
        ksize = self.receptive_field[0] * self.receptive_field[1]

        # Zero out messages not to NESW pixels
        factor_to_var_eta *= tf.cast(tf.math.mod(tf.range(ksize), 2), factor_to_var_eta.dtype)[None, None, None, None, :, None]
        factor_to_var_Lambda *= tf.cast(tf.math.mod(tf.range(ksize), 2), factor_to_var_eta.dtype)[None, None, None, None, :, None]

        self.input_var_edges.fac_to_var_eta = factor_to_var_eta
        self.input_var_edges.fac_to_var_Lambda = factor_to_var_Lambda

    def get_edge_messages(self, named=False):
        attr_to_get = 'named_state' if named else 'state'
        edges = [getattr(self.input_var_edges, attr_to_get)]
        enames = [str(self.input_var_edges)]
        if named:
            edges = list(zip(enames, edges))
        return edges

    def set_edge_messages(self, edges):
        self.input_var_edges.state, = edges
        return edges
