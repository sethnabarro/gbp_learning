# coding=utf-8
import tensorflow as tf

from core.factors.base_nonlin import NonLinearFactor
from core.utils.utils import flatten_filters


class DiversityFactor(NonLinearFactor):
    N_PARALLEL_PROC_MARG = 1
    def __init__(self, sigma, var_edges,
                 N_rob=None,
                 rob_type='tukey',
                 relin_freq=1,
                 lin_point=None,
                 multidim_filters=False):
        super(DiversityFactor, self).__init__(sigma, lin_point, relin_freq, N_rob, rob_type)
        self.var_edges = var_edges
        self.multidim_filters = multidim_filters
        self._set_eta_Lambda()
        self.fix_attr_fields()

    def _set_eta_Lambda(self):
        """
        Non-linear factor, so will need current
        values of filters to set variables
        """
        self._eta = None
        self._Lambda = None

    def _get_incoming_messages(self):
        n_vars = self.var_edges.var_to_fac_eta.shape[0]
        msgs_in_eta_Lambda = []
        for etype in ('eta', 'Lambda'):
            msgs_in = []
            msgs_in.append(getattr(self.var_edges, f'var_to_fac_{etype}'))
            msgs_in_eta_Lambda.append(tf.reshape(msgs_in, (n_vars, n_vars, -1)))
        return msgs_in_eta_Lambda

    def update_outgoing_messages(self, conn_vars, **kw_msgs_in):
        factor_eta, factor_Lambda = self.get_eta_Lambda(conn_vars)
        n_filts = factor_Lambda.shape[0]
        incoming_eta, incoming_Lambda = self._get_incoming_messages()
        factor_eta = tf.convert_to_tensor([factor_eta[i, j] for j in range(n_filts) for i in range(n_filts) if j > i])
        factor_Lambda = tf.convert_to_tensor([factor_Lambda[i, j] for j in range(n_filts) for i in range(n_filts) if j > i])
        # print(tf.reduce_sum(factor_Lambda, [1, 2]), 'FL')
        # afds
        incoming_eta = tf.convert_to_tensor([incoming_eta[i, j] for j in range(n_filts) for i in range(n_filts) if j > i])
        incoming_Lambda = tf.convert_to_tensor([incoming_Lambda[i, j] for j in range(n_filts) for i in range(n_filts) if j > i])

        eta_fact_plus_msg = factor_eta + incoming_eta
        Lambda_fact_plus_msg = factor_Lambda + tf.linalg.diag(incoming_Lambda)
        if self.multidim_filters:
            factor_to_var_eta, factor_to_var_Lambda = \
                self.marginalise_multidim(eta_fact_plus_msg,
                                 Lambda_fact_plus_msg,
                                 factor_eta,
                                 factor_Lambda)
        else:
            factor_to_var_eta, factor_to_var_Lambda = \
                self.marginalise(eta_fact_plus_msg,
                                 Lambda_fact_plus_msg,
                                 factor_eta,
                                 factor_Lambda)
        self.var_edges.fac_to_var_eta = tf.convert_to_tensor([[factor_to_var_eta[i * n_filts + j - int((i + 1) * (i + 2) / 2)] if j > i else tf.zeros_like(factor_to_var_eta[0])
                                                  for j in range(n_filts)]
                                                  for i in range(n_filts)])

        self.var_edges.fac_to_var_Lambda = tf.convert_to_tensor([[factor_to_var_Lambda[i * n_filts + j - int((i + 1) * (i + 2) / 2)] if j > i else tf.zeros_like(factor_to_var_Lambda[0])
                                                    for j in range(n_filts)]
                                                    for i in range(n_filts)])

    def marginalise_multidim(self):
        raise NotImplementedError('Should be implemented in child class')


class DiversityFactorCosineSim(DiversityFactor):
    def energy(self, conn_vars, robust=None, aggregate=True):
        filters_flat = flatten_filters(conn_vars[0])
        ffT = tf.einsum('ab,cb->ac', filters_flat, filters_flat)
        fnorm = tf.norm(filters_flat, axis=-1)
        fnormfnormT = fnorm[None] * tf.transpose(fnorm[None])
        # Cosine similarity
        E = (ffT / (fnormfnormT * self.sigma)) ** 2
        if robust:
            return self._robust_correct_energy(E)
        else:
            return E

    @staticmethod
    def filtersTfilters(filters_flat):
        return tf.einsum('ab,cb->ac', filters_flat, filters_flat)

    @staticmethod
    def fnorm_fnormT(filters_flat):
        fnorm = tf.norm(filters_flat, axis=-1)
        return fnorm[None] * tf.transpose(fnorm[None])

    def get_eta_Lambda(self, conn_vars, **kw_msgs_in):
        filters_flat_0 = flatten_filters(self.var0[0])
        nfilt = filters_flat_0.shape[0]

        f0Tf0 = self.filtersTfilters(filters_flat_0)
        f0normf0normT = self.fnorm_fnormT(filters_flat_0)
        f0norm = tf.norm(filters_flat_0, axis=-1)
        filters0_bc = tf.broadcast_to(filters_flat_0[..., None, :],  (nfilt, nfilt, filters_flat_0.shape[-1]))
        filters0_bcT = tf.transpose(filters0_bc, (1, 0, 2))
        J0 = tf.concat([filters0_bcT - filters0_bc * f0Tf0[..., None] / (f0norm ** 2)[..., None],
                       filters0_bc - filters0_bcT * f0Tf0[..., None] / tf.transpose(tf.broadcast_to(f0norm[..., None] ** 2, (nfilt, nfilt)))[..., None]], axis=-1) / f0normf0normT[..., None]

        varstack = tf.convert_to_tensor([[tf.concat([f1, f2], axis=-1)
                                          for f2 in filters_flat_0[0]]
                                        for f1 in filters_flat_0[0]])

        h = f0Tf0 / f0normf0normT
        J0Tf = tf.reduce_sum(J0 * varstack, axis=-1)
        eta = J0 * (J0Tf - h)[..., None] / self.sigma ** 2

        Lambda = tf.einsum('abc,abd->abcd', J0, J0) / self.sigma ** 2

        if self.N_rob is not None:
            E = self.energy(self.var0, robust=False, aggregate=False)
            k = self.get_k_robust(E, N_rob=self.N_rob)
            eta *= k[..., None]
            Lambda *= k[..., None, None]
        return eta, Lambda

    def marginalise_multidim(self,
                             factor_plus_mess_eta,
                             factor_plus_mess_Lambda,
                             factor_eta,
                             factor_Lambda):
        nfilt = int(factor_eta.shape[-1] * 0.5)
        etas = []
        Lambdas = []
        for start, end in ((0, nfilt), (nfilt, 2 * nfilt)):
            etaA = factor_eta[..., start:end]
            etaB = tf.concat([factor_plus_mess_eta[..., :start],
                              factor_plus_mess_eta[..., end:]], axis=-1)
            LambdaAA = factor_Lambda[..., start:end, start:end]
            LambdaB = tf.concat([factor_plus_mess_Lambda[..., :start],
                                 factor_plus_mess_Lambda[..., end:]], axis=-1)
            LambdaAB = LambdaB[..., start:end, :]
            LambdaBB = tf.concat([LambdaB[..., :start, :],
                                  LambdaB[..., end:, :]], axis=-2)
            LambdaBBinv = tf.linalg.inv(LambdaBB)

            fac_to_var_eta_filt = etaA - tf.einsum('hij,hjk,hk->hi', LambdaAB, LambdaBBinv, etaB)
            fac_to_var_Lambda_filt = LambdaAA - tf.einsum('hij,hjk,hlk->hil', LambdaAB, LambdaBBinv, LambdaAB)
            etas.append(fac_to_var_eta_filt)
            Lambdas.append(fac_to_var_Lambda_filt)

        return tf.concat(etas, axis=-1), tf.concat(Lambdas, axis=-1)


class DiversityFactorRecip(DiversityFactor):

    def energy(self, conn_vars, robust=None, aggregate=True):
        filters_flat = flatten_filters(conn_vars[0])
        norm = tf.norm(filters_flat[None] - filters_flat[:, None], axis=-1)
        E = 1. / (self.sigma * norm) ** 2
        E = tf.where(norm == 0., -1., E)   # Will ignore the diagonal elems anyway
        if robust:
            return self._robust_correct_energy(E)
        else:
            return E

    def get_eta_Lambda(self, conn_vars, **kwargs):
        filters_flat = flatten_filters(conn_vars[0])
        diff = filters_flat[None] - filters_flat[:, None]
        diffnorm = tf.norm(diff, axis=-1)
        J = tf.concat([- diff, diff], axis=-1) / diffnorm[..., None] ** 3
        varstack = tf.convert_to_tensor([[tf.concat([f1, f2], axis=-1) for f1 in filters_flat]
                                        for f2 in filters_flat])
        h = 1 / diffnorm
        JTf = tf.reduce_sum(J * varstack, axis=-1)
        z = 0.
        eta = J * (JTf + z - h)[..., None] / self.sigma ** 2  # pseudo-obs z=0
        Lambda = J[..., None, :] * J[..., None] / self.sigma ** 2

        E = self.energy(conn_vars, robust=False, aggregate=False)
        if self.N_rob is not None:
            k = self.get_k_robust(E, N_rob=self.N_rob)
            eta *= k[..., None]
            Lambda *= k[..., None, None]

        return eta, Lambda


def get_filter_to_filter_diversity_factor_messages(div_fact, filts_marg_eta, filts_marg_Lambda):
    """Input marginals should be flattened filters"""
    nfilt = filts_marg_eta.shape[0]
    filter_marg_eta_upper_tri_f1 = tf.convert_to_tensor([[filts_marg_eta[j] if i > j else tf.zeros_like(filts_marg_eta[j])
                                                       for i in range(nfilt)]
                                                      for j in range(nfilt)])
    filter_marg_eta_upper_tri_f2 = tf.convert_to_tensor([[filts_marg_eta[i] if i > j else tf.zeros_like(filts_marg_eta[j])
                                                       for i in range(nfilt)]
                                                      for j in range(nfilt)])
    filter_marg_eta_upper_tri = tf.concat([filter_marg_eta_upper_tri_f1, filter_marg_eta_upper_tri_f2], axis=-1)

    filter_marg_Lambda_upper_tri_f1 = tf.convert_to_tensor([[filts_marg_Lambda[j] if i > j else tf.zeros_like(filts_marg_Lambda[j])
                                                       for i in range(nfilt)]
                                                      for j in range(nfilt)])
    filter_marg_Lambda_upper_tri_f2 = tf.convert_to_tensor([[filts_marg_Lambda[i] if i > j else tf.zeros_like(filts_marg_Lambda[j])
                                                       for i in range(nfilt)]
                                                      for j in range(nfilt)])
    filter_marg_Lambda_upper_tri = \
        tf.concat([filter_marg_Lambda_upper_tri_f1,
                   filter_marg_Lambda_upper_tri_f2], axis=-1)

    return filter_marg_eta_upper_tri - div_fact.var_edges.fac_to_var_eta,\
        filter_marg_Lambda_upper_tri - div_fact.var_edges.fac_to_var_Lambda
