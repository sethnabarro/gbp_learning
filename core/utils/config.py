# coding=utf-8
from core.utils.utils import dotdict



class FactorConfig(dotdict):
    def __init__(self,
                 sigma,
                 mean=None,
                 N_rob=None,
                 rob_type=None,
                 relin_freq=1,
                 kmult=1.):
        super().__init__()
        self.sigma = sigma
        self.N_rob = N_rob
        self.rob_type = rob_type
        self.relin_freq = relin_freq
        self.kmult = kmult
        self.mean = mean


class ReconFactorConfig(FactorConfig):
    def __init__(self,
                 sigma,
                 ksize=3,
                 stride=1,
                 decompose=True,
                 sum_filters=True,
                 relative_to_centre=False,
                 additive_factor=True,
                 feedforward=False,
                 use_bias=True,
                 nonlin=None,
                 nonlin_yscale=1.,
                 nonlin_xscale=1.,
                 N_rob=4.,
                 rob_type='tukey',
                 relin_freq=1,
                 kmult=1.):
        super().__init__(sigma=sigma,
                         N_rob=N_rob,
                         rob_type=rob_type,
                         relin_freq=relin_freq,
                         kmult=kmult)
        self.ksize = ksize
        self.stride = stride
        self.decompose = decompose
        self.sum_filters = sum_filters
        self.relative_to_centre = relative_to_centre
        self.feedforward = feedforward
        self.use_bias = use_bias
        self.nonlin = nonlin
        self.nonlin_yscale = nonlin_yscale
        self.nonlin_xscale = nonlin_xscale
        self.additive_factor = additive_factor


class DenseFactorConfig(FactorConfig):
    def __init__(self,
                 sigma,
                 use_bias=True,
                 N_rob=4.,
                 rob_type='tukey',
                 relin_freq=1,
                 kmult=1.):
        super().__init__(sigma=sigma,
                         N_rob=N_rob,
                         rob_type=rob_type,
                         relin_freq=relin_freq,
                         kmult=kmult)
        self.use_bias = use_bias


class AvgPoolFactorConfig(FactorConfig):
    def __init__(self,
                 sigma,
                 ksize=2,
                 N_rob=4.,
                 rob_type='tukey',
                 relin_freq=1,
                 kmult=1.):
        super().__init__(sigma=sigma,
                         N_rob=N_rob,
                         rob_type=rob_type,
                         relin_freq=relin_freq,
                         kmult=kmult)
        self.ksize = ksize


class AllFactorsConfig(dotdict):
    def __init__(self,
                 recon,
                 weight_prior,
                 coeff_prior,
                 pixel_obs,
                 softmax_class_obs=None,
                 avg_pool=None,
                 bias_prior=None,
                 dense=None):
        """Combines configs for each factor type"""
        super().__init__()
        self.recon = recon
        self.weight_prior = weight_prior
        self.coeff_prior = coeff_prior
        self.pixel_obs = pixel_obs
        self.softmax_class_obs = softmax_class_obs
        self.avg_pool = avg_pool
        self.bias_prior = bias_prior
        self.dense = dense


class GBPConfig(dotdict):
    def __init__(self,
                 n_iters,
                 momentum,
                 dropout=0.):
        super().__init__()
        self.n_iters = n_iters
        self.momentum = momentum
        self.dropout = dropout


class InitConfig(dotdict):
    def __init__(self,
                 init_weight_std,
                 init_coeff_std,
                 weight_init_seed,
                 coeff_init_seed):
        super().__init__()
        self.init_weight_std = init_weight_std
        self.init_coeff_std = init_coeff_std
        self.weight_init_seed = weight_init_seed
        self.coeff_init_seed = coeff_init_seed
