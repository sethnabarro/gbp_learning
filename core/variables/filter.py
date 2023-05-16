# coding=utf-8
from core.variables.base import Variable

from core.utils.utils import flatten_filters


class FilterVariable(Variable):
    def __init__(self, mu_init, sigma_init):
        super(FilterVariable, self).__init__(mu_init, sigma_init)
        self.in_channels = self.shape[-2]
        self.n_filters = self.shape[-1]
        self.k_size = self.shape[0]
        self.fix_attr_fields()

    @property
    def mu_flat(self):
        return flatten_filters(self.mu)

    @property
    def eta_flat(self):
        return flatten_filters(self._eta)

    @property
    def Lambda_flat(self):
        return flatten_filters(self._Lambda)
