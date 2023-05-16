# coding=utf-8
from core.variables.base import Variable


class BiasVariable(Variable):
    def __init__(self, mu_init, sigma_init):
        super(BiasVariable, self).__init__(mu_init, sigma_init)
        self.fix_attr_fields()
