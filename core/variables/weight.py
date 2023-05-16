# coding=utf-8
from core.variables.base import Variable


class WeightVariable(Variable):
    def __init__(self, mu_init, sigma_init):
        super(WeightVariable, self).__init__(mu_init, sigma_init)
        self.fix_attr_fields()
