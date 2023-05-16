# coding=utf-8
from core.variables.base import Variable


class ComponentVariable(Variable):
    def __init__(self, mu_init, sigma_init):
        super(ComponentVariable, self).__init__(mu_init, sigma_init)
        self.fix_attr_fields()
