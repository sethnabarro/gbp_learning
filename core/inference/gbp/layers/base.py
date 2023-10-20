# coding=utf-8
from abc import ABC, abstractmethod
import tensorflow as tf

from core.inference.base import Layer
from core.variables import CoeffVariable, PixelVariable


class GBPLayer(Layer, ABC):
    def __init__(self, input_vars: [PixelVariable, CoeffVariable], coeff_vars: [CoeffVariable, None]):
        self.update_input_marginals = self._update_input_marginals
        self.update_coeff_marginals = self._update_coeff_marginals
        self.input_vars = input_vars
        self.coeff_vars = coeff_vars

    def link_layer_before(self, layer_before):
        def _update_input_marginals(**kwargs):
            # Coeffs in previous layer are inputs to this layer
            # First get bottom up message from previous layer
            eta_before, Lambda_before = layer_before._update_coeff_marginals(return_eta_Lambda=True, **kwargs)

            # Top down from this layer
            eta_here, Lambda_here = self._update_input_marginals(return_eta_Lambda=True, **kwargs)

            self.input_vars.eta = eta_before + eta_here
            self.input_vars.Lambda = Lambda_before + Lambda_here
            tf.assert_equal(self.input_vars.eta, layer_before.coeff_vars.eta)
            tf.assert_equal(self.input_vars.Lambda, layer_before.coeff_vars.Lambda)
        layer_before.update_coeff_marginals = _update_input_marginals
        self.update_input_marginals = _update_input_marginals

    @abstractmethod
    def _update_input_marginals(self, return_eta_Lambda=False, **kwargs):
        raise NotImplementedError('Implemented in child class')

    @abstractmethod
    def _update_coeff_marginals(self, return_eta_Lambda=False, **kwargs):
        raise NotImplementedError('Implemented in child class')

    @abstractmethod
    def intra_layer_inference_iter(self, itr):
        raise NotImplementedError('Implemented in child class')

    @abstractmethod
    def update_marginals(self):
        raise NotImplementedError('Implemented in child class')

    @abstractmethod
    def energy(self, *args, **kwargs):
        raise NotImplementedError('Implemented in child class')

    @property
    @abstractmethod
    def state(self):
        raise NotImplementedError('Implemented in child class')

    @state.setter
    @abstractmethod
    def state(self, new_state):
        raise NotImplementedError('Implemented in child class')

    @property
    @abstractmethod
    def named_state(self):
        raise NotImplementedError('Implemented in child class')

    @property
    @abstractmethod
    def edges(self):
        raise NotImplementedError('Implemented in child class')

    @property
    def variables(self):
        raise NotImplementedError('Implemented in child class')

    @property
    def params(self):
        raise NotImplementedError('Implemented in child class')

