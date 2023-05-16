# coding=utf-8
from core.inference.gbp.layers.base import GBPLayer


class GBPLayerEmpty(GBPLayer):
    """Useful for constructing customisable layers on the fly, which can be linked with other layers"""
    def _update_input_marginals(self, return_eta_Lambda=False):
        pass

    def _update_coeff_marginals(self, return_eta_Lambda=False):
        pass

    def intra_layer_inference_iter(self, itr):
        pass

    def update_marginals(self):
        pass

    def energy(self, *args, **kwargs):
        pass

    @property
    def state(self):
        pass

    @state.setter
    def state(self, new_state):
        pass

    @property
    def named_state(self):
        pass

    @property
    def variables(self):
        pass

    @property
    def params(self):
        pass

    @property
    def edges(self):
        pass
