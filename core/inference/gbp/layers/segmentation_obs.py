# coding=utf-8
import tensorflow as tf

from core.inference.gbp.layers.base import GBPLayer
from core.factors import SegmentationObservationFactor
from core.variables import CoeffVariable, PixelVariable


class GBPSegmentationObservationLayer(GBPLayer):
    def __init__(self,
                 segmentation_factor: SegmentationObservationFactor,
                 input_vars: [CoeffVariable, PixelVariable]):
        # No output/coeff variables: softmax head at last layer
        super(GBPSegmentationObservationLayer, self).__init__(input_vars=input_vars,
                                                              coeff_vars=None)
        self.segmentation_factor = segmentation_factor

    def _update_input_marginals(self, return_eta_Lambda=False):
        eta = self.segmentation_factor.var_edges.fac_to_var_eta
        Lambda = self.segmentation_factor.var_edges.fac_to_var_Lambda

        if return_eta_Lambda:
            return eta, Lambda
        else:
            self.input_vars.eta = eta
            self.input_vars.Lambda = Lambda

    def _update_coeff_marginals(self, return_eta_Lambda=False):
        raise AttributeError('Segmentation observation layer has no coefficient (output) variables')

    def intra_layer_inference_iter(self, itr):
        self.update_marginals()
        self.update_input_vars_to_factor_message()
        self.update_factor_to_input_vars_message()
        self.update_marginals()
        self.relinearise_factors(itr)

    def update_input_vars_to_factor_message(self):
        for mtype in ('eta', 'Lambda'):
            marg = getattr(self.input_vars, mtype)
            fac_to_var = getattr(self.segmentation_factor.var_edges, f'fac_to_var_{mtype}')
            setattr(self.segmentation_factor.var_edges,
                    f'var_to_fac_{mtype}',
                    marg - fac_to_var)

    def update_factor_to_input_vars_message(self):
        self.segmentation_factor.update_outgoing_messages(self.segmentation_factor.var0)

    def update_marginals(self):
        self.update_input_marginals()

    def energy(self, logits=None, robust=None, as_numpy=False, sum_all=False):
        inputs = logits if logits is not None else self.input_vars.mu
        energies = {'segmentation_obs': self.segmentation_factor.energy([inputs], robust=robust, aggregate=True)}
        if as_numpy:
            energies = {n: e.numpy() for n, e in energies.items()}
        if sum_all:
            return tf.reduce_sum(list(energies.values()))
        else:
            return energies

    @property
    def state(self):
        state = [[self.input_vars.eta, self.input_vars.Lambda]]
        state.append(self.segmentation_factor.state)
        return state

    @state.setter
    def state(self, new_state):
        self.input_vars.eta, self.input_vars.Lambda = new_state[0]
        self.segmentation_factor.state = new_state[1]

    def relinearise_factors(self, itr):
        relin_points = self.linearisation_points
        self._relinearise_factors(itr, relin_points)

    def _relinearise_factors(self, itr, relin_points):
        self.segmentation_factor.relinearise(itr, relin_points['softmax'])

    @property
    def linearisation_points(self):
        return dict(softmax=[self.input_vars.mu])

    @linearisation_points.setter
    def linearisation_points(self, new_lin_points):
        self._relinearise_factors(itr=0, relin_points=new_lin_points)

    @property
    def variables(self):
        return self.input_vars,

    @property
    def params(self):
        return ()

    @property
    def named_state(self):
        state = [('input_marginals', [self.input_vars.eta, self.input_vars.Lambda])]
        state.append(('segmentation_factor', self.segmentation_factor.named_state))
        return state

    @property
    def edges(self):
        return [self.segmentation_factor.var_edges]
