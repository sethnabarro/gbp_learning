# coding=utf-8
import tensorflow as tf

from core.inference.gbp.layers.base import GBPLayer


class GBPInputObsLayer(GBPLayer):
    def __init__(self, pixel_obs_factor, input_vars):
        super(GBPInputObsLayer, self).__init__(coeff_vars=input_vars,
                                               input_vars=None)
        self.pixel_obs_factor = pixel_obs_factor

    def intra_layer_inference_iter(self, itr):
        self.pixel_obs_factor.update_outgoing_messages([self.coeff_vars.mu])

    def _update_input_marginals(self, return_eta_Lambda: bool = False, **kwargs):
        # No inputs to update
        pass

    def _update_coeff_marginals(self, return_eta_Lambda: bool = False, **kwargs):
        to_return = []
        for ptype in ('eta', 'Lambda'):
            pix_obs_msg = getattr(self.pixel_obs_factor.var_edges, f"fac_to_var_{ptype}")
            if return_eta_Lambda:
                to_return.append(pix_obs_msg)
            else:
                setattr(self.coeff_vars, ptype, pix_obs_msg)
            setattr(self.coeff_vars, ptype, pix_obs_msg)
        if return_eta_Lambda:
            return to_return

    def update_marginals(self):
        self.update_coeff_marginals()

    @property
    def state(self):
        state = [[self.coeff_vars.eta, self.coeff_vars.Lambda]]
        state.append(self.pixel_obs_factor.state)
        return state

    @property
    def named_state(self):
        state = [('coeff_marginals', [self.coeff_vars.eta, self.coeff_vars.Lambda])]
        state.append(('pixel_obs_factor', self.pixel_obs_factor.named_state))
        return state

    @state.setter
    def state(self, new_state):
        self.coeff_vars.eta, self.coeff_vars.Lambda = new_state[0]
        self.pixel_obs_factor.state = new_state[1]

    @property
    def variables(self):
        variables = (self.coeff_vars,)
        return variables

    @property
    def params(self):
        return ()

    def relinearise_factors(self, itr):
        relin_points = self.linearisation_points
        self._relinearise_factors(itr, relin_points)

    def _relinearise_factors(self, itr, relin_points):
        self.pixel_obs_factor.relinearise(itr, relin_points['pixel_obs'])

    @property
    def linearisation_points(self):
        relin_points = dict()   # Mapping factor_name -> factor_relin_point
        relin_points['pixel_obs'] = [self.coeff_vars.mu]
        return relin_points

    @linearisation_points.setter
    def linearisation_points(self, new_relin_points):
        # Give i=0 to force relinearisation
        self._relinearise_factors(itr=0, relin_points=new_relin_points)

    def energy(self, as_numpy=False, robust=False, sum_all=True, inputs=None):
        inputs = inputs if inputs is not None else self.coeff_vars.mu
        energies = dict()
        energies['pixel_obs'] = self.pixel_obs_factor.energy([inputs], robust=robust, aggregate=True)
        if as_numpy:
            energies = {n: e.numpy() for n, e in energies.items()}
        if sum_all:
            return tf.reduce_sum(list(energies.values()))
        else:
            return energies

    @property
    def edges(self):
        edges = [self.pixel_obs_factor.var_edges]
        return edges
