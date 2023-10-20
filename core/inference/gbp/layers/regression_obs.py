# coding=utf-8
import tensorflow as tf

from core.inference.gbp.layers.base import GBPLayer


class GBPRegressionObsLayer(GBPLayer):
    def __init__(self, regression_obs_factor, input_vars):
        super(GBPRegressionObsLayer, self).__init__(coeff_vars=None,
                                                    input_vars=input_vars)
        self.regression_obs_factor = regression_obs_factor

    def intra_layer_inference_iter(self, itr):
        self.regression_obs_factor.update_outgoing_messages([self.input_vars.mu])

    def _update_input_marginals(self, return_eta_Lambda: bool = False, **kwargs):
        to_return = []
        for ptype in ('eta', 'Lambda'):
            inp_obs_msg = getattr(self.regression_obs_factor.var_edges, f"fac_to_var_{ptype}")
            if return_eta_Lambda:
                to_return.append(inp_obs_msg)
            else:
                setattr(self.input_vars, ptype, inp_obs_msg)
            setattr(self.input_vars, ptype, inp_obs_msg)
        if return_eta_Lambda:
            return to_return

    def _update_coeff_marginals(self, return_eta_Lambda: bool = False, **kwargs):
        pass  # No coeffs here

    def update_marginals(self):
        self.update_input_marginals()

    @property
    def state(self):
        state = [[self.input_vars.eta, self.input_vars.Lambda]]
        state.append(self.regression_obs_factor.state)
        return state

    @property
    def named_state(self):
        state = [('coeff_marginals', [self.input_vars.eta, self.input_vars.Lambda])]
        state.append(('pixel_obs_factor', self.regression_obs_factor.named_state))
        return state

    @state.setter
    def state(self, new_state):
        self.input_vars.eta, self.input_vars.Lambda = new_state[0]
        self.regression_obs_factor.state = new_state[1]

    @property
    def variables(self):
        variables = (self.input_vars,)
        return variables

    @property
    def params(self):
        return ()

    def relinearise_factors(self, itr):
        relin_points = self.linearisation_points
        self._relinearise_factors(itr, relin_points)

    def _relinearise_factors(self, itr, relin_points):
        self.regression_obs_factor.relinearise(itr, relin_points['pixel_obs'])

    @property
    def linearisation_points(self):
        relin_points = dict()   # Mapping factor_name -> factor_relin_point
        relin_points['regression_obs'] = [self.input_vars.mu]
        return relin_points

    @linearisation_points.setter
    def linearisation_points(self, new_relin_points):
        # Give i=0 to force relinearisation
        self._relinearise_factors(itr=0, relin_points=new_relin_points)

    def energy(self, as_numpy=False, robust=False, sum_all=True, inputs=None):
        inputs = inputs if inputs is not None else self.input_vars.mu
        energies = dict()
        energies['pixel_obs'] = self.regression_obs_factor.energy([inputs], robust=robust, aggregate=True)
        if as_numpy:
            energies = {n: e.numpy() for n, e in energies.items()}
        if sum_all:
            return tf.reduce_sum(list(energies.values()))
        else:
            return energies

    @property
    def edges(self):
        edges = [self.regression_obs_factor.var_edges]
        return edges
