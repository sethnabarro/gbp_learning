# coding=utf-8
import tensorflow as tf

from core.factors import AvgPoolFactor, MaxPoolFactor, NonLinearUnaryFactor
from core.inference.gbp.layers.base import GBPLayer
from core.utils.utils import patchify_image
from core.variables import CoeffVariable, PixelVariable


class GBPPoolLayer(GBPLayer):
    def __init__(self,
                 input_vars: [CoeffVariable, PixelVariable],
                 coeff_vars: CoeffVariable,
                 pool_factor: [AvgPoolFactor, MaxPoolFactor],
                 coeff_prior_factor: NonLinearUnaryFactor):
        # Order of if/elif below is important
        # As MaxPoolFactor inherits from AvgPoolFactor
        if isinstance(pool_factor, MaxPoolFactor):
            self.pool_type = 'max'
        elif isinstance(pool_factor, AvgPoolFactor):
            self.pool_type = 'avg'
        self.pool_factor = pool_factor
        self.coeff_prior_factor = coeff_prior_factor
        GBPLayer.__init__(self,
                          input_vars=input_vars,
                          coeff_vars=coeff_vars)

    def intra_layer_inference_iter(self, itr):
        self.update_input_to_factor_message()
        self.update_coeff_to_factor_message()
        self.update_factor_to_variables_message()
        self.update_marginals()
        self.relinearise_factors(itr)

    def update_input_to_factor_message(self):
        ks = self.pool_factor.ksize
        for mtype in ('eta', 'Lambda'):
            input_margs = getattr(self.input_vars, mtype)
            outgoing = getattr(self.pool_factor.input_var_edges, f'fac_to_var_{mtype}')

            # Reshape input margs to match fac to var messages
            input_margs_rs = patchify_image(input_margs, ksize_x=ks, stride=ks)

            setattr(self.pool_factor.input_var_edges,
                    f'var_to_fac_{mtype}',
                    input_margs_rs - outgoing)

    def update_coeff_to_factor_message(self):
        for mtype in ('eta', 'Lambda'):
            out_margs = getattr(self.coeff_vars, mtype)
            outgoing = getattr(self.pool_factor.output_var_edges, f'fac_to_var_{mtype}')
            setattr(self.pool_factor.output_var_edges,
                    f'var_to_fac_{mtype}',
                    out_margs - outgoing)

    def update_factor_to_variables_message(self):
        self.pool_factor.update_outgoing_messages(None)

    @property
    def state(self):
        state = [[self.input_vars.eta, self.input_vars.Lambda],
                 [self.coeff_vars.eta, self.coeff_vars.Lambda]]
        state.append(self.pool_factor.state)
        state.append(self.coeff_prior_factor.state)
        return state

    @state.setter
    def state(self, new_state):
        self.input_vars.eta, self.input_vars.Lambda = new_state[0]
        self.coeff_vars.eta, self.coeff_vars.Lambda = new_state[1]
        self.pool_factor.state = new_state[2]
        self.coeff_prior_factor.state = new_state[3]

    @property
    def named_state(self):
        state = [('input_marginals', [self.input_vars.eta, self.input_vars.Lambda]),
                 ('coeff_marginals', [self.coeff_vars.eta, self.coeff_vars.Lambda])]
        state.append((f'{self.pool_type}_pool_factor', self.pool_factor.named_state))
        state.append(('coeff_prior_factor', self.coeff_prior_factor.named_state))
        return state

    @property
    def edges(self):
        return self.pool_factor.input_var_edges,\
               self.pool_factor.output_var_edges,\
               self.coeff_prior_factor.var_edges

    @property
    def variables(self):
        return self.input_vars, self.coeff_vars

    @property
    def params(self):
        return ()

    def _update_input_marginals(self, return_eta_Lambda=False):
        ks = self.pool_factor.ksize
        to_return = []
        for mtype in ('eta', 'Lambda'):
            inshp = getattr(self.input_vars, mtype).shape
            fac_to_var = getattr(self.pool_factor.input_var_edges, f'fac_to_var_{mtype}')

            # Reshape fac to var messages to be same as input margs
            fac_to_var_rs = tf.reshape(fac_to_var, fac_to_var.shape.as_list()[:-1] + [ks, ks])
            fac_to_var_rs_tp = tf.transpose(fac_to_var_rs, [0, 1, 4, 2, 5, 3])
            inp_marg = tf.reshape(fac_to_var_rs_tp, inshp)

            if return_eta_Lambda:
                to_return.append(inp_marg)
            else:
                setattr(self.input_vars, mtype, inp_marg)
        if return_eta_Lambda:
            return to_return

    def _update_coeff_marginals(self, return_eta_Lambda=False):
        to_return = []
        for mtype in ('eta', 'Lambda'):
            fac_to_var = getattr(self.pool_factor.output_var_edges, f'fac_to_var_{mtype}')
            prior = getattr(self.coeff_prior_factor.var_edges, f'fac_to_var_{mtype}')
            marg = prior + fac_to_var

            if return_eta_Lambda:
                to_return.append(marg)
            else:
                setattr(self.coeff_vars, mtype, marg)
        if return_eta_Lambda:
            return to_return

    def update_marginals(self):
        self.update_input_marginals()
        self.update_coeff_marginals()

    def energy(self, as_numpy=False, robust=False, sum_all=True, inputs=None, coeffs=None):
        inputs = inputs if inputs is not None else self.input_vars.mu
        coeffs = coeffs if coeffs is not None else self.coeff_vars.mu
        energies = {f'{self.pool_type}_pool': self.pool_factor.energy([inputs, coeffs], robust=robust, aggregate=True)}
        energies['coeff_prior'] = self.coeff_prior_factor.energy([coeffs], robust=robust, aggregate=True)
        if as_numpy:
            energies = {n: e.numpy() for n, e in energies.items()}
        if sum_all:
            return tf.reduce_sum(list(energies.values()))
        else:
            return energies

    def relinearise_factors(self, itr):
        relin_points = self.linearisation_points
        self._relinearise_factors(itr, relin_points)

    def _relinearise_factors(self, itr, relin_points):
        # Avg pool is linear - no need to relinearise
        if self.pool_type == 'max':
            self.pool_factor.relinearise(itr, relin_points[f'{self.pool_type}_pool'])

    @property
    def linearisation_points(self):
        return {f'{self.pool_type}_pool': [self.input_vars.mu, self.coeff_vars.mu]}

    @linearisation_points.setter
    def linearisation_points(self, new_lin_points):
        self._relinearise_factors(itr=0, relin_points=new_lin_points)
