# coding=utf-8
import tensorflow as tf

from core.factors import DenseFactor, NonLinearUnaryFactor, UnaryFactor
from core.inference.gbp.layers.base import GBPLayer
from core.variables import CoeffVariable, PixelVariable, WeightVariable, BiasVariable


class GBPDenseLayer(GBPLayer):
    def __init__(self,
                 dense_factor: DenseFactor,
                 coeff_prior_factor: [NonLinearUnaryFactor, UnaryFactor],
                 weight_prior_factor: [NonLinearUnaryFactor, UnaryFactor],
                 input_vars: [CoeffVariable, PixelVariable, None],   # Can be None if input is noiseless
                 weight_vars: WeightVariable,
                 coeff_vars: [CoeffVariable],
                 bias_prior_factor: [NonLinearUnaryFactor, UnaryFactor, None] = None,
                 bias_vars: [BiasVariable, None] = None,
                 fixed_params: bool = False):
        super(GBPDenseLayer, self).__init__(input_vars=input_vars,
                                            coeff_vars=coeff_vars)
        self.weight_vars = weight_vars
        self.bias_vars = bias_vars
        self.use_bias = bias_vars is not None
        self.input_shp = None if dense_factor.is_noiseless_input else self.input_vars.shape

        self.dense_factor = dense_factor
        self.coeff_prior_factor = coeff_prior_factor
        self.weight_prior_factor = weight_prior_factor
        self.bias_prior_factor = bias_prior_factor

        self.using_nonlinear_coeff_prior = hasattr(coeff_prior_factor, 'relin_freq') and coeff_prior_factor.relin_freq is not None
        self.using_nonlinear_weight_prior = hasattr(weight_prior_factor, 'relin_freq') and weight_prior_factor.relin_freq is not None
        if self.use_bias:
            assert bias_prior_factor is not None, "Must provide `bias_prior_factor` if using bias vars"
            self.using_nonlinear_bias_prior = hasattr(bias_prior_factor, 'relin_freq') and bias_prior_factor.relin_freq is not None
        else:
            self.using_nonlinear_bias_prior = False
        self.fixed_params = fixed_params

    def intra_layer_inference_iter(self, itr):
        # self.update_marginals()
        # self.relinearise_factors(itr)
        if not self.dense_factor.is_noiseless_input:
            self.update_input_to_dense_factor_message()
        self.update_coeff_to_dense_factor_message()
        self.update_dense_factor_to_variables_message()
        self.update_marginals()
        self.relinearise_factors(itr)
        # if not self.dense_factor.is_noiseless_input:
        #     self.update_input_to_dense_factor_message()
        # self.update_coeff_to_dense_factor_message()
        if not self.fixed_params:
            self.update_weight_to_dense_factor_message()
            if self.use_bias:
                self.update_bias_to_dense_factor_message()
        if self.using_nonlinear_coeff_prior:
            self.update_coeff_prior_factor_to_coeff_message()
        if self.using_nonlinear_weight_prior:
            self.update_weight_prior_factor_to_weight_message()
        # self.update_marginals()

    def update_input_to_dense_factor_message(self):
        for mtype in ('eta', 'Lambda'):
            input_margs = getattr(self.input_vars, mtype)
            input_margs = tf.reshape(input_margs, [input_margs.shape[0], tf.reduce_prod(input_margs.shape[1:])])
            outgoing = getattr(self.dense_factor.input_var_edges, f'fac_to_var_{mtype}')

            setattr(self.dense_factor.input_var_edges,
                    f'var_to_fac_{mtype}',
                    input_margs[..., None, :] - outgoing)

    def update_weight_to_dense_factor_message(self):
        for mtype in ('eta', 'Lambda'):
            weight_margs = getattr(self.weight_vars, mtype)
            outgoing = getattr(self.dense_factor.weight_var_edges, f'fac_to_var_{mtype}')

            setattr(self.dense_factor.weight_var_edges,
                    f'var_to_fac_{mtype}',
                    tf.linalg.matrix_transpose(weight_margs)[None] - outgoing)

    def update_bias_to_dense_factor_message(self):
        for mtype in ('eta', 'Lambda'):
            bias_margs = getattr(self.bias_vars, mtype)
            outgoing = getattr(self.dense_factor.bias_var_edges, f'fac_to_var_{mtype}')
            setattr(self.dense_factor.bias_var_edges,
                    f'var_to_fac_{mtype}',
                    bias_margs[None] - outgoing)

    def update_coeff_to_dense_factor_message(self):
        for mtype in ('eta', 'Lambda'):
            out_margs = getattr(self.coeff_vars, mtype)
            outgoing = getattr(self.dense_factor.output_var_edges, f'fac_to_var_{mtype}')
            setattr(self.dense_factor.output_var_edges,
                    f'var_to_fac_{mtype}',
                    out_margs - outgoing)

    def update_weight_prior_factor_to_weight_message(self):
        self.weight_prior_factor.update_outgoing_messages(self.weight_prior_factor.var0)

    def update_coeff_prior_factor_to_coeff_message(self):
        self.coeff_prior_factor.update_outgoing_messages(self.coeff_prior_factor.var0)

    def update_dense_factor_to_variables_message(self):
        self.dense_factor.update_outgoing_messages(self.dense_factor.var0)

    @property
    def state(self):
        state = [] if self.dense_factor.is_noiseless_input else [[self.input_vars.eta, self.input_vars.Lambda]]
        state += [[self.weight_vars.eta, self.weight_vars.Lambda]]
        state += [[self.bias_vars.eta, self.bias_vars.Lambda]] if self.use_bias else []
        state += [[self.coeff_vars.eta, self.coeff_vars.Lambda]]
        state.append(self.dense_factor.state)
        if self.using_nonlinear_weight_prior:
            state.append(self.weight_prior_factor.state)
        if self.using_nonlinear_bias_prior:
            state.append(self.bias_prior_factor.state)
        if self.using_nonlinear_coeff_prior:
            state.append(self.coeff_prior_factor.state)
        return state

    @state.setter
    def state(self, new_state):
        if not self.dense_factor.is_noiseless_input:
            self.input_vars.eta, self.input_vars.Lambda = new_state[0]
        n_in = 1 - int(self.dense_factor.is_noiseless_input)
        n_bias = int(self.use_bias)
        self.weight_vars.eta, self.weight_vars.Lambda = new_state[n_in]
        lastvar_id = n_in
        if self.use_bias:
            self.bias_vars.eta, self.bias_vars.Lambda = new_state[1 + lastvar_id]
            lastvar_id += n_bias
        self.coeff_vars.eta, self.coeff_vars.Lambda = new_state[1 + lastvar_id]
        self.dense_factor.state = new_state.pop(2 + lastvar_id)
        if self.using_nonlinear_weight_prior:
            self.weight_prior_factor.state = new_state.pop(2 + lastvar_id)
        if self.using_nonlinear_bias_prior:
            self.bias_prior_factor.state = new_state.pop(2 + lastvar_id)
        if self.using_nonlinear_coeff_prior:
            self.coeff_prior_factor.state = new_state.pop(2 + lastvar_id)

    @property
    def variables(self):
        variables = () if self.dense_factor.is_noiseless_input else (self.input_vars,)
        variables += (self.weight_vars,)
        variables += (self.bias_vars,) if self.use_bias else ()
        variables += (self.coeff_vars,)
        return variables

    @property
    def params(self):
        params = (self.weight_vars,)
        if self.use_bias:
            params += (self.bias_vars,)
        return params

    @property
    def named_state(self):
        state = [] if self.dense_factor.is_noiseless_input else \
            [('input_marginals', [self.input_vars.eta, self.input_vars.Lambda])]
        state += [('weight_marginals', [self.weight_vars.eta, self.weight_vars.Lambda])]
        state += [('bias_marginals', [self.bias_vars.eta, self.bias_vars.Lambda])] if self.use_bias else []
        state += [('coeff_marginals', [self.coeff_vars.eta, self.coeff_vars.Lambda])]
        state.append(('dense_factor', self.dense_factor.named_state))
        if self.using_nonlinear_weight_prior:
            state.append(('weight_prior_factor', self.weight_prior_factor.named_state))
        if self.using_nonlinear_bias_prior:
            state.append(('bias_prior_factor', self.bias_prior_factor.named_state))
        if self.using_nonlinear_coeff_prior:
            state.append(('coeff_prior_factor', self.coeff_prior_factor.named_state))
        return state

    def _update_input_marginals(self, return_eta_Lambda=False):
        to_return = []
        for mtype in ('eta', 'Lambda'):
            fac_to_var = getattr(self.dense_factor.input_var_edges, f'fac_to_var_{mtype}')

            # Sum over output dim (have separate factor per output dim)
            fac_to_var = tf.reduce_sum(fac_to_var, axis=-2)

            # Reshape (for non-flattened input)
            fac_to_var = tf.reshape(fac_to_var, self.input_shp)

            if return_eta_Lambda:
                to_return.append(fac_to_var)
            else:
                setattr(self.input_vars, mtype, fac_to_var)
        if return_eta_Lambda:
            return to_return

    def _update_coeff_marginals(self, return_eta_Lambda=False):
        to_return = []
        for mtype in ('eta', 'Lambda'):
            fac_to_var = getattr(self.dense_factor.output_var_edges, f'fac_to_var_{mtype}')
            prior = getattr(self.coeff_prior_factor.var_edges, f'fac_to_var_{mtype}')
            marg = prior + fac_to_var

            if return_eta_Lambda:
                to_return.append(marg)
            else:
                setattr(self.coeff_vars, mtype, marg)
        if return_eta_Lambda:
            return to_return

    def update_weight_marginals(self):
        for mtype in ('eta', 'Lambda'):
            fac_to_var = getattr(self.dense_factor.weight_var_edges, f'fac_to_var_{mtype}')

            # Sum messages over observations (and transpose)
            fac_to_var = tf.transpose(tf.reduce_sum(fac_to_var, axis=-3))

            prior = getattr(self.weight_prior_factor.var_edges, f'fac_to_var_{mtype}')
            marg = prior + fac_to_var
            setattr(self.weight_vars, mtype, marg)

    def update_bias_marginals(self):
        for mtype in ('eta', 'Lambda'):
            fac_to_var = getattr(self.dense_factor.bias_var_edges, f'fac_to_var_{mtype}')

            # Sum messages over observations
            fac_to_var = tf.reduce_sum(fac_to_var, axis=-2)

            prior = getattr(self.bias_prior_factor.var_edges, f'fac_to_var_{mtype}')
            marg = prior + fac_to_var
            setattr(self.bias_vars, mtype, marg)

    def update_marginals(self):
        if not self.dense_factor.is_noiseless_input:
            self.update_input_marginals()
        if not self.fixed_params:
            self.update_weight_marginals()
            if self.use_bias:
                self.update_bias_marginals()
        self.update_coeff_marginals()

    def energy(self, as_numpy=False, robust=False, sum_all=True, inputs=None, weights=None, coeffs=None, bias=None):
        weights = weights if weights is not None else self.weight_vars.mu
        coeffs = coeffs if coeffs is not None else self.coeff_vars.mu
        if self.use_bias:
            bias = bias if bias is not None else self.bias_vars.mu
            current_vars = [weights, bias, coeffs]
        else:
            bias = None
            current_vars = [weights, coeffs]
        if not self.dense_factor.is_noiseless_input:
            inputs = inputs if inputs is not None else self.input_vars.mu
            current_vars = [inputs] + current_vars
        energies = dict()
        energies['dense'] = self.dense_factor.energy(current_vars, robust=robust, aggregate=True)
        energies['weight_prior'] = self.weight_prior_factor.energy([weights], robust=robust, aggregate=True)
        energies['coeff_prior'] = self.coeff_prior_factor.energy([coeffs], robust=robust, aggregate=True)
        if self.use_bias:
            bias = bias if bias is not None else self.bias_vars.mu
            energies['bias_prior'] = self.bias_prior_factor.energy([bias], robust=robust, aggregate=True)
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
        self.dense_factor.relinearise(itr, relin_points['dense'])
        if self.using_nonlinear_weight_prior:
            self.weight_prior_factor.relinearise(itr, relin_points['weight_prior'])
        if self.using_nonlinear_coeff_prior:
            self.coeff_prior_factor.relinearise(itr, relin_points['coeff_prior'])
        if self.using_nonlinear_bias_prior:
            self.bias_prior_factor.relinearise(itr, relin_points['bias_prior'])

    @property
    def linearisation_points(self):
        relin_points = dict()   # Mapping factor_name -> factor_relin_point
        linvars = [] if self.dense_factor.is_noiseless_input else [self.input_vars.mu]
        linvars += [self.weight_vars.mu]
        linvars += [self.bias_vars.mu] if self.use_bias else []
        linvars += [self.coeff_vars.mu]
        relin_points['dense'] = linvars
        if self.using_nonlinear_weight_prior:
            relin_points['weight_prior'] = [self.weight_vars.mu]
        if self.using_nonlinear_coeff_prior:
            relin_points['coeff_prior'] = [self.coeff_vars.mu]
        if self.using_nonlinear_bias_prior:
            relin_points['bias_prior'] = [self.bias_vars.mu]

        return relin_points

    @linearisation_points.setter
    def linearisation_points(self, new_relin_points):
        # Give i=0 to force relinearisation
        self._relinearise_factors(itr=0, relin_points=new_relin_points)

    def fix_params(self):
        self.fixed_params = True
        self.dense_factor.fixed_params = True

    def unfix_params(self):
        self.fixed_params = False
        self.dense_factor.fixed_params = False

    def relinearise_factor_weights(self, new_weights):
        weight_id = 0 if self.dense_factor.is_noiseless_input else 1
        self.dense_factor.var0[weight_id] = new_weights

    def relinearise_factor_bias(self, new_bias):
        assert self.use_bias, \
            "Attempted to relinearise bias vars in layer without bias"
        bias_id = 1 if self.dense_factor.is_noiseless_input else 2
        self.dense_factor.var0[bias_id] = new_bias

    @property
    def edges(self):
        edges = [self.dense_factor.input_var_edges,
                 self.dense_factor.weight_var_edges]
        if self.use_bias:
            edges += [self.dense_factor.bias_var_edges]
        edges += [self.dense_factor.output_var_edges,
                  self.weight_prior_factor.var_edges]
        if self.use_bias:
            edges += [self.bias_prior_factor.var_edges]
        edges += [self.coeff_prior_factor.var_edges]
        return edges

    # def deterministic_dense_proj(self):
    #     inputs_rs = tf.reshape(self.input_vars.mu, [self.input_vars.mu.shape[0], -1])
    #     weightsTinputs = tf.reduce_sum(tf.transpose(self.weight_vars.mu)[None] * inputs_rs[:, None], axis=-1)
    #     if self.use_bias:
    #         weightsTinputs += self.bias_vars.mu
    #     return self.dense_factor.nonlin(weightsTinputs)
