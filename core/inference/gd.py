# coding=utf-8
import tensorflow as tf

from core.inference.base import Layer
from core.utils.utils import flatten_filters


class GradientDescentLayer(Layer):
    def __init__(self,
                 *args,
                 **kwargs):
        super(GradientDescentLayer, self).__init__(*args, **kwargs)
        self.filtervars_tf = tf.Variable(self.filter_vars.mu)
        self.inputvars_tf = tf.Variable(self.input_vars.mu)
        self.weightvars_tf = tf.Variable(self.weight_vars.mu)
        if self.use_component_vars:
            self.compvars_tf = tf.Variable(self.component_vars.mu)

    def layer_energy(self):
        if self.use_component_vars:
            filtfac_vars_mu = [self.compvars_tf, self.inputvars_tf]
        else:
            filtfac_vars_mu = [self.filtervars_tf, self.inputvars_tf]
            if self.use_filter_weights:
                filtfac_vars_mu.append(self.weightvars_tf)
        energies = {}
        energies['filter_pixel'] = self.filter_factor.energy(filtfac_vars_mu,
                                                             robust=self.filter_factor.N_rob is not None,
                                                             aggregate=True)
        if self.is_first_layer:
            energies['pixel_obs'] = self.pixel_obs_factor.energy([self.inputvars_tf],
                                                                 robust=self.pixel_obs_factor.N_rob is not None,
                                                                 aggregate=True)
        energies['filter_prior'] = self.filter_prior_factor.energy([flatten_filters(self.filtervars_tf)],
                                                                   aggregate=True)
        if self.use_filter_weights:
            energies['weight_prior'] = self.weight_prior_factor.energy([self.weightvars_tf], aggregate=True)
        if self.use_weight_norm_factor:
            energies['non_zero_weight_norm'] = self.non_zero_weight_norm_factor.energy([self.weightvars_tf],
                                                                                       aggregate=True)
        if self.use_component_vars:
            energies['component_consist'] = self.component_factor.energy([self.filtervars_tf,
                                                                          self.weightvars_tf,
                                                                          self.compvars_tf],
                                                                         aggregate=True)
        return tf.reduce_sum(list(energies.values()))

    def run_gradient_descent(self, n_iters, img_new, callback_fn,
                             filter_factor_recon_only=False,
                             callback_freq=10, optim=None, lr=0.01):
        optimclass = {'adam': tf.optimizers.Adam,
                      'sgd': tf.optimizers.SGD}[optim]
        gdopt = optimclass(learning_rate=lr)

        def _step():
            with tf.GradientTape() as gt:
                E = self.layer_energy()
            grad_vars_mu = [self.filtervars_tf, self.inputvars_tf]
            if self.use_filter_weights:
                grad_vars_mu.append(self.weightvars_tf)
            allvars = [self.inputvars_tf, self.filtervars_tf]
            if self.use_filter_weights:
                allvars.append(self.weightvars_tf)
            if self.use_component_vars:
                allvars.append(self.compvars_tf)
            grads = gt.gradient(E, allvars)
            gdopt.apply_gradients(zip(grads, allvars))

        for i in range(n_iters):
            if i % callback_freq == 0 or i == n_iters - 1:
                self.input_vars.eta = tf.convert_to_tensor(self.inputvars_tf) * self.input_vars.Lambda
                self.filter_vars.eta = tf.convert_to_tensor(self.filtervars_tf) * self.filter_vars.Lambda
                self.weight_vars.eta = tf.convert_to_tensor(self.weightvars_tf) * self.weight_vars.Lambda
                if self.use_component_vars:
                    self.component_vars.eta = tf.convert_to_tensor(self.compvars_tf) * self.component_vars.Lambda
                callback_fn(self.inputvars_tf, self.filtervars_tf, itr=i,
                            weights=self.weightvars_tf if self.use_filter_weights else None,
                            components=self.compvars_tf if self.use_component_vars else None)
            _step()
        out = [[self.input_vars.eta, self.input_vars.Lambda],
               [self.filter_vars.eta, self.filter_vars.Lambda]]
        if self.use_filter_weights:
            out.append([self.weight_vars.eta, self.weight_vars.Lambda])
        return out

GradientDescentLearner = GradientDescentLayer
