# coding=utf-8
import tensorflow as tf

from core.inference.base import Learner
from core.inference.gbp.layers import GBPConvLayer, GBPDenseLayer
from core.factors import AdditiveFilterFactorSumDecomp, Factor
from core.utils.utils import flatten_filters, dotdict


class GBPLearner(Learner):

    def _inference_iter(self, itr, *state):
        self.state = state

        control_deps = []
        if self.layer_schedule == 'random':
            schedule = tf.random.uniform((len(self.layers),),
                                         minval=0,
                                         maxval=len(self.layers),
                                         dtype=tf.int32)
            # schedule = tf.random.shuffle(tf.range(len(self.layers)))
            schedule = [int(s) for s in schedule]
        else:
            schedule = self.layer_schedule
        for lays_sched in schedule:
            if isinstance(lays_sched, (tuple, list)):
                self.multi_layer_inference_iter(lays_sched, itr)
            else:
                assert isinstance(lays_sched, int), \
                    "Elements of `layer_schedule` must be ints or iterables"
                self.layers[lays_sched].intra_layer_inference_iter(itr)

        return [itr + 1] + self.state

    def multi_layer_inference_iter(self, layers, itr):
        for l in layers:
            # Set new relinarisation point (at specified freq)
            self.layers[l].relinearise_factors(itr)
        for l in layers:
            if self.layers[l].use_weight_norm_factor:
                self.layers[l].update_weight_to_non_zero_weight_norm_factor_message()
                self.layers[l].update_non_zero_weight_norm_factor_to_weight_message()

        for l in layers:
            # Pixel vars to filter factors
            self.layers[l].update_input_to_filter_factor_message()

        # Do parallel updating of messages from filter factors to vars
        self.multi_layer_update_filter_factor_to_variable_messages(layers)

        for l in layers:
            self.layers[l].update_input_marginals()
        for l in layers:
            if self.layers[l].use_component_vars:
                self.layers[l].update_component_marginals()
                for _ in range(1):
                    self.layers[l].update_component_to_component_factor_message()
                    self.layers[l].update_component_factor_to_variable_messages()
                    self.layers[l].update_coeff_marginals()
                    self.layers[l].update_filter_marginals()

                    self.layers[l].update_filter_to_component_factor_message()
                    self.layers[l].update_coeff_to_component_factor_message()
                    self.layers[l].update_component_factor_to_variable_messages()
                    self.layers[l].update_component_marginals()
                    # self.update_coeff_marginals()
                    # self.update_filter_marginals()
                    self.layers[l].update_component_to_filter_factor_message()

            else:
                self.layers[l].update_filter_marginals()

                # Filter variables to filter factor
                self.layers[l].update_filter_to_filter_factor_message()

                self.layers[l].update_coeff_marginals()
                if self.layers[l].use_filter_coeffs:
                    self.layers[l].update_coeff_to_filter_factor_message()

        for l in layers:
            if self.layers[l].use_filter_diversity_factor:
                self.layers[l].update_filter_diversity_factor_to_filter_message()

            if self.layers[l].use_filter_normalisation_factor:
                # self.update_filter_normalisation_factor_to_filter_message()
                self.layers[l].update_filter_normalisation_factor_to_filter_message()
                self.layers[l].update_filter_to_filter_normalisation_factor_message()

            if self.layers[l].use_filter_diversity_factor:
                self.layers[l].update_filter_to_filter_diversity_factor_message()

            if self.layers[l].is_first_layer:
                # Pixel observation factors to pixel variables
                self.layers[l].update_pixel_obs_factor_to_input_message()

                if self.layers[l].doing_segmentation:
                    # Segment labels to segment vars message
                    self.layers[l].update_segment_obs_factor_to_segment_vars_message()

            if self.layers[l].use_robust_coeff_prior_factor:
                self.layers[l].update_coeff_prior_factor_to_coeff_message()

            # Update marginals
            self.layers[l].update_marginals()

        # self.relinearise_factors(itr)

    def multi_layer_update_filter_factor_to_variable_messages(self, layer_ids):
        # Currently require all layers to have same number of filters for easy stacking
        def get_nfilt(layid):
            return self.layers[layid].filter_vars.shape[-1]

        n_filt_first = get_nfilt(layer_ids[0])
        assert all(get_nfilt(layid) == n_filt_first for layid in layer_ids), \
            "All layers in block must have same number of filters to allow " \
            "for parallelising messages"

        # Check all layers using the decomposed filter sum factors
        assert all(isinstance(self.layers[l].filter_factor, AdditiveFilterFactorSumDecomp)
                   for l in layer_ids), "Only `AdditiveFilterFactorSumDecomp` supported for parallel, " \
                                        "multi-layer message passing"

        vars_per_filter = n_filt_first + 2  # Two extra vars for connected inputs in neighbourhood

        # Get incoming messages for all layers
        msgs_in = [self.layers[lay].filter_factor._get_incoming_messages() for lay in layer_ids]
        msgs_in_eta, msgs_in_Lambda = zip(*msgs_in)

        # Create tensor stacker object. We'll use this for combining inputs
        # for all layers, and then disaggregating their outputs
        stacker = TensorStacker(*[m.shape for m in msgs_in_eta])

        # Now stack incoming messages
        msgs_in_eta_stack = stacker.stack_tensors(msgs_in_eta)
        msgs_in_Lambda_stack = stacker.stack_tensors(msgs_in_Lambda)

        # Check unstacking takes us back to original tensors
        # msgs_in_eta_us = stacker.unstack_tensor(msgs_in_eta_stack)
        # msgs_in_Lambda_us = stacker.unstack_tensor(msgs_in_Lambda_stack)
        # for m, m_us in zip(msgs_in_eta_us, msgs_in_eta):
        #     tf.assert_equal(m_us, m)
        # for ml, ml_us in zip(msgs_in_Lambda_us, msgs_in_Lambda):
        #     tf.assert_equal(ml, ml_us)

        # Get factor etas and Jacobians for each layer
        eta_Js = [self.layers[lay].filter_factor.get_eta_J(None) for lay in layer_ids]
        eta_J_div_sigmas = [(e, J / self.layers[lay].filter_factor.sigma) for (e, J), lay in zip(eta_Js, layer_ids)]
        factors_eta, factors_J_div_sigmas = zip(*eta_J_div_sigmas)

        # Pack Factor Jacobians for all layers into matrix
        factor_J_div_sigma_stack = stacker.stack_tensors(factors_J_div_sigmas)

        # Pack factor etas for all layers into matrix
        factor_eta_stack = stacker.stack_tensors(factors_eta)

        # Do marginalisation for factors in all layers, to compute their outgoings messages
        fac_to_var_eta_stack, fac_to_var_Lambda_stack = \
            Factor.marginalise_sherman_morrison(mess_eta=msgs_in_eta_stack,
                                                factor_eta=factor_eta_stack,
                                                mess_Lambda=msgs_in_Lambda_stack,
                                                J_div_sigma=factor_J_div_sigma_stack)

        fac_to_var_eta_stack = tf.reshape(fac_to_var_eta_stack, (-1, 18))
        fac_to_var_Lambda_stack = tf.reshape(fac_to_var_Lambda_stack, (-1, 18))

        # Unstack the messages
        fac_to_var_etas = stacker.unstack_tensor(fac_to_var_eta_stack)
        fac_to_var_Lambdas = stacker.unstack_tensor(fac_to_var_Lambda_stack)

        # # Update messages on edges
        for i, (msg_out_eta, msg_out_Lambda) in enumerate(zip(fac_to_var_etas, fac_to_var_Lambdas)):
            self.layers[i].filter_factor.update_outgoing_edge_messages(msg_out_eta, msg_out_Lambda)

    def fix_layer_params(self):
        for lay in self.layers:
            if isinstance(lay, (GBPDenseLayer, GBPConvLayer)):
                lay.fix_params()

    def fix_layer_coeffs(self):
        for lay in self.layers:
            if isinstance(lay, GBPConvLayer):
                lay.fix_coeffs()

    def fix_pixels(self):
        self.layers[0].fix_inputs()

    @property
    def state(self):
        return [lay.state for lay in self.layers]

    @state.setter
    def state(self, new_state):
        for lay, ns in zip(self.layers, new_state):
            lay.state = ns

    @property
    def variables(self):
        vs = ()
        for lay in self.layers:
            vs += (lay.variables,)
        return vs


class TensorStacker(object):
    def __init__(self, *shps_to_stack):
        self.orig_shps = shps_to_stack
        self.n_elem_per_tensor = [tf.reduce_prod(s[:-1]) for s in shps_to_stack]

    def collapse_batch_dims(self, x):
        x_reshp = tf.reshape(x, (-1, x.shape[-1]))
        return x_reshp

    def stack_tensors(self, xs):
        return tf.concat([self.collapse_batch_dims(x) for x in xs], axis=0)

    def unstack_tensor(self, x):
        unstck = []
        start_id = 0
        for tens_id in range(len(self.orig_shps)):
            tens = tf.gather(x, tf.range(start_id, start_id + self.n_elem_per_tensor[tens_id]), axis=0)
            tens_reshp = tf.reshape(tens, self.orig_shps[tens_id], name='unstck_reshp')
            unstck.append(tens_reshp)
            start_id += self.n_elem_per_tensor[tens_id]
        return unstck


def filtering_on_weights(gbplearner_prev: [GBPLearner, list, tuple],
                         gbplearner_new: GBPLearner,
                         prec_rescale_factor: float = 1.,
                         prec_rescale_conv_only: bool = False,
                         alpha: [None, float] = None,
                         init_prior_sigmas: [None, list, tuple] = None):
    """
    Takes posterior over learnable weights/filters in `gbplearner_prev` and
    sets it as a prior on the same variables in `gbplearner_new`
    """
    # Previous posterior may either be as a GBPLearner, or an iterable of weight variables
    # For the latter case need to reformat so same code can be used
    from_weight_vars = isinstance(gbplearner_prev, (list, tuple))
    if from_weight_vars:
        gbplearnerprev = dotdict(layers=[])
        weight_id = 0
        vnames = ('weight_vars', 'filter_vars')
        for i, l in enumerate(gbplearner_new.layers):
            has_weights = False
            for v in vnames:
                if hasattr(l, v):
                    gbplearnerprev.layers.append(dotdict({v: dotdict(gbplearner_prev[weight_id])}))
                    weight_id += 1
                    has_weights = True
            if not has_weights:
                gbplearnerprev.layers.append(None)
        gbplearner_prev = gbplearnerprev

    init_prior_sigmas = init_prior_sigmas if init_prior_sigmas is None else iter(init_prior_sigmas)

    # Get posterior marginals over weights (for dense layers)
    # and filters (for conv layers) in gbplearner_prev
    # Set the eta and Lambda for the corresponding prior factors in gbplearner_new
    for lay_old, lay_new in zip(gbplearner_prev.layers, gbplearner_new.layers):
        if not from_weight_vars:
            assert type(lay_old) is type(lay_new), \
                "`gbplearner_prev` and `gbplearner_new` have different architectures."
        if isinstance(lay_new, GBPConvLayer):
            # lay_old.update_filter_marginals()   # In case some messages not yet gathered

            # Get marginal posterior eta and Lambda, reshape to be consistent with prior
            # Rescale both the precision and info by specified factor (rescaling variance)
            posterior_eta = flatten_filters(lay_old.filter_vars.eta * prec_rescale_factor)
            posterior_Lambda = flatten_filters(lay_old.filter_vars.Lambda * prec_rescale_factor)

            if alpha is not None:
                # Make prior from weighted mix of old posterior and init prior
                mean = posterior_eta / posterior_Lambda
                std = tf.math.sqrt(1. / posterior_Lambda)
                mean = alpha * mean + (1. - alpha) * 0.
                std = alpha * std + (1. - alpha) * next(init_prior_sigmas)
                posterior_eta = mean / std ** 2.
                posterior_Lambda = 1. / std ** 2.

            # Set prior over filters in new graph to posterior in old graph
            lay_new.filter_prior_factor.reset_eta_Lambda(posterior_eta, posterior_Lambda)
            print(tf.reduce_mean(posterior_Lambda, axis=[1, 2]), 'post lambda mean filt')

            # Initialise message from filter variables to recon factors equal to prev posterior
            edge_shp = lay_new.filter_factor.filter_var_edges.var_to_fac_eta.shape
            lay_new.filter_factor.filter_var_edges.var_to_fac_eta = \
                tf.broadcast_to(posterior_eta[None, None, None], edge_shp)
            lay_new.filter_factor.filter_var_edges.var_to_fac_Lambda = \
                tf.broadcast_to(posterior_Lambda[None, None, None], edge_shp)

            # We adjusted the priors, so recompute marginals to keep up to date
            lay_new.update_filter_marginals()

            # Relinearise recon factors around new weight values
            # lay_new.relinearise_factor_weights(lay_new.filter_vars.mu)

        elif isinstance(lay_new, GBPDenseLayer):
            # lay_old.update_weight_marginals()  # In case some messages not yet gathered

            # Get posterior marginals over weights
            # Rescale both the precision and info by specified factor (rescaling variance)
            posterior_eta = lay_old.weight_vars.eta
            posterior_Lambda = lay_old.weight_vars.Lambda
            if not prec_rescale_conv_only:
                posterior_eta *= prec_rescale_factor
                posterior_Lambda *= prec_rescale_factor

            if alpha is not None:
                # Make prior from weighted mix of old posterior and init prior
                mean = posterior_eta / posterior_Lambda
                std = tf.math.sqrt(1. / posterior_Lambda)
                mean = alpha * mean + (1. - alpha) * 0.
                std = alpha * std + (1. - alpha) * next(init_prior_sigmas)
                posterior_eta = mean / std ** 2.
                posterior_Lambda = 1. / std ** 2.

            # Set weight prior in new graph to weight posterior from old graph
            lay_new.weight_prior_factor.reset_eta_Lambda(posterior_eta, posterior_Lambda)

            # Initialise message from filter variables to recon factors equal to prev posterior
            edge_shp = lay_new.dense_factor.weight_var_edges.var_to_fac_eta.shape
            lay_new.dense_factor.weight_var_edges.var_to_fac_eta = tf.broadcast_to(tf.transpose(posterior_eta)[None], edge_shp)
            lay_new.dense_factor.weight_var_edges.var_to_fac_Lambda = tf.broadcast_to(tf.transpose(posterior_Lambda)[None], edge_shp)

            # lay_new.dense_factor.input_var_edges.var_to_fac_eta = tf.broadcast_to(tf.transpose(posterior_eta)[None], edge_shp)
            # lay_new.dense_factor.input_var_edges.var_to_fac_Lambda = tf.broadcast_to(tf.transpose(posterior_Lambda)[None], edge_shp)
            print(tf.reduce_mean(posterior_Lambda), 'post lambda mean weight')
            # We adjusted the priors, so recompute marginals to keep up to date
            lay_new.update_weight_marginals()

            # Relinearise recon factors around new weight values
            # lay_new.relinearise_factor_weights(lay_new.weight_vars.mu)


def filtering_on_biases(gbplearner_prev: [GBPLearner, list, tuple],
                        gbplearner_new: GBPLearner,
                        prec_rescale_factor: float = 1.,
                        prec_rescale_conv_only: bool = False,
                        alpha: [None, float] = None,
                        init_prior_sigmas: [None, list, tuple] = None):
    """
    Takes posterior over learnable biases in `gbplearner_prev` and
    sets it as a prior on the same variables in `gbplearner_new`
    """
    # Previous posterior may either be as a GBPLearner, or an iterable of bias variables
    # For the latter case need to reformat so same code can be used
    from_bias_vars = isinstance(gbplearner_prev, (list, tuple))
    if from_bias_vars:
        gbplearnerprev = dotdict(layers=[])
        bias_id = 0
        for i, l in enumerate(gbplearner_new.layers):
            if hasattr(l, 'use_bias'):
                if not l.use_bias:
                    continue
            if hasattr(l, 'bias_vars'):
                gbplearnerprev.layers.append(dotdict({'bias_vars': dotdict(gbplearner_prev[bias_id])}))
                bias_id += 1
            else:
                gbplearnerprev.layers.append(None)
        gbplearner_prev = gbplearnerprev

    init_prior_sigmas = init_prior_sigmas if init_prior_sigmas is None else iter(init_prior_sigmas)

    # Get posterior marginals over biases
    # Set the eta and Lambda for the corresponding prior factors in gbplearner_new
    for lay_old, lay_new in zip(gbplearner_prev.layers, gbplearner_new.layers):
        if not from_bias_vars:
            assert type(lay_old) is type(lay_new), \
                "`gbplearner_prev` and `gbplearner_new` have different architectures."
        if isinstance(lay_new, GBPConvLayer):
            if lay_new.use_bias:
                # lay_old.update_bias_marginals()  # In case some messages not yet gathered

                # Get posterior marginals over biases
                # Rescale both the precision and info by specified factor (rescaling variance)
                posterior_eta = lay_old.bias_vars.eta * prec_rescale_factor
                posterior_Lambda = lay_old.bias_vars.Lambda * prec_rescale_factor
                if alpha is not None:
                    # Make prior from weighted mix of old posterior and init prior
                    mean = posterior_eta / posterior_Lambda
                    std = tf.math.sqrt(1. / posterior_Lambda)
                    mean = alpha * mean + (1. - alpha) * 0.
                    std = alpha * std + (1. - alpha) * next(init_prior_sigmas)
                    posterior_eta = mean / std ** 2.
                    posterior_Lambda = 1. / std ** 2.

                # Set weight prior in new graph to weight posterior from old graph
                lay_new.bias_prior_factor.reset_eta_Lambda(posterior_eta, posterior_Lambda)

                # Initialise message from filter variables to recon factors equal to prev posterior
                edge_shp = lay_new.filter_factor.bias_var_edges.var_to_fac_eta.shape
                if lay_new.use_feedforward_factor or lay_new.use_tpose_recon:
                    eta_expand = posterior_eta[None, None, None]
                    Lambda_expand = posterior_Lambda[None, None, None]
                else:
                    eta_expand = posterior_eta[None, None, None, :, None]
                    Lambda_expand = posterior_Lambda[None, None, None, :, None]
                lay_new.filter_factor.bias_var_edges.var_to_fac_eta = tf.broadcast_to(eta_expand, edge_shp)
                lay_new.filter_factor.bias_var_edges.var_to_fac_Lambda = tf.broadcast_to(Lambda_expand, edge_shp)
                print(tf.reduce_mean(posterior_Lambda), 'post lambda mean conv bias')

                # We adjusted the priors, so recompute marginals to keep up to date
                lay_new.update_bias_marginals()

                # Relinearise recon factors around new bias values
                # lay_new.relinearise_factor_bias(lay_old.bias_vars.mu)

        elif isinstance(lay_new, GBPDenseLayer):

            if lay_new.use_bias:
                # lay_old.update_bias_marginals()  # In case some messages not yet gathered

                # Get posterior marginals over biases
                # Rescale both the precision and info by specified factor (rescaling variance)
                posterior_eta = lay_old.bias_vars.eta
                posterior_Lambda = lay_old.bias_vars.Lambda
                if not prec_rescale_conv_only:
                    posterior_eta *= prec_rescale_factor
                    posterior_Lambda *= prec_rescale_factor

                if alpha is not None:
                    # Make prior from weighted mix of old posterior and init prior
                    mean = posterior_eta / posterior_Lambda
                    std = tf.math.sqrt(1. / posterior_Lambda)
                    mean = alpha * mean + (1. - alpha) * 0.
                    std = alpha * std + (1. - alpha) * next(init_prior_sigmas)
                    posterior_eta = mean / std ** 2.
                    posterior_Lambda = 1. / std ** 2.

                # Set weight prior in new graph to weight posterior from old graph
                lay_new.bias_prior_factor.reset_eta_Lambda(posterior_eta, posterior_Lambda)

                # Initialise message from filter variables to recon factors equal to prev posterior
                edge_shp = lay_new.dense_factor.bias_var_edges.var_to_fac_eta.shape
                lay_new.dense_factor.bias_var_edges.var_to_fac_eta = \
                    tf.broadcast_to(posterior_eta[None], edge_shp)
                lay_new.dense_factor.bias_var_edges.var_to_fac_Lambda = \
                    tf.broadcast_to(posterior_Lambda[None], edge_shp)
                print(tf.reduce_mean(posterior_Lambda), 'post lambda mean dense bias')

                # We adjusted the priors, so recompute marginals to keep up to date
                lay_new.update_bias_marginals()

                # Relinearise recon factors around new bias values
                # lay_new.relinearise_factor_bias(lay_old.bias_vars.mu)


def filtering_on_coeffs(gbplearner_prev: [GBPLearner, list, tuple],
                        gbplearner_new: GBPLearner,
                        prec_rescale_factor: float = 1.):
    """
    Takes posterior over coefficient vars in `gbplearner_prev` and
    sets it as a prior on the same variables in `gbplearner_new`
    """
    # Previous posterior may either be as a GBPLearner, or an iterable of bias variables
    # For the latter case need to reformat so same code can be used
    from_coeff_vars = isinstance(gbplearner_prev, (list, tuple))
    if from_coeff_vars:
        gbplearnerprev = dotdict(layers=[])
        coeff_id = 0
        for i, l in enumerate(gbplearner_new.layers):
            if hasattr(l, 'coeff_vars'):
                if l.coeff_vars is not None:
                    print(len(gbplearner_prev), l)
                    gbplearnerprev.layers.append(dotdict({'coeff_vars': dotdict(gbplearner_prev[coeff_id])}))
                    coeff_id += 1
                else:
                    gbplearnerprev.layers.append(None)
            else:
                gbplearnerprev.layers.append(None)
        gbplearner_prev = gbplearnerprev

    # Get posterior marginals over coeffs
    # Set the eta and Lambda for the corresponding prior factors in gbplearner_new
    for lay_old, lay_new in zip(gbplearner_prev.layers, gbplearner_new.layers):
        if not from_coeff_vars:
            assert type(lay_old) is type(lay_new), \
                "`gbplearner_prev` and `gbplearner_new` have different architectures."
        if hasattr(lay_new, 'coeff_vars'):
            if lay_new.coeff_vars is None:
                print(f"Layer of type {type(lay_new)} has coeff vars which are NoneType")
                continue
            if lay_old.coeff_vars.eta.shape != lay_new.coeff_vars.eta.shape:
                print(f'Shape mismatch in coeff filtering: '
                      f'{lay_old.coeff_vars.eta.shape} != {lay_new.coeff_vars.eta.shape}')
                continue
            if not hasattr(lay_new, 'coeff_prior_factor'):
                print(f"Layer of type {type(lay_new)} does not have coeff prior factor")
                continue
            print(f'Doing coeff filtering for layer {lay_new}, {lay_new.coeff_vars.mu} {lay_old.coeff_vars.eta / lay_old.coeff_vars.Lambda}')
            # Get posterior marginals over coeffs
            # Rescale both the precision and info by specified factor (rescaling variance)
            posterior_eta = lay_old.coeff_vars.eta * prec_rescale_factor
            posterior_Lambda = lay_old.coeff_vars.Lambda * prec_rescale_factor

            # Set coeff prior in new graph to coeff posterior from old graph
            lay_new.coeff_prior_factor.reset_eta_Lambda(posterior_eta, posterior_Lambda)

            # Initialise message from filter variables to recon factors equal to prev posterior
            lay_new.coeff_prior_factor.var_edges.var_to_fac_eta = posterior_eta
            lay_new.coeff_prior_factor.var_edges.var_to_fac_Lambda = posterior_Lambda
            print(tf.reduce_mean(posterior_Lambda), 'post lambda mean conv bias')

            # We adjusted the priors, so recompute marginals to keep up to date
            lay_new.update_coeff_marginals()

            # Relinearise recon factors around new bias values
            # lay_new.relinearise_factor_bias(lay_old.bias_vars.mu)


def copy_linearisation_points(gbplearner_new: GBPLearner,
                              to_copy: [list, tuple, None] = None,
                              gbplearner_prev: [GBPLearner, None] = None):
    """
    Copies the value of the linearisation points from
    `gbplearner_prev` to `gbplearner_new`
    """
    assert to_copy is not None or gbplearner_prev is not None, \
        "Must provide either old model or it's linearisation points."
    if gbplearner_prev is None:
        for lin_old, lay_new in zip(to_copy, gbplearner_new.layers):
            to_copy_lay = lin_old
            for tc in to_copy_lay:
                for ttc in range(len(to_copy_lay[tc])):
                    if to_copy_lay[tc][ttc].shape != lay_new.linearisation_points[tc][ttc].shape:
                        print(to_copy_lay[tc][ttc].shape, lay_new.linearisation_points[tc][ttc].shape, 'Shape mismatch')
                        to_copy_lay[tc][ttc] = lay_new.linearisation_points[tc][ttc]  # e.g. if batch size changes
            lay_new.linearisation_points = to_copy_lay
    else:
        for lay_old, lay_new in zip(gbplearner_prev.layers, gbplearner_new.layers):
            assert type(lay_old) is type(lay_new), \
                "`gbplearner_prev` and `gbplearner_new` have different architectures."
            to_copy_lay = lay_old.linearisation_points
            for tc in to_copy_lay:
                for ttc in range(len(to_copy_lay[tc])):
                    if to_copy_lay[tc][ttc].shape != lay_new.linearisation_points[tc][ttc].shape:
                        print(to_copy_lay[tc][ttc].shape, lay_new.linearisation_points[tc][ttc].shape, 'Shape mismatch')
                        to_copy_lay[tc][ttc] = lay_new.linearisation_points[tc][ttc]  # e.g. if batch size changes
            lay_new.linearisation_points = to_copy_lay
