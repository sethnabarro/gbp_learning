# coding=utf-8
from abc import abstractmethod
import tensorflow as tf
from typing import Union, Iterable

from core.factors import UnaryFactor, NonLinearUnaryFactor, AdditiveFilterFactor,\
    AdditiveFilterFactorDecomp, AdditiveFilterFactorSum, AdditiveFilterFactorSumDecomp, DiversityFactor, \
    FilterNormalisationFactor, NonZeroWeightNormFactor, ComponentFactor, PairwiseSmoothingFactorDecomp,\
    SegmentationObservationFactor, AdditiveFilterFactorSumPixel, FeedforwardFilterFactor
from core.utils.utils import tf_fn_wrapper, FixedAttributeFieldsObject, get_n_strides, flatten_filters, get_filter_factor_recon
from core.variables import PixelVariable, FilterVariable, CoeffVariable, ComponentVariable, SegmentationVariable, BiasVariable


class Layer(FixedAttributeFieldsObject):
    @property
    @abstractmethod
    def linearisation_points(self):
        raise NotImplementedError('Implemented in child class')

    @linearisation_points.setter
    @abstractmethod
    def linearisation_points(self, new_relin_point):
        raise NotImplementedError('Implemented in child class')


class ConvLayer(Layer):
    def __init__(self,
                 filter_vars: FilterVariable,
                 input_vars: [PixelVariable, CoeffVariable],
                 filter_factor: AdditiveFilterFactor,
                 filter_prior_factor: UnaryFactor,
                 coeff_prior_factor: Union[None, NonLinearUnaryFactor],
                 pixel_obs_factor: Union[UnaryFactor, NonLinearUnaryFactor, None] = None,
                 img_in: Union[tf.Tensor, None] = None,
                 segmentation_obs_factor: Union[None, SegmentationObservationFactor] = None,
                 segmentation_prior_factor: Union[None, UnaryFactor] = None,
                 component_factor: Union[None, ComponentFactor] = None,
                 component_prior_factor: Union[None, UnaryFactor] = None,
                 bias_prior_factor: Union[None, UnaryFactor] = None,
                 non_zero_weight_norm_factor: Union[NonZeroWeightNormFactor, None] = None,
                 filter_diversity_factor: Union[None, DiversityFactor] = None,
                 filter_normalisation_factor: Union[None, FilterNormalisationFactor] = None,
                 img_ground_truth: Union[None, tf.Tensor] = None,
                 stride: int = 1,
                 coeff_vars: Union[None, CoeffVariable] = None,
                 bias_vars: Union[None, BiasVariable] = None,
                 component_vars: Union[None, ComponentVariable] = None,
                 segmentation_vars: Union[None, SegmentationVariable] = None,
                 fixed_params: bool = False,
                 fixed_coeffs: bool = False,
                 fixed_inputs: bool = False,
                 fixed_first_n_channels: int = 0
                 ):
        self.img_in = img_in

        self.is_first_layer = img_in is not None
        if not self.is_first_layer:
            assert pixel_obs_factor is None

        self.filter_vars = filter_vars
        self.input_vars = input_vars
        self.coeff_vars = coeff_vars
        self.component_vars = component_vars
        self.segment_vars = segmentation_vars
        self.bias_vars = bias_vars
        self.doing_segmentation = segmentation_vars is not None
        self.use_component_vars = component_vars is not None
        self.use_filter_coeffs = coeff_vars is not None
        self.use_bias = bias_vars is not None
        self.use_weight_norm_factor = non_zero_weight_norm_factor is not None
        self.use_robust_coeff_prior_factor = False if coeff_prior_factor is None else coeff_prior_factor.N_rob is not None
        self.use_per_pixel_recon = isinstance(filter_factor, AdditiveFilterFactorSumPixel)
        self.use_feedforward_factor = isinstance(filter_factor, FeedforwardFilterFactor)
        self.stride = stride

        self.filter_factor = filter_factor
        self.filter_prior_factor = filter_prior_factor
        self.coeff_prior_factor = coeff_prior_factor
        self.pixel_obs_factor = pixel_obs_factor
        self.bias_prior_factor = bias_prior_factor
        self.component_factor = component_factor
        self.component_prior_factor = component_prior_factor
        self.segment_obs_factor = segmentation_obs_factor
        self.segment_prior_factor = segmentation_prior_factor
        self.non_zero_weight_norm_factor = non_zero_weight_norm_factor
        self.filter_diversity_factor = filter_diversity_factor
        self.filter_normalisation_factor = filter_normalisation_factor
        self.use_filter_diversity_factor = isinstance(filter_diversity_factor, DiversityFactor)
        self.use_filter_normalisation_factor = isinstance(filter_normalisation_factor, FilterNormalisationFactor)
        self.use_decomp_filter_factors = \
            isinstance(filter_factor, AdditiveFilterFactorDecomp) or \
            isinstance(filter_factor, AdditiveFilterFactorSumDecomp) or \
            isinstance(filter_factor, PairwiseSmoothingFactorDecomp)
        self.use_sum_filter_factors = isinstance(filter_factor, AdditiveFilterFactorSum)
        self.use_pairwise_smoothing = isinstance(filter_factor, PairwiseSmoothingFactorDecomp)
        if self.use_pairwise_smoothing:
            self.use_bias = False

        self.ground_truth = img_ground_truth
        self.n_strides_x, self.n_strides_y = \
            get_n_strides(img_in=input_vars.mu, stride=self.stride, ksize=self.filter_vars.k_size)

        self.compiled = None

        self.fixed_params = fixed_params
        self.fixed_coeffs = fixed_coeffs
        self.fixed_inputs = fixed_inputs
        self.fixed_first_n_channels = fixed_first_n_channels

    @property
    def n_input_vars(self):
        inshp = self.input_vars.shape
        return tf.reduce_prod(inshp)

    @property
    def n_filter_vars(self):
        filtshp = self.filter_vars.shape
        return tf.reduce_prod(filtshp)

    @property
    def n_coeff_vars(self):
        if self.use_filter_coeffs:
            coeffshp = self.coeff_vars.shape
            return tf.reduce_prod(coeffshp)
        else:
            return 0

    @property
    def n_bias_vars(self):
        if self.use_bias:
            biasshp = self.bias_vars.shape
            return tf.reduce_prod(biasshp)
        else:
            return 0

    @property
    def n_segment_vars(self):
        if self.doing_segmentation:
            segshp = self.segment_vars.shape
            return tf.reduce_prod(segshp)
        else:
            return 0

    @property
    def n_component_vars(self):
        if self.use_component_vars:
            compshp = self.component_vars.shape
            return tf.reduce_prod(compshp)
        else:
            return 0

    @property
    def n_vars(self):
        return self.n_coeff_vars + \
           self.n_filter_vars + \
           self.n_input_vars + \
           self.n_component_vars +\
           self.n_segment_vars +\
           self.n_bias_vars

    @property
    def fixed_first_n_channels(self):
        return self._fixed_first_n_channels

    @fixed_first_n_channels.setter
    def fixed_first_n_channels(self, nfix):
        self._fixed_first_n_channels = nfix
        self.filter_factor.fixed_first_n_channels = nfix

    def intra_layer_inference_iter(self, *args, **kwargs):
        raise NotImplementedError('intra_layer_inference_iter() should be implemented in child class')

    def get_base_state(self):
        if self.use_pairwise_smoothing:
            return [[self.input_vars.eta, self.input_vars.Lambda]]
        state = [
            [self.input_vars.eta, self.input_vars.Lambda],
            [self.filter_vars.eta, self.filter_vars.Lambda]]
        if self.use_filter_coeffs:
            state.append([self.coeff_vars.eta, self.coeff_vars.Lambda])
        if self.use_component_vars:
            state.append([self.component_vars.eta,
                          self.component_vars.Lambda])
        if self.doing_segmentation and self.is_first_layer:
            state.append([self.segment_vars.eta,
                          self.segment_vars.Lambda])
        if self.use_bias:
            state.append([self.bias_vars.eta,
                          self.bias_vars.Lambda])
        return state

    def get_base_named_state(self):
        if self.use_pairwise_smoothing:
            return [('input_marginals', [self.input_vars.eta, self.input_vars.Lambda]),]
        state = [('input_marginals', [self.input_vars.eta, self.input_vars.Lambda]),
                 ('weight_marginals', [self.filter_vars.eta, self.filter_vars.Lambda])]
        if self.use_filter_coeffs:
            state.append(('coeff_marginals', [self.coeff_vars.eta, self.coeff_vars.Lambda]))
        if self.use_component_vars:
            state.append(('component_marginals',
                          [self.component_vars.eta,
                          self.component_vars.Lambda]))
        if self.doing_segmentation and self.is_first_layer:
            state.append(('segmentation_marginals',
                         [self.segment_vars.eta,
                          self.segment_vars.Lambda]))
        if self.use_bias:
            state.append(('bias_marginals',
                          [self.bias_vars.eta,
                          self.bias_vars.Lambda]))
        return state

    @property
    def state(self):
        return self.get_base_state()

    def set_base_state(self, new_state):
        self.input_vars.eta, self.input_vars.Lambda = new_state[0]
        if self.use_pairwise_smoothing:
            return
        self.filter_vars.eta, self.filter_vars.Lambda = new_state[1]
        v_id = 2
        if self.use_filter_coeffs:
            self.coeff_vars.eta, self.coeff_vars.Lambda = new_state[v_id]
            v_id += 1
        if self.use_component_vars:
            # If using components, must be using weights
            self.component_vars.eta, self.component_vars.Lambda = new_state[v_id]
            v_id += 1
        if self.doing_segmentation and self.is_first_layer:
            self.segment_vars.eta, self.segment_vars.Lambda = new_state[v_id]
            v_id += 1
        if self.use_bias:
            self.bias_vars.eta, self.bias_vars.Lambda = new_state[v_id]
            v_id += 1

    @state.setter
    def state(self, new_state):
        self.set_base_state(new_state)

    def depatchify(self, x_patches, nstridex=None, nstridey=None, ntargx=None, ntargy=None):
        """
        Reconstructs `x_patches` to something image shaped
        """
        return self.depatchify_static(x_patches,
                                      k_size=self.filter_vars.k_size,
                                      stride=self.stride)

    @staticmethod
    def depatchify_static(x_patches, k_size=3, stride=1):
        """
        Reconstructs `x_patches` to something image shaped
        """
        nstridex, nstridey = x_patches.shape[1:3]
        n_channels_in = x_patches.shape[3]
        batch_size = x_patches.shape[0]
        paramshp = [batch_size,
                    nstridex,
                    nstridey,
                    n_channels_in,
                    k_size,
                    k_size]
        x_patches = tf.reshape(x_patches, paramshp)
        recfield = k_size ** 2

        # Get recon img size - rearrange equations in core.utils.get_n_strides()
        ntargx = (nstridex - 1) * stride + k_size
        ntargy = (nstridey - 1) * stride + k_size
        depatched = tf.zeros([batch_size, ntargx, ntargy, n_channels_in])

        if stride == 1:
            # Efficient and clear, but only works for stride==1
            # TODO: Generalise this method to stride != 1
            for v in range(recfield):
                vx, vy = v // k_size, v % k_size
                depatched_xy = tf.zeros_like(depatched)
                padx = [vx, k_size - vx - 1]
                pady = [vy, k_size - vy - 1]
                padding = [[0, 0]] + [padx, pady] + [[0, 0]]
                depatched_xy = depatched_xy + tf.pad(x_patches[..., vx, vy], padding)
                depatched = depatched + depatched_xy

        else:
            # Clunky and slow, works for stride!=1
            stride_ids = tf.meshgrid(tf.range(batch_size),
                                     tf.range(nstridey * stride, delta=stride),
                                     tf.range(nstridex * stride, delta=stride))
            stride_ids = tf.transpose(stride_ids, (2, 1, 3, 0))
            for v in range(recfield):
                for c in range(n_channels_in):
                    vx, vy = v // k_size, v % k_size
                    stride_ids_c = tf.concat([stride_ids, tf.ones_like(stride_ids[..., :1]) * c], axis=-1)
                    depatched = tf.tensor_scatter_nd_add(depatched,
                                                     stride_ids_c + [[0, vx, vy, 0]],
                                                     x_patches[..., c, vx, vy],
                                                     name='depatch_scatter_add')

        return depatched

    def depatchify_Lambda(self, Lambda_patches):
        recfield = self.filter_vars.k_size ** 2
        paramshp = [-1,
                    self.n_strides_y,
                    self.n_strides_x,
                    recfield,
                    recfield]
        x_patches = tf.reshape(Lambda_patches, paramshp)

        out_shp = (self.img_in.shape[1], self.img_in.shape[2],
                   self.img_in.shape[1], self.img_in.shape[2])
        Lambda_out = tf.zeros(out_shp)
        stride_ids = \
            tf.transpose(tf.meshgrid(tf.range(self.n_strides_y * self.stride, delta=self.stride),
                                     tf.range(self.n_strides_x * self.stride, delta=self.stride)), (2, 1, 0))

        for v1 in range(recfield):
            for v2 in range(recfield):
                v1y, v1x = v1 // self.filter_vars.k_size, v1 % self.filter_vars.k_size
                v2y, v2x = v2 // self.filter_vars.k_size, v2 % self.filter_vars.k_size
                idx = tf.concat((stride_ids + [[v1y, v1x]], stride_ids + [[v2y, v2x]]), axis=-1)
                Lambda_out = tf.tensor_scatter_nd_add(Lambda_out,
                                                      idx,
                                                      x_patches[0, :, :, v1, v2])

        return Lambda_out[None, ..., None]  # TODO: allow for batched input

    def relinearise_factor_weights(self, new_weights):
        self.filter_factor.var0[0] = new_weights

    def relinearise_factor_bias(self, new_bias):
        assert self.use_bias, \
            "Attempted to relinearise bias vars in layer without bias"
        self.filter_factor.var0[-1] = new_bias

    def relinearise_factors(self, i):
        relin_points = self.linearisation_points
        self._relinearise_factors(i, relin_points)

    def _relinearise_factors(self, i, relin_points):
        if self.filter_factor.relin_freq is not None:
            self.filter_factor.relinearise(i, relin_points['recon'])
        if isinstance(self.pixel_obs_factor, NonLinearUnaryFactor) and \
                self.pixel_obs_factor.relin_freq is not None:
            self.pixel_obs_factor.relinearise(i, relin_points['pixel_obs'])
        if self.use_filter_diversity_factor:
            self.filter_diversity_factor.relinearise(i, [self.filter_vars.mu])
        if self.use_filter_normalisation_factor:
            self.filter_normalisation_factor.relinearise(i, [self.filter_vars.mu])
        if self.use_weight_norm_factor:
            self.non_zero_weight_norm_factor.relinearise(i, [self.coeff_vars.mu])
        if self.use_component_vars:
            compfac_var0 = [self.filter_vars.mu, self.coeff_vars.mu, self.component_vars.mu]
            self.component_factor.relinearise(i, compfac_var0)
        if self.doing_segmentation and self.is_first_layer and self.segment_obs_factor is not None:
            self.segment_obs_factor.relinearise(i, relin_points['segment_obs'])
        if self.use_robust_coeff_prior_factor:
            self.coeff_prior_factor.relinearise(i, relin_points['coeff_prior'])

    @property
    def linearisation_points(self):
        relin_points = dict()   # Mapping factor_name -> factor_relin_point
        inputs_for_filtfac = self.input_vars.mu
        if self.use_component_vars:
            filtfac_vars_mu = [self.component_vars.mu, inputs_for_filtfac]
        else:
            if self.doing_segmentation and self.is_first_layer:
                inputs_for_filtfac = tf.concat((inputs_for_filtfac, self.segment_vars.mu), axis=-1)
            filtfac_vars_mu = [self.filter_vars.mu, inputs_for_filtfac]
            if self.use_filter_coeffs:
                filtfac_vars_mu.append(self.coeff_vars.mu)
            if self.use_bias:
                filtfac_vars_mu.append(self.bias_vars.mu)
        relin_points['recon'] = filtfac_vars_mu
        if isinstance(self.pixel_obs_factor, NonLinearUnaryFactor) and \
                self.pixel_obs_factor.relin_freq is not None:
            relin_points['pixel_obs'] = [self.input_vars.mu]

        if self.doing_segmentation and self.is_first_layer and self.segment_obs_factor is not None:
            relin_points['segment_obs'] = [self.segment_vars.mu]
        if self.use_robust_coeff_prior_factor:
            relin_points['coeff_prior'] = [self.coeff_vars.mu]

        return relin_points

    @linearisation_points.setter
    def linearisation_points(self, new_relin_points):
        # Give i=0 to force relinearisation
        self._relinearise_factors(i=0, relin_points=new_relin_points)

    def energy(self, as_numpy=False, robust=True, sum_all=True, components=None, filters=None, inputs=None, coeffs=None, bias=None, segvars=None):
        inputs = inputs if inputs is not None else self.input_vars.mu
        filters = filters if filters is not None else self.filter_vars.mu
        if self.doing_segmentation:
            segvars = segvars if segvars is not None else self.segment_vars.mu
        if self.use_component_vars:
            components = components if components is not None else self.component_vars.mu
            coeffs = coeffs if coeffs is not None else self.coeff_vars.mu
            filtfac_vars_mu = [components, inputs]
        else:
            if self.doing_segmentation:
                inputs_for_filtfac = tf.concat([inputs, segvars], axis=-1)
            else:
                inputs_for_filtfac = inputs
            filtfac_vars_mu = [filters, inputs_for_filtfac]
            if self.use_filter_coeffs:
                coeffs = coeffs if coeffs is not None else self.coeff_vars.mu
                filtfac_vars_mu.append(coeffs)
            if self.use_bias:
                bias = self.bias_vars.mu if bias is None else bias
                filtfac_vars_mu.append(bias)
        energies = {}
        energies['filter_pixel'] = self.filter_factor.energy(filtfac_vars_mu, robust=robust, aggregate=True)
        if self.is_first_layer:
            energies['pixel_obs'] = self.pixel_obs_factor.energy([inputs], robust=robust, aggregate=True)
        energies['filter_prior'] = self.filter_prior_factor.energy([flatten_filters(filters)], aggregate=True)
        if self.filter_diversity_factor:
            energies['filter_diversity'] = self.filter_diversity_factor.energy([filters], aggregate=True)
        if self.filter_normalisation_factor:
            energies['filter_normalisation'] = self.filter_normalisation_factor.energy([filters], aggregate=True)
        if self.use_filter_coeffs:
            energies['coeff_prior'] = self.coeff_prior_factor.energy([coeffs], robust=robust, aggregate=True)
        if self.use_weight_norm_factor:
            energies['non_zero_weight_norm'] = self.non_zero_weight_norm_factor.energy([coeffs], aggregate=True)
        if self.use_component_vars:
            energies['component_consist'] = self.component_factor.energy([filters, coeffs, components], aggregate=True)
            energies['component_prior'] = self.component_prior_factor.energy([components], aggregate=True)
        if self.doing_segmentation:
            if self.segment_obs_factor is not None:
                energies['segmentation'] = self.segment_obs_factor.energy([segvars], robust=robust, aggregate=True)
            energies['segmentation_prior'] = self.segment_prior_factor.energy([segvars], robust=robust, aggregate=True)
        if self.use_bias:
            energies['bias_prior'] = self.bias_prior_factor.energy([bias], robust=robust, aggregate=True)
        if as_numpy:
            energies = {n: e.numpy() for n, e in energies.items()}
        if sum_all:
            return tf.reduce_sum(list(energies.values()))
        else:
            total_energy = tf.reduce_sum(list(energies.values()))
            if as_numpy:
                total_energy = total_energy.numpy()
            energies['total'] = total_energy
            return energies


class Learner(object):
    def __init__(self, layers: Iterable[Layer],
                 layer_schedule: Union[Iterable[int], Iterable[Iterable[int]], None] = None,
                 fixed_params: bool = False,
                 fixed_coeffs: bool = False,
                 random_layer_schedule: bool = False):
        self.layers = layers
        self.compiled = None

        if random_layer_schedule:
            self.layer_schedule = 'random'
        else:
            self.layer_schedule = layer_schedule or (list(range(len(layers))) + list(range(1, len(layers) - 1)[::-1]))

        self.fixed_params = fixed_params
        if fixed_params:
            self.fix_layer_params()

        self.fixed_coeffs = fixed_coeffs
        if fixed_coeffs:
            self.fix_layer_coeffs()

    def _run_inference(self,
                       n_iters,
                       state,
                       n_iters_start,
                       filter_factor_recon_only=False,
                       layers=None,
                       callback_fn=None):
        # if layers is not None:
        #     layers_backup = self.layers.copy()
        #     self.layers = [l for i, l in enumerate(self.layers) if i in layers]

        # Close over callback
        def _iter(itr, *args, **kwargs):
            if callback_fn is not None:
                callback_fn(itr)
            out = self._inference_iter(itr, *args, **kwargs)
            return out

        init_vars = [n_iters_start] + state
        out = tf.while_loop(cond=lambda i, *_: i < n_iters + n_iters_start,
                            body=_iter,
                            back_prop=False,
                            loop_vars=init_vars,
                            parallel_iterations=1,
                            maximum_iterations=n_iters)

        # if layers is not None:
        #     for i, l in zip(layers, self.layers):
        #         layers_backup[i] = l
        #     self.layers = layers_backup
        return out

    def _inference_iter(self, itr, *args, **kwargs):
        raise NotImplementedError('_inference_iter() should be implemented '
                                  'in child class.')

    @property
    def variables(self):
        raise NotImplementedError('Implemented in child class')

    def run_gd_inference(self, learning_rate, optim, *args, **kwargs):
        # for lay in range(len(self.layers)):
        #     self.layers[lay].update_marginals()
        tf.random.set_seed(666)
        variables = [[tf.Variable(tf.where(tf.reduce_sum(tf.abs(v.mu)) == 0.,
                                  tf.random.uniform(v.mu.shape, -0.05, 0.05),
                                  v.mu), trainable=True) for v in vlay] for vlay in self.variables]
        variables[0][-1] = tf.Variable(tf.zeros_like(variables[0][-1]))

        opt_types = {'sgd': tf.optimizers.SGD,
                     'adam': tf.optimizers.Adam}
        opt = opt_types[optim](learning_rate=learning_rate)

        def _set_vars(new_vars):
            for l in range(len(self.layers)):
                if len(new_vars) > l:
                    for nv in range(len(self.layers[l].variables)):
                        if len(new_vars[l]) > nv:
                            self.layers[l].variables[nv].eta = new_vars[l][nv] / self.layers[l].variables[nv].sigma ** 2.
                            self.layers[l].variables[nv].Lambda = tf.ones_like(new_vars[l][nv]) / self.layers[l].variables[nv].sigma ** 2.
        ins = None
        for l, layr in enumerate(self.layers):
            if isinstance(layr, ConvLayer):
                coeffs = variables[l][2]
            elif hasattr(layr, 'avg_pool_factor'):
                coeffs = variables[l][1]
            elif hasattr(layr, 'segmentation_factor'):
                pass
            else:
                raise TypeError(f'GD only supported for avg pool, segmentation '
                                f'and conv layers not {type(layr)}')
            if ins is not None:
                variables[l][0] = ins
            ins = coeffs

        def _inference_iter(itr, state):
            self.state = state
            hasgrad = [[True for _ in vr] for vr in variables]
            with tf.GradientTape() as tape:
                E = 0.
                ins = None
                for l, layr in enumerate(self.layers):
                    if isinstance(layr, ConvLayer):
                        if layr.doing_segmentation:
                            inps, filts, coeffs, segs, bias = variables[l]
                        else:
                            inps, filts, coeffs, bias = variables[l]
                            # coeffs = layr.filter_factor.nonlin(layr.filter_factor.conv2d(inps, filts) + bias)
                            segs = None

                        kwargs = dict(sum_all=True, segvars=segs)
                        hasgrad[l][0] = not layr.fixed_inputs
                        hasgrad[l][1] = not layr.fixed_params
                        hasgrad[l][2] = not layr.fixed_coeffs
                        hasgrad[l][-1] = not layr.fixed_params
                        if not layr.fixed_inputs:
                            ins = inps if ins is None else ins
                            kwargs['inputs'] = ins
                        if not layr.fixed_coeffs:
                            kwargs['coeffs'] = coeffs
                        if not layr.fixed_params:
                            kwargs['filters'] = filts
                            kwargs['bias'] = bias
                        ins = coeffs
                    elif hasattr(layr, 'avg_pool_factor'):
                        inps, coeffs = variables[l]
                        ins = inps if ins is None else ins
                        kwargs = dict(sum_all=True, inputs=ins, coeffs=coeffs)
                        ins = coeffs
                    elif hasattr(layr, 'segmentation_factor'):
                        inps, = variables[l] if len(variables[l]) else (None,)
                        ins = inps if ins is None else ins
                        kwargs = dict(sum_all=True, logits=ins)
                    else:
                        raise TypeError(f'GD only supported for avg pool, segmentation '
                                        f'and conv layers not {type(layr)}')

                    E += layr.energy(**kwargs)
            vars_with_grads = [v for vb, hg in zip(variables, hasgrad) for v, h in zip(vb, hg) if h]
            grads = tape.gradient(E, vars_with_grads)
            ops = opt.apply_gradients(grads_and_vars=zip(grads, vars_with_grads))
            with tf.control_dependencies([ops]):
                return [itr + 1] + [self.state]

        if self.compiled is not None:
            # Inference mode may have been different in compiled version
            self.compiled = None

        def _run_inference(i, state):
            j, state = \
                tf.while_loop(cond=lambda itr, *_: itr < kwargs['n_iters'],
                              body=_inference_iter,
                              loop_vars=[i, state],
                              maximum_iterations=kwargs['n_iters'])
            return j, state

        compiled = tf_fn_wrapper(_run_inference,
                                 xla_compile=kwargs['xla_compile'],
                                 use_static_graph=kwargs['use_static_graph'])
        j, self.state = compiled(0, self.state)
        # for i in range(kwargs['n_iters']):
        #     j, variables, self.state = _inference_iter(i, variables, self.state)

        _set_vars(variables)

        for lay in self.layers:
            lay.relinearise_factors(0)   # For the deterministic coefficient estimation in ff recon factors
        return j, self.state

    def run_inference(self, n_iters=100, x_img=None, x_mask=None,
                      maybe_y_label=None, maybe_y_seg_label=None, callback_fn=None,
                      xla_compile=False, use_static_graph=False,
                      filter_factor_recon_only=False, layers=None,
                      n_iters_start=0, gd=False):

        def _callback(itr):
            for l, layer in enumerate(self.layers):
                if not isinstance(layer, ConvLayer):
                    continue
                pix_mu = layer.input_vars.mu
                if filter_factor_recon_only:
                    # TODO: below code only for GBP, refactor
                    pix_mu = get_filter_factor_recon(layer)

                components_mu = layer.component_vars.mu if layer.use_component_vars else None
                filters_mu = layer.filter_vars.mu
                coeffs_mu = layer.coeff_vars.mu if layer.use_filter_coeffs else None
                filters_std = layer.filter_vars.sigma
                callback_fn(itr, pix_mu,
                            components=components_mu,
                            filters=[filters_mu, filters_std],
                            coeffs=coeffs_mu,
                            layer_id=l)
        if callback_fn is not None and n_iters_start == 0:
            _callback(0)

        if x_img is not None:
            self.layers[0].pixel_obs_factor.obs = x_img
        if x_mask is not None:
            self.layers[0].pixel_obs_factor.mask = x_mask
        if maybe_y_label is not None:
            self.layers[-1].softmax_factor.one_hot = maybe_y_label
        if maybe_y_seg_label is not None:
            if self.layers[0].doing_segmentation:
                self.layers[0].segment_obs_factor.obs = maybe_y_seg_label
            elif hasattr(self.layers[-1], 'segmentation_factor'):
                self.layers[-1].segment_factor.obs = maybe_y_seg_label

        if use_static_graph:
            if self.compiled is None:
                self.compiled = tf_fn_wrapper(self._run_inference,
                                              xla_compile=xla_compile,
                                              use_static_graph=use_static_graph)
            out = self.compiled(n_iters, self.state, n_iters_start, layers=layers,
                                filter_factor_recon_only=filter_factor_recon_only)

        else:
            out = self._run_inference(n_iters, self.state,
                                      n_iters_start=n_iters_start,
                                      layers=layers,
                                      filter_factor_recon_only=filter_factor_recon_only,
                                      callback_fn=None if callback_fn is None else _callback)
        self.state = out[1:]
        if callback_fn is not None and n_iters_start > 0:
            _callback(n_iters_start + n_iters - 1)
        return out

    def energy(self, sum_over_layers=False, as_numpy=False, *args, **kwargs):

        if sum_over_layers:
            Edict = self.layers[0].energy(as_numpy=as_numpy, *args, **kwargs)
            for lay in self.layers[1:]:
                Elaydict = lay.energy(*args, **kwargs)
                for Enam, Eterm in Elaydict.items():
                    if Enam in Edict:
                        Edict[Enam] += Eterm
                    else:
                        Edict[Enam] = Eterm
        else:
            Edict = {}
            for layid, lay in enumerate(self.layers):
                kwargs_copy = kwargs.copy()
                lay_kwargs = kwargs_copy[f'layer_{layid}'] if f'layer_{layid}' in kwargs else {}
                kwargs_copy = {k: v for k, v in kwargs_copy.items() if 'layer_' not in k}
                Edict[f'layer_{layid}'] = lay.energy(as_numpy=as_numpy, *args, **lay_kwargs, **kwargs_copy)
            E_total = 0.
            for lname, lE in Edict.items():
                E_total += lE['total'] if 'total' in lE else sum([E for E in lE.values()])
            Edict['total'] = E_total

        return Edict

    @property
    def state(self):
        return [l.state for l in self.layers]

    @state.setter
    def state(self, new_state):
        for lay, ns in zip(self.layers, new_state):
            lay.state = ns

    def fix_layer_params(self):
        raise NotImplementedError('Implemented in child class')

    def fix_layer_coeffs(self):
        raise NotImplementedError('Implemented in child class')

    def fix_pixels(self):
        raise NotImplementedError('Implemented in child class')
