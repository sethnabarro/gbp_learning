# coding=utf-8
import tensorflow as tf

from core.factors import AdditiveFilterFactorSumDecomp
from core.inference.gbp.layers.base import GBPLayer
from core.inference.base import ConvLayer
from core.utils.utils import patchify_image, make_tf_filter_shp


class GBPConvLayer(GBPLayer, ConvLayer):
    def __init__(self, *args, **kwargs):
        ConvLayer.__init__(self, *args, **kwargs)
        GBPLayer.__init__(self,
                          input_vars=kwargs['input_vars'],
                          coeff_vars=kwargs['coeff_vars'])

    @property
    def edges(self):
        edgs = [self.filter_factor.filter_var_edges,
                self.filter_factor.input_var_edges,
                self.filter_factor.coeff_var_edges,
                self.filter_prior_factor.var_edges]
        if self.is_first_layer:
                edgs += [self.pixel_obs_factor.var_edges]
        edgs += [self.coeff_prior_factor.var_edges]
        if self.use_bias:
            edgs.append(self.filter_factor.bias_var_edges)
            edgs.append(self.bias_prior_factor.var_edges)
        return edgs

    def intra_layer_inference_iter(self, itr):
        # self.state = state
        if self.use_pairwise_smoothing:
            self.update_pixel_obs_factor_to_input_message()
            self.update_input_to_filter_factor_message()
            self.update_filter_factor_to_variable_messages()
            self.update_marginals()
            self.relinearise_factors(itr)
            return

        # Set new relinarisation point (at specified freq)
        # self.update_marginals()
        # self.relinearise_factors(itr)

        if self.use_filter_coeffs and not self.fixed_coeffs:
            self.update_coeff_marginals()
            self.update_coeff_to_filter_factor_message()

        # Pixel vars to filter factors
        self.update_input_marginals()
        self.update_input_to_filter_factor_message()

        self.update_filter_factor_to_variable_messages()

        if not self.fixed_params:
            self.update_filter_marginals()

            # Filter variables to filter factor
            self.update_filter_to_filter_factor_message()

            if self.use_bias:
                self.update_bias_marginals()
                self.update_bias_to_filter_factor_message()

            if self.use_filter_coeffs and not self.fixed_coeffs:
                self.update_coeff_marginals()

        if self.is_first_layer:
            # Pixel observation factors to pixel variables
            self.update_pixel_obs_factor_to_input_message()

        if self.use_robust_coeff_prior_factor and not self.fixed_coeffs:
            self.update_coeff_prior_factor_to_coeff_message()

        # Update marginals
        self.update_marginals()
        self.relinearise_factors(itr)

    @property
    def state(self):
        state = self.get_base_state()
        state.append(self.filter_factor.state)
        if self.is_first_layer:
            state.append(self.pixel_obs_factor.state)
        if self.use_robust_coeff_prior_factor:
            state.append(self.coeff_prior_factor.state)
        state.append(self.filter_prior_factor.state)
        if self.use_bias:
            state.append(self.bias_prior_factor.state)
        return state

    @property
    def named_state(self):
        state = self.get_base_named_state()
        state.append(('filter_factor', self.filter_factor.named_state))
        if self.is_first_layer:
            state.append(('pixel_obs_factor', self.pixel_obs_factor.named_state))
        if self.use_robust_coeff_prior_factor:
            state.append(('coeff_prior_factor', self.coeff_prior_factor.named_state))
        state.append(('filter_prior_factor', self.filter_prior_factor.state))
        if self.use_bias:
            state.append(('bias_prior_factor', self.bias_prior_factor.state))
        return state

    @state.setter
    def state(self, new_state):
        if self.use_pairwise_smoothing:
            nvars = 1
        else:
            nvars = 2 + \
                int(self.use_filter_coeffs) + \
                int(self.use_bias)
        self.set_base_state(new_state[:nvars])
        lastvarid = nvars
        self.filter_factor.state = new_state.pop(lastvarid)
        if self.is_first_layer:
            self.pixel_obs_factor.state = new_state.pop(lastvarid)
        if self.use_robust_coeff_prior_factor:
            self.coeff_prior_factor.state = new_state.pop(lastvarid)
        self.filter_prior_factor.state = new_state.pop(lastvarid)
        if self.use_bias:
            self.bias_prior_factor.state = new_state.pop(lastvarid)

    @property
    def variables(self):
        variables = (self.input_vars, self.filter_vars)
        if self.use_filter_coeffs:
            variables += (self.coeff_vars,)
        if self.use_bias:
            variables += (self.bias_vars,)
        return variables

    @property
    def params(self):
        params = (self.filter_vars,)
        if self.use_bias:
            params += (self.bias_vars,)
        return params

    def update_input_to_filter_factor_message(self):
        central_inp_id = int(self.filter_vars.k_size ** 2 / 2)
        for ptype in ('eta', 'Lambda'):
            # Create patches of image, last dim is the vector of intensities in each neighbourhood (length FxF)
            input_margs = getattr(self.input_vars, ptype)

            factor_to_input_mess = getattr(self.filter_factor.input_var_edges, f'fac_to_var_{ptype}')
            if self.use_tpose_recon:
                input_to_factor = input_margs - factor_to_input_mess
                # if self.is_first_layer:
                #     if self.pixel_obs_factor.mask is not None:
                #         input_to_factor = tf.where(tf.cast(self.pixel_obs_factor.mask, tf.float32) == 0., input_margs, input_to_factor)
            else:
                input_mess_marginals = patchify_image(input_margs,
                                                      ksize_x=self.filter_vars.k_size,
                                                      stride=self.stride)  # B x (H - F + 1) x (W - F + 1) x (F ** 2)

                # Subtract incoming message from the patch factor who outgoing message is directed at
                if self.use_decomp_filter_factors:
                    if self.use_feedforward_factor:
                        input_to_factor = input_mess_marginals[..., None, :] - factor_to_input_mess
                    # elif self.use_sum_filter_factors and not self.filter_factor.relative_to_centre:
                    #     input_to_factor = input_mess_marginals[..., None] - factor_to_input_mess[..., :1]
                    elif self.use_sum_filter_factors:
                        input_to_factor_outer = input_mess_marginals[..., None] - factor_to_input_mess[..., :1]
                        input_to_factor_inner = \
                            input_mess_marginals[..., central_inp_id][..., None, None] - factor_to_input_mess[..., 1:]
                        input_to_factor = tf.concat([input_to_factor_outer, input_to_factor_inner], axis=-1)
                    elif self.use_pairwise_smoothing:
                        input_to_factor_outer = input_mess_marginals - factor_to_input_mess[..., 0]
                        input_to_factor_inner = input_mess_marginals[..., central_inp_id:central_inp_id + 1] - factor_to_input_mess[..., 1]
                        input_to_factor = tf.stack([input_to_factor_outer, input_to_factor_inner], axis=-1)
                    else:
                        input_to_factor_outer = input_mess_marginals[..., None, :] - factor_to_input_mess[..., 0]
                        if self.filter_factor.relative_to_centre:
                            input_to_factor_inner = \
                                input_mess_marginals[..., None, central_inp_id][..., None] - factor_to_input_mess[..., 1]
                            input_to_factor = tf.stack([input_to_factor_outer, input_to_factor_inner], axis=-1)
                        else:
                            input_to_factor = input_to_factor_outer[..., None]
                elif self.use_sum_filter_factors:
                    input_to_factor = input_mess_marginals - factor_to_input_mess
                elif self.use_pairwise_smoothing:
                    input_to_factor = input_mess_marginals[..., None, :, None] - factor_to_input_mess
                else:
                    input_to_factor = input_mess_marginals[..., None, :] - factor_to_input_mess  # B x (H - F + 1) x (W - F + 1) x K x (F ** 2)
            # if ptype == 'Lambda':
            #     tf.assert_greater(input_to_factor, -1e-9, message=f'bias to filtfac Lambda neg {input_to_factor.shape}')
            # tf.debugging.check_numerics(input_to_factor, message=f'bias to filtfac {ptype}')
            setattr(self.filter_factor.input_var_edges, f'var_to_fac_{ptype}', input_to_factor)

    def update_filter_to_filter_factor_message(self):
        # Subtract incoming messages from marginals to get outgoing
        for ptype in ('eta', 'Lambda'):
            incoming = getattr(self.filter_factor.filter_var_edges, f'fac_to_var_{ptype}')
            outgoing = getattr(self.filter_vars, f'{ptype}_flat')[None, None, None] - incoming

            # If outgoing msg is 0. then don't update
            outgoing = tf.where((tf.reduce_sum(tf.abs(outgoing)) == 0.)[None, None, None, None, None, None],
                                getattr(self.filter_factor.filter_var_edges, f'var_to_fac_{ptype}'),
                                outgoing)
            setattr(self.filter_factor.filter_var_edges, f'var_to_fac_{ptype}', outgoing)

    def update_coeff_to_filter_factor_message(self):
        # Subtract incoming messages from marginals to get outgoing
        for ptype in ('eta', 'Lambda'):
            incoming = getattr(self.filter_factor.coeff_var_edges, f'fac_to_var_{ptype}')
            if self.use_tpose_recon:
                margs = getattr(self.coeff_vars, f'{ptype}')
                pad = self.filter_vars.k_size // 2 * 2
                margs_pad = tf.pad(margs, [[0, 0], [pad, pad], [pad, pad], [0, 0]], constant_values=0.)
                margs_patched = patchify_image(margs_pad, ksize_x=self.filter_vars.k_size, stride=self.stride)
                outgoing = margs_patched[..., None, :, :] - incoming
                setattr(self.filter_factor.coeff_var_edges, f'var_to_fac_{ptype}', outgoing)
                continue
            elif self.use_feedforward_factor:
                margs = getattr(self.coeff_vars, f'{ptype}')
            elif self.use_sum_filter_factors and self.use_decomp_filter_factors:
                margs = getattr(self.coeff_vars, f'{ptype}')[..., None, :, None]
                # outgoing = tf.transpose(outgoing, (0, 1, 2, 4, 3))
            else:
                if self.use_sum_filter_factors:
                    incoming = incoming[..., None]
                margs = getattr(self.coeff_vars, f'{ptype}')[..., None]
            if self.filter_factor.coeff_padding > 0:
                pad = self.filter_factor.coeff_padding
                margs = margs[:, pad:-pad, pad:-pad]
            setattr(self.filter_factor.coeff_var_edges, f'var_to_fac_{ptype}', margs - incoming)

    def update_bias_to_filter_factor_message(self):
        # Subtract incoming messages from marginals to get outgoing
        for ptype in ('eta', 'Lambda'):
            incoming = getattr(self.filter_factor.bias_var_edges, f'fac_to_var_{ptype}')
            if self.use_feedforward_factor or self.use_tpose_recon:
                outgoing = getattr(self.bias_vars, f'{ptype}')[None, None, None] - incoming
            else:
                outgoing = getattr(self.bias_vars, f'{ptype}')[None, None, None, :, None] - incoming
            # if ptype == 'Lambda':
            #     tf.assert_greater(outgoing, -1e-9, message=f'bias to filtfac Lambda neg {outgoing.shape}, min {tf.reduce_min(outgoing)}')
            # tf.debugging.check_numerics(outgoing, message=f'bias to filtfac {ptype}')
            setattr(self.filter_factor.bias_var_edges, f'var_to_fac_{ptype}', outgoing)

    def update_pixel_obs_factor_to_input_message(self):
        self.pixel_obs_factor.update_outgoing_messages([self.input_vars.mu])

    def update_coeff_prior_factor_to_coeff_message(self):
        self.coeff_prior_factor.update_outgoing_messages([self.coeff_vars.mu])

    def update_filter_factor_to_variable_messages(self):
        vars_mu = [self.filter_vars.mu, self.input_vars.mu]
        if self.use_filter_coeffs:
            vars_mu.append(self.coeff_vars.mu)
        self.filter_factor.update_outgoing_messages(vars_mu)
        # es = dict(filter=self.filter_factor.filter_var_edges,
        #           coeffs=self.filter_factor.coeff_var_edges,
        #           inputs=self.filter_factor.input_var_edges)

        # for en, e in es.items():
        #     for ptype in ('eta', 'Lambda'):
        #         tf.debugging.check_numerics(getattr(e, f'fac_to_var_{ptype}'), message=f'filtfac to {en} {ptype} {e.shape}')

    def _update_input_marginals(self, return_eta_Lambda: bool = False):
        # Sum messages to inputs from all filters at each neighbourhood
        filt_axis = -3 if self.use_decomp_filter_factors else -2
        to_return = []
        n_pix_channel_in = self.input_vars.shape[-1]
        for ptype in ('eta', 'Lambda'):
            msg_from_filtfacs = getattr(self.filter_factor.input_var_edges,
                                        f'fac_to_var_{ptype}')

            # Some messages from recon factor for segment vars - take only those for pixels
            msg_from_filtfacs = msg_from_filtfacs[:, :, :, :n_pix_channel_in]
            if not self.use_sum_filter_factors and \
                    not self.use_pairwise_smoothing and \
                    not self.use_tpose_recon:
                msg_from_filtfacs = tf.reduce_sum(msg_from_filtfacs, axis=filt_axis)
            if self.is_first_layer:
                msg_lhood = getattr(self.pixel_obs_factor.var_edges, f'fac_to_var_{ptype}')
            else:
                msg_lhood = 0.
            input_marg = self.get_input_marginal(sum_filtfac_msg=msg_from_filtfacs,
                                                 lhood_msg=msg_lhood)
            if return_eta_Lambda:
                to_return.append(input_marg)
            else:
                setattr(self.input_vars, ptype, input_marg)
        if return_eta_Lambda:
            return to_return

    def get_input_marginal(self, sum_filtfac_msg, lhood_msg):
        recfield = self.filter_vars.k_size ** 2

        if self.use_tpose_recon:
            return lhood_msg + sum_filtfac_msg

        if self.use_decomp_filter_factors:
            if self.filter_factor.relative_to_centre and not self.use_bias:
                sum_filtfac_msg = tf.where(tf.range(recfield) == int(recfield / 2),  # Central pixel
                                           sum_filtfac_msg[..., 0] +
                                           tf.reduce_sum(sum_filtfac_msg[..., 1], axis=-1)[..., None],
                                           sum_filtfac_msg[..., 0])
            elif self.use_feedforward_factor:
                sum_filtfac_msg = tf.reduce_sum(sum_filtfac_msg, axis=-2)
            else:
                sum_filtfac_msg = sum_filtfac_msg[..., 0]
            filtfac_msg_depatch = self.depatchify(sum_filtfac_msg)
        else:
            filtfac_msg_depatch = self.depatchify(sum_filtfac_msg)

        return lhood_msg + filtfac_msg_depatch

    def update_filter_marginals(self):
        for ptype in ('eta', 'Lambda'):
            filtfac_msg_in = getattr(self.filter_factor.filter_var_edges,
                                         f'fac_to_var_{ptype}')
            priorfac_msg_in = getattr(self.filter_prior_factor.var_edges,
                                      f'fac_to_var_{ptype}')

            ptype_marg = tf.reduce_sum(filtfac_msg_in, axis=[0, 1, 2]) + priorfac_msg_in

            ptype_marg_rs = make_tf_filter_shp(ptype_marg, self.filter_vars.k_size)
            setattr(self.filter_vars, ptype, ptype_marg_rs)

    def _update_coeff_marginals(self, return_eta_Lambda: bool = False):
        to_return = []
        for ptype in ('eta', 'Lambda'):
            prior_incoming = getattr(self.coeff_prior_factor.var_edges,
                                     f'fac_to_var_{ptype}')
            if self.use_tpose_recon:
                incoming = \
                   getattr(self.filter_factor.coeff_var_edges, f'fac_to_var_{ptype}')
                inshp = incoming.shape.as_list()
                wshapexy = self.coeff_vars.shape[1:3]
                incoming = tf.reshape(incoming, inshp[:3] + [inshp[3] * inshp[4], inshp[5]])
                incoming = self.depatchify(incoming)
                incoming = tf.reshape(incoming, incoming.shape.as_list()[:3] + inshp[3:-1])
                npad = self.filter_vars.k_size // 2 * 2 * self.stride  # x2 because on both sides
                incoming = tf.reduce_sum(incoming, axis=-2)[:, npad:npad + wshapexy[0], npad:npad + wshapexy[1]]
            elif self.use_feedforward_factor:
                incoming = getattr(self.filter_factor.coeff_var_edges,
                                   f'fac_to_var_{ptype}')
            else:
                incoming = getattr(self.filter_factor.coeff_var_edges,
                                   f'fac_to_var_{ptype}')
                incoming = tf.reduce_sum(incoming, axis=-3)

                if self.use_decomp_filter_factors:
                    incoming = tf.reduce_sum(incoming, axis=-1)

            if self.filter_factor.coeff_padding > 0:
                pad = self.filter_factor.coeff_padding
                incoming = tf.pad(incoming, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

            # tf.debugging.check_numerics(incoming, f'conv coeff compute inc, {incoming.shape} {ptype}')
            # tf.debugging.check_numerics(prior_incoming, f'conv coeff compute prior, {prior_incoming.shape}')

            ptype_marg = incoming + prior_incoming

            if return_eta_Lambda:
                to_return.append(ptype_marg)
            else:
                setattr(self.coeff_vars, ptype, ptype_marg)
        if return_eta_Lambda:
            return to_return

    def update_bias_marginals(self):
        for ptype in ('eta', 'Lambda'):
            msg_from_filtfacs = getattr(self.filter_factor.bias_var_edges,
                                        f'fac_to_var_{ptype}')
            sum_axis = [0, 1, 2] if self.use_tpose_recon else [0, 1, 2, -1]
            msg_from_filtfacs = tf.reduce_sum(msg_from_filtfacs, axis=sum_axis)   # sum over factors decomposed over nhood
            msg_prior = getattr(self.bias_prior_factor.var_edges, f'fac_to_var_{ptype}')
            setattr(self.bias_vars, ptype, msg_from_filtfacs + msg_prior)

    def update_marginals(self):
        self.update_input_marginals()
        if self.use_pairwise_smoothing:
            return
        if not self.fixed_params:
            self.update_filter_marginals()
            if self.use_bias:
                self.update_bias_marginals()
        if self.use_filter_coeffs and not self.fixed_coeffs:
            self.update_coeff_marginals()

    def energy(self, as_numpy=False, robust=True, sum_all=True, components=None, filters=None, inputs=None, coeffs=None, bias=None, segvars=None):
        # TODO: Should be a cleaner way to implement the below
        return ConvLayer.energy(self,
                                as_numpy,
                                robust,
                                sum_all,
                                components,
                                filters,
                                inputs,
                                coeffs,
                                bias,
                                segvars)

    def fix_params(self):
        assert isinstance(self.filter_factor, AdditiveFilterFactorSumDecomp), \
            "Fixing filters not yet supported with any other filter factor types"
        self.fixed_params = True
        self.filter_factor.fixed_params = True

    def unfix_params(self):
        assert isinstance(self.filter_factor, AdditiveFilterFactorSumDecomp), \
            "Fixing filters not yet supported with any other filter factor types"
        self.fixed_params = False
        self.filter_factor.fixed_params = False

    def fix_coeffs(self):
        assert isinstance(self.filter_factor, AdditiveFilterFactorSumDecomp), \
            "Fixing coeffs not yet supported with any other filter factor types"
        self.fixed_coeffs = True
        self.filter_factor.fixed_coeffs = True

    def fix_inputs(self):
        assert isinstance(self.filter_factor, AdditiveFilterFactorSumDecomp), \
            "Fixing inputs not yet supported with any other filter factor types"
        self.fixed_inputs = True
        self.filter_factor.fixed_inputs = True
