# coding=utf-8
import tensorflow as tf

from core.factors import get_filter_to_filter_diversity_factor_messages, AdditiveFilterFactorSumDecomp
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

        if self.use_weight_norm_factor:
            self.update_weight_to_non_zero_weight_norm_factor_message()
            self.update_non_zero_weight_norm_factor_to_weight_message()

        if self.is_first_layer and self.doing_segmentation and self.segment_obs_factor is not None:
            self.update_segment_var_to_segment_obs_factor_message()

        if self.use_filter_coeffs and not self.fixed_coeffs:
            self.update_coeff_marginals()
            self.update_coeff_to_filter_factor_message()

        if self.is_first_layer and self.doing_segmentation and self.segment_obs_factor is not None:
            self.update_segment_obs_factor_to_segment_var_message()
            self.update_segment_var_marginals()
            # self.update_segment_var_to_segment_obs_factor_message()

        # Pixel vars to filter factors
        self.update_input_marginals()
        self.update_input_to_filter_factor_message()

        self.update_filter_factor_to_variable_messages()

        if self.use_component_vars:
            self.update_component_marginals()
            self.update_component_to_component_factor_message()
            self.update_component_factor_to_variable_messages()
            self.update_coeff_marginals()
            self.update_filter_marginals()

            self.update_filter_to_component_factor_message()
            self.update_coeff_to_component_factor_message()
            self.update_component_factor_to_variable_messages()
            self.update_component_marginals()
            # self.update_coeff_marginals()
            # self.update_filter_marginals()
            self.update_component_to_filter_factor_message()

        else:
            if not self.fixed_params:
                self.update_filter_marginals()

                # Filter variables to filter factor
                self.update_filter_to_filter_factor_message()

                if self.use_bias:
                    self.update_bias_marginals()
                    self.update_bias_to_filter_factor_message()

            if self.use_filter_coeffs and not self.fixed_coeffs:
                self.update_coeff_marginals()

        if self.use_filter_diversity_factor:
            self.update_filter_diversity_factor_to_filter_message()

        if self.use_filter_normalisation_factor:
            # self.update_filter_normalisation_factor_to_filter_message()
            self.update_filter_normalisation_factor_to_filter_message()
            self.update_filter_to_filter_normalisation_factor_message()

        if self.use_filter_diversity_factor:
            self.update_filter_to_filter_diversity_factor_message()

        if self.is_first_layer:
            # Pixel observation factors to pixel variables
            self.update_pixel_obs_factor_to_input_message()

            if self.is_first_layer and self.doing_segmentation and self.segment_obs_factor is not None:
                self.update_segment_obs_factor_to_segment_var_message()

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
        if self.use_component_vars:
            state.append(self.component_factor.state)
        if self.use_robust_coeff_prior_factor:
            state.append(self.coeff_prior_factor.state)
        if self.is_first_layer and self.doing_segmentation and \
                self.segment_obs_factor is not None:
            state.append(self.segment_obs_factor.state)
        return state

    @property
    def named_state(self):
        state = self.get_base_named_state()
        state.append(('filter_factor', self.filter_factor.named_state))
        if self.is_first_layer:
            state.append(('pixel_obs_factor', self.pixel_obs_factor.named_state))
        if self.use_component_vars:
            state.append(('component_factor', self.component_factor.named_state))
        if self.use_robust_coeff_prior_factor:
            state.append(('coeff_prior_factor', self.coeff_prior_factor.named_state))
        if self.is_first_layer and self.doing_segmentation and \
                self.segment_obs_factor is not None:
            state.append(('segment_obs_factor', self.segment_obs_factor.named_state))
        return state

    @state.setter
    def state(self, new_state):
        if self.use_pairwise_smoothing:
            nvars = 1
        else:
            nvars = 2 + \
                int(self.use_filter_coeffs) + \
                int(self.use_component_vars) + \
                int(self.doing_segmentation and self.is_first_layer) +\
                int(self.use_bias)
        self.set_base_state(new_state[:nvars])
        lastvarid = nvars
        self.filter_factor.state = new_state.pop(lastvarid)
        if self.is_first_layer:
            self.pixel_obs_factor.state = new_state.pop(lastvarid)
        if self.use_component_vars:
            self.component_factor.state = new_state.pop(lastvarid)
        if self.use_robust_coeff_prior_factor:
            self.coeff_prior_factor.state = new_state.pop(lastvarid)
        if self.doing_segmentation and self.is_first_layer and \
                self.segment_obs_factor is not None:
            self.segment_obs_factor.state = new_state.pop(lastvarid)

    @property
    def variables(self):
        variables = (self.input_vars, self.filter_vars)
        if self.use_filter_coeffs:
            variables += (self.coeff_vars,)
        if self.use_component_vars:
            variables += (self.component_vars,)
        if self.doing_segmentation and self.is_first_layer:
            variables += (self.segment_vars,)
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
            if self.is_first_layer and self.doing_segmentation:
                input_margs = tf.concat((input_margs, getattr(self.segment_vars, ptype)), axis=-1)

            factor_to_input_mess = getattr(self.filter_factor.input_var_edges, f'fac_to_var_{ptype}')
            if self.use_per_pixel_recon:
                input_to_factor = input_margs - factor_to_input_mess
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
            # if ptype == 'Lambda':
            #     tf.assert_greater(outgoing, -1e-9, message=f'filter to filtfac Lambda neg {outgoing.shape}')
            # tf.debugging.check_numerics(outgoing, message=f'filter to filtfac {ptype}')

            # If outgoing msg is 0. then don't update
            outgoing = tf.where((tf.reduce_sum(tf.abs(outgoing)) == 0.)[None, None, None, None, None, None],
                                getattr(self.filter_factor.filter_var_edges, f'var_to_fac_{ptype}'),
                                outgoing)
            setattr(self.filter_factor.filter_var_edges, f'var_to_fac_{ptype}', outgoing)

    def update_coeff_to_filter_factor_message(self):
        # Subtract incoming messages from marginals to get outgoing
        for ptype in ('eta', 'Lambda'):
            incoming = getattr(self.filter_factor.coeff_var_edges, f'fac_to_var_{ptype}')
            if self.use_per_pixel_recon:
                margs = getattr(self.coeff_vars, f'{ptype}')
                pad = self.filter_vars.k_size // 2 + 1
                margs_pad = tf.pad(margs, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
                margs_patched = patchify_image(margs_pad, ksize_x=self.filter_vars.k_size, stride=self.stride)
                outgoing = margs_patched[..., None, :, :] - incoming
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
            # if ptype == 'Lambda':
            #     tf.assert_greater(margs - incoming, -1e-9, message=f'coeff to filtfac Lambda neg {margs.shape}')
            # tf.debugging.check_numerics(margs - incoming, message=f'coeff to filtfac {ptype}')
            setattr(self.filter_factor.coeff_var_edges, f'var_to_fac_{ptype}', margs - incoming)

    def update_bias_to_filter_factor_message(self):
        # Subtract incoming messages from marginals to get outgoing
        for ptype in ('eta', 'Lambda'):
            incoming = getattr(self.filter_factor.bias_var_edges, f'fac_to_var_{ptype}')
            if self.use_feedforward_factor:
                outgoing = getattr(self.bias_vars, f'{ptype}')[None, None, None] - incoming
            else:
                outgoing = getattr(self.bias_vars, f'{ptype}')[None, None, None, :, None] - incoming
            # if ptype == 'Lambda':
            #     tf.assert_greater(outgoing, -1e-9, message=f'bias to filtfac Lambda neg {outgoing.shape}')
            # tf.debugging.check_numerics(outgoing, message=f'bias to filtfac {ptype}')
            setattr(self.filter_factor.bias_var_edges, f'var_to_fac_{ptype}', outgoing)

    def update_filter_to_filter_diversity_factor_message(self):
        filt_to_div_fac_eta, filt_to_div_fac_Lambda = \
            get_filter_to_filter_diversity_factor_messages(self.filter_diversity_factor,
                                                           self.filter_vars.eta_flat,
                                                           self.filter_vars.Lambda_flat)
        self.filter_diversity_factor.var_edges.var_to_fac_eta = filt_to_div_fac_eta
        self.filter_diversity_factor.var_edges.var_to_fac_Lambda = filt_to_div_fac_Lambda

    def update_filter_diversity_factor_to_filter_message(self):
        self.filter_diversity_factor.update_outgoing_messages([self.filter_vars.mu])

    def update_filter_to_filter_normalisation_factor_message(self):
        self.filter_normalisation_factor.var_edges.var_to_fac_eta = \
            self.filter_vars.eta_flat - self.filter_normalisation_factor.var_edges.fac_to_var_eta
        self.filter_normalisation_factor.var_edges.var_to_fac_Lambda = \
            self.filter_vars.Lambda_flat - self.filter_normalisation_factor.var_edges.fac_to_var_Lambda

    def update_filter_normalisation_factor_to_filter_message(self):
        self.filter_normalisation_factor.update_outgoing_messages([self.filter_vars.mu])

    def sum_filter_diversity_factor_to_filter_messages(self, ptype):
        msgs_in = getattr(self.filter_diversity_factor.var_edges, f'fac_to_var_{ptype}')
        fieldsize = self.filter_vars.shape[0] * self.filter_vars.shape[1]
        msg_sum = tf.reduce_sum(msgs_in, axis=0)[..., -fieldsize:]
        msg_sum += tf.reduce_sum(msgs_in, axis=1)[..., :fieldsize]
        return msg_sum

    def update_component_to_filter_factor_message(self):
        self.filter_factor.component_var_edges.var_to_fac_eta = \
            self.component_vars.eta - self.filter_factor.component_var_edges.fac_to_var_eta
        self.filter_factor.component_var_edges.var_to_fac_Lambda = \
            self.component_vars.Lambda - self.filter_factor.component_var_edges.fac_to_var_Lambda

    def update_component_to_component_factor_message(self):
        outgoing = self.component_vars.eta - self.component_factor.component_var_edges.fac_to_var_eta
        self.component_factor.component_var_edges.var_to_fac_eta = outgoing
        outgoing = self.component_vars.Lambda - self.component_factor.component_var_edges.fac_to_var_Lambda
        self.component_factor.component_var_edges.var_to_fac_Lambda = outgoing

    def update_filter_to_component_factor_message(self):
        outgoing = self.filter_vars.eta_flat[None, None, None] - self.component_factor.filter_var_edges.fac_to_var_eta
        self.component_factor.filter_var_edges.var_to_fac_eta = outgoing
        outgoing = self.filter_vars.Lambda_flat[None, None, None] - self.component_factor.filter_var_edges.fac_to_var_Lambda
        self.component_factor.filter_var_edges.var_to_fac_Lambda = outgoing

    def update_coeff_to_component_factor_message(self):
        outgoing = self.coeff_vars.eta[..., None, :, None] - self.component_factor.coeff_var_edges.fac_to_var_eta
        self.component_factor.coeff_var_edges.var_to_fac_eta = outgoing
        outgoing = self.coeff_vars.Lambda[..., None, :, None] - self.component_factor.coeff_var_edges.fac_to_var_Lambda
        self.component_factor.coeff_var_edges.var_to_fac_Lambda = outgoing

    def update_component_factor_to_variable_messages(self):
        self.component_factor.update_outgoing_messages([self.filter_vars.mu,
                                                        self.coeff_vars.mu,
                                                        self.component_vars.mu])

    def update_pixel_obs_factor_to_input_message(self):
        self.pixel_obs_factor.update_outgoing_messages([self.input_vars.mu])

    def update_segment_obs_factor_to_segment_var_message(self):
        self.segment_obs_factor.update_outgoing_messages([self.segment_vars.mu])

    def update_segment_var_to_segment_obs_factor_message(self):
        for mtype in ('eta', 'Lambda'):
            incoming = getattr(self.segment_obs_factor.var_edges, f'fac_to_var_{mtype}')
            marginal = getattr(self.segment_vars, f'{mtype}')
            outgoing = marginal - incoming
            setattr(self.segment_obs_factor.var_edges, f'var_to_fac_{mtype}', outgoing)

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

    def update_non_zero_weight_norm_factor_to_weight_message(self):
        weights_mu = [self.coeff_vars.mu]
        self.non_zero_weight_norm_factor.update_outgoing_messages(weights_mu)

    def update_weight_to_non_zero_weight_norm_factor_message(self):
        self.non_zero_weight_norm_factor.var_edges.var_to_fac_eta = \
            self.coeff_vars.eta - self.non_zero_weight_norm_factor.var_edges.fac_to_var_eta
        self.non_zero_weight_norm_factor.var_edges.var_to_fac_Lambda = \
            self.coeff_vars.Lambda - self.non_zero_weight_norm_factor.var_edges.fac_to_var_Lambda

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
            if not self.use_sum_filter_factors and not self.use_pairwise_smoothing:
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

        if self.use_per_pixel_recon:
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
        return lhood_msg + self.depatchify(sum_filtfac_msg)

    def update_filter_marginals(self):
        for ptype in ('eta', 'Lambda'):
            if self.use_component_vars:
                filtfac_msg_in = getattr(self.component_factor.filter_var_edges,
                                         f'fac_to_var_{ptype}')
            else:
                filtfac_msg_in = getattr(self.filter_factor.filter_var_edges,
                                         f'fac_to_var_{ptype}')
            priorfac_msg_in = getattr(self.filter_prior_factor.var_edges,
                                      f'fac_to_var_{ptype}')

            ptype_marg = tf.reduce_sum(filtfac_msg_in, axis=[0, 1, 2]) + priorfac_msg_in
            if self.use_filter_diversity_factor:
                ptype_marg += self.sum_filter_diversity_factor_to_filter_messages(ptype)
            if self.use_filter_normalisation_factor:
                ptype_marg += getattr(self.filter_normalisation_factor.var_edges,
                                      f'fac_to_var_{ptype}')

            ptype_marg_rs = make_tf_filter_shp(ptype_marg, self.filter_vars.k_size)
            setattr(self.filter_vars, ptype, ptype_marg_rs)

    def update_component_marginals(self):
        for ptype in ('eta', 'Lambda'):
            incoming_prior = getattr(self.component_prior_factor.var_edges,
                                     f'fac_to_var_{ptype}')
            incoming_consist = getattr(self.component_factor.component_var_edges,
                               f'fac_to_var_{ptype}')
            incoming_filtfac = getattr(self.filter_factor.component_var_edges,
                               f'fac_to_var_{ptype}')
            ptype_marg = incoming_prior + incoming_filtfac + incoming_consist

            setattr(self.component_vars, ptype, ptype_marg)

    def _update_coeff_marginals(self, return_eta_Lambda: bool = False):
        to_return = []
        for ptype in ('eta', 'Lambda'):
            prior_incoming = getattr(self.coeff_prior_factor.var_edges,
                                     f'fac_to_var_{ptype}')
            if self.use_component_vars:
                incoming = getattr(self.component_factor.coeff_var_edges,
                                   f'fac_to_var_{ptype}')
                sum_ax = [-1, -3] if self.use_decomp_filter_factors else [-1]
                incoming = tf.reduce_sum(incoming, axis=sum_ax)
            elif self.use_per_pixel_recon:
                incoming = \
                   getattr(self.filter_factor.coeff_var_edges, f'fac_to_var_{ptype}')
                inshp = incoming.shape.as_list()
                wshapexy = self.coeff_vars.shape[1:3]
                incoming = tf.reshape(incoming, inshp[:3] + [inshp[3] * inshp[4], inshp[5]])
                npad = self.filter_vars.k_size // 2 * 2 * self.stride  # x2 because on both sides
                incoming = self.depatchify(incoming,
                                              nstridey=incoming.shape[1],
                                              nstridex=incoming.shape[2],
                                              ntargy=incoming.shape[1] + npad,
                                              ntargx=incoming.shape[2] + npad)[:, npad:wshapexy[0] + npad, npad:wshapexy[1] + npad]
                incoming = tf.reshape(incoming, incoming.shape.as_list()[:3] + inshp[3:-1])
                incoming = tf.reduce_sum(incoming, axis=-2)
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

            if self.use_weight_norm_factor:
                norm_factor_incoming = getattr(self.non_zero_weight_norm_factor.var_edges,
                                               f'fac_to_var_{ptype}')
                ptype_marg += norm_factor_incoming
            if return_eta_Lambda:
                to_return.append(ptype_marg)
            else:
                setattr(self.coeff_vars, ptype, ptype_marg)
        if return_eta_Lambda:
            return to_return

    def update_segment_var_marginals(self):
        # Sum messages to inputs from all filters at each neighbourhood
        assert self.is_first_layer, "Only have segmentation vars in first layer"
        assert self.use_sum_filter_factors, "Segmentation currently only supported with sum recon factors"
        n_pix_channel_in = self.input_vars.mu.shape[-1]
        for ptype in ('eta', 'Lambda'):
            msg_from_filtfacs = getattr(self.filter_factor.input_var_edges,
                                        f'fac_to_var_{ptype}')[:, :, :, n_pix_channel_in:]
            if self.segment_obs_factor is None:
                msg_lhood = 0.
            else:
                msg_lhood = getattr(self.segment_obs_factor.var_edges, f'fac_to_var_{ptype}')
            seg_marg = self.get_input_marginal(sum_filtfac_msg=msg_from_filtfacs,
                                               lhood_msg=msg_lhood)
            msg_prior = getattr(self.segment_prior_factor.var_edges, f'fac_to_var_{ptype}')
            setattr(self.segment_vars, ptype, seg_marg + msg_prior)

    def update_bias_marginals(self):
        for ptype in ('eta', 'Lambda'):
            msg_from_filtfacs = getattr(self.filter_factor.bias_var_edges,
                                        f'fac_to_var_{ptype}')
            msg_from_filtfacs = tf.reduce_sum(msg_from_filtfacs, axis=[0, 1, 2, -1])   # sum over factors decomposed over nhood
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
        if self.use_component_vars:
            self.update_component_marginals()
        if self.doing_segmentation and self.is_first_layer:
            self.update_segment_var_marginals()
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

    # def deterministic_approx_conv_transpose(self):
    #     coeffs = self.coeff_vars.mu
    #     filts = self.filter_vars.mu
    #     fs = filts.shape[0]
    #     filters_flat = flatten_filters(filts)
    #     coeffsTfilters = tf.linalg.einsum('abcd,edf->abcef', coeffs, filters_flat, optimize='optimal')
    #     bias_patches = self.bias_vars.mu[None, None, None, :, None]
    #     patches = self.filter_factor.nonlin(coeffsTfilters + bias_patches)
    #     depatch = self.depatchify_static(patches, fs, self.stride)
    #
    #     # Have to normalise by number of factors connected to each pixel (roughly)
    #     depatch /= ((fs ** 2.) / (self.stride ** 2.))
    #     return depatch
