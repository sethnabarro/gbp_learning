# coding=utf-8
"""Saving and loading models"""
import copy
import json
import numpy as np
import os
import tensorflow as tf
from typing import Iterable

from experiments.utils.graph_setup_utils import init_layers
from core.inference.gbp.layers import GBPLayer, GBPConvLayer, GBPDenseLayer
from core.inference.gbp.learner import GBPLearner
from core.utils.data import get_batch
from core.utils.utils import dotdict


def mkdir_if_new(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


def save_gbp_net(gbp_net: GBPLearner,
                 savedir: str, args: dotdict,
                 n_batch_so_far: int,
                 n_iter_current_batch: int,
                 input_image: [tf.Tensor, None] = None,
                 input_mask: [tf.Tensor, None] = None,
                 output_obs: [tf.Tensor, None] = None,
                 params_only: bool = True):
    mkdir_if_new(savedir)

    args_save = copy.deepcopy(args)
    args_save.save_params_only = params_only

    # We have model config specified in the args dict
    # But need to store how many batches have been used so far
    args_save.n_batch_so_far = n_batch_so_far
    args_save.n_iter_current_batch = n_iter_current_batch

    # Write them to model_spec.json config file in `savedir`
    conf_path = os.path.join(savedir, 'model_spec.json')
    with open(conf_path, mode='w') as model_spec:
        json.dump(args_save, model_spec)

    # Save input and output training observations
    if input_image is not None:
        np.save(os.path.join(savedir, 'input.npy'), input_image)
    if input_mask is not None:   # For inpainting
        np.save(os.path.join(savedir, 'input_mask.npy'), input_mask)
    if output_obs is not None:
        np.save(os.path.join(savedir, 'output.npy'), output_obs)

    # Now to save the tensors
    # Create subdir for each layer
    n_layers = len(gbp_net.layers)
    for lay_id in range(n_layers):
        laydir = os.path.join(savedir, f'{lay_id}')
        mkdir_if_new(laydir)

    # Use `layer.state` to get all necessary tensors to save
    # This is a nested list. Save its elements in a directory structure
    # which reflects the nesting of that list
    param_names = ('weight_marginals', 'bias_marginals')

    def recursive_save(state_name, state, subdir):
        subdir = os.path.join(subdir, state_name)
        mkdir_if_new(subdir)
        if isinstance(state, tf.Tensor):
            if (not params_only) or any(p in subdir for p in param_names):
                np.save(os.path.join(subdir, 'arr.npy'), state.numpy())
            else:
                os.rmdir(subdir)
        elif isinstance(state[0], tf.Tensor):
            if (not params_only) or any(p in subdir for p in param_names):
                for i, s in enumerate(state):
                    np.save(os.path.join(subdir, f'arr{i}.npy'), s.numpy())
            else:
                os.rmdir(subdir)
        elif isinstance(state, Iterable):
            if params_only and '_factor' in subdir:
                os.rmdir(subdir)
            else:
                for n, s in state:
                    recursive_save(n, s, subdir)

    for l, lay in enumerate(gbp_net.layers):
        lay.update_marginals()
        for sn, st in lay.named_state:
            if sn == 'input_marginals' and l > 0:
                # Input vars in this layer same as coeffs (outputs) from previous
                # No need to save twice
                pass
            recursive_save(sn, st, os.path.join(savedir, f'{l}'))


def to_dotdict(adict):
    dd = dotdict()

    def _convert_item(itm):
        if isinstance(itm, dict):
            return to_dotdict(itm)
        elif isinstance(itm, Iterable) and not isinstance(itm, str):
            return type(itm)([_convert_item(i) for i in itm])
        else:
            return itm

    for k, v in adict.items():
        dd[k] = _convert_item(v)
    return dd


def get_saved_state(fresh_layers: Iterable[GBPLayer], loaddir: str):
    totens = tf.convert_to_tensor

    def recursive_load(state_name, state, subdir, state_list):
        subdir = os.path.join(subdir, state_name)

        if isinstance(state, (tf.Tensor, np.ndarray)):
            state_list.append(totens(np.load(os.path.join(subdir, 'arr.npy'))))
        elif isinstance(state[0], (tf.Tensor, np.ndarray)):
            states = []
            for i, s in enumerate(state):
                states.append(totens(np.load(os.path.join(subdir, f'arr{i}.npy'))))
            state_list.append(states)
        elif isinstance(state, Iterable):
            state_list.append([])
            for n, s in state:
                recursive_load(n, s, subdir, state_list[-1])

    state = []
    for l, lay in enumerate(fresh_layers):
        state_ls = []
        for sn, st in lay.named_state:
            if sn == 'input_marginals' and l > 0:
                # Input vars in this layer same as coeffs (outputs) from previous
                # No need to save twice
                pass
            recursive_load(sn, st, os.path.join(loaddir, f'{l}'), state_list=state_ls)
        state.append(state_ls)
    return state


def load_layer_params(layer: [GBPConvLayer, GBPDenseLayer], lay_dir: str):
    weight_dir = os.path.join(lay_dir, 'weight_marginals')
    weight_vars = layer.filter_vars if isinstance(layer, GBPConvLayer) else layer.weight_vars
    weight_vars.eta = np.load(os.path.join(weight_dir, 'arr0.npy'))
    weight_vars.Lambda = np.load(os.path.join(weight_dir, 'arr1.npy'))
    if layer.use_bias:
        bias_dir = os.path.join(lay_dir, 'bias_marginals')
        layer.bias_vars.eta = np.load(os.path.join(bias_dir, 'arr0.npy'))
        layer.bias_vars.Lambda = np.load(os.path.join(bias_dir, 'arr1.npy'))


def load_gbp_net(loaddir: str, input_img=None, output=None):
    # Load the model spec file
    with open(os.path.join(loaddir, 'model_spec.json'), mode='r') as model_spec:
        args = to_dotdict(json.load(model_spec))

    params_only = args.save_params_only

    # Load input and output if saved
    def _load_arr_if_saved(path):
        if os.path.exists(path):
            return np.load(path)
        else:
            return None

    input_img = input_img or _load_arr_if_saved(os.path.join(loaddir, 'input.npy')) if input_img is None else input_img
    input_mask = _load_arr_if_saved(os.path.join(loaddir, 'input_mask.npy'))
    output = _load_arr_if_saved(os.path.join(loaddir, 'output.npy')) if output is None else output

    # Init layer and model objects
    def init_model():
        layers = init_layers(args,
                             img_obs=input_img,
                             weight_init_std=args.init_weight_std,
                             weight_init_seed=args.experiment.weight_init_seed,
                             coeff_init_seed=args.experiment.coeff_init_seed,
                             img_mask=input_mask,
                             output_class_obs=output)
        return GBPLearner(layers)
    gbp_net = init_model()

    if params_only:
        for layr_id in range(len(gbp_net.layers)):
            if isinstance(gbp_net.layers[layr_id], (GBPConvLayer, GBPDenseLayer)):
                load_layer_params(gbp_net.layers[layr_id], os.path.join(loaddir, str(layr_id)))

    else:
        # Extract state from saved files
        state = get_saved_state(gbp_net.layers, loaddir)
        gbp_net.state = copy.deepcopy(state)

        # Test if messages are consistent with the marginals
        gbp_net_update = init_model()
        gbp_net_update.state = state

        # Now update marginals - what do the messages say they should be?
        for layr in gbp_net_update.layers:
            layr.update_marginals()

        # Are the marginals according to the messages the same as the saved marginals?
        assert_model_marginals_equal(gbp_net, gbp_net_update)

    return gbp_net, args


def assert_model_marginals_equal(model_a: GBPLearner,
                                 model_b: GBPLearner,
                                 params_only: bool = False):
    # First extract marginals from both models
    marginals_a = []
    for layr in model_a.layers:
        params_or_vars = layr.params if params_only else layr.variables
        marginals_a.append([[v.eta, v.Lambda] for v in params_or_vars])

    marginals_b = []
    for layr in model_b.layers:
        params_or_vars = layr.params if params_only else layr.variables
        marginals_b.append([[v.eta, v.Lambda] for v in params_or_vars])

    # Check the two sets of marginals are equal
    for margs_a_layr, margs_b_layr in zip(marginals_a, marginals_b):
        for marg_a, marg_b in zip(margs_a_layr, margs_b_layr):
            tf.assert_equal(marg_a[0], marg_b[0],
                            "Saved eta not consistent with saved messages")
            tf.assert_equal(marg_a[1], marg_b[1],
                            "Saved Lambda not consistent with saved messages")


def load_checkpoint(dataset,  checkpoint_dir, args, classes_sub=None):
    # TODO fix input_img and output args below, needed to load model but not used
    # Load from checkpoint
    x_batch, maybe_y_batch, maybe_y_batch_seg, _ = \
        get_batch(dataset,
                  n_examples=1,
                  n_examples_so_far=0,
                  n_classes=args.experiment.n_classes,
                  n_seg_classes=args.experiment.n_seg_classes,
                  classes_sub=classes_sub,
                  )
    return load_gbp_net(loaddir=checkpoint_dir,
                        input_img=x_batch,
                        output=maybe_y_batch)
