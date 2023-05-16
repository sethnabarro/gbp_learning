# coding=utf-8
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds

from core.utils.utils import load_video, corrupt_image


VIDEO_PATHS = {'walking': '../../inputs/walking_5fps/',
               'duck': '../../inputs/mallard-fly/',
               'bear': '../../inputs/bear/',
               'monkeys': '../../inputs/monkeys/'}


def load_and_corrupt_video(dataset_name, noise_seed=None, frac_noise=None, noise_dist=None, get_noise_mask=True):
    noise_seed = noise_seed or 876
    frac_noise = frac_noise or 0.1
    w, h = None, None
    if '_' in dataset_name:
        splits = dataset_name.split('_')
        if len(splits) == 2:
            dataset_name, h = splits
            h = int(h)
        elif len(splits) == 3:
            dataset_name, h, w = splits
            h = int(h)
            w = int(w)

    video_path = VIDEO_PATHS[dataset_name]
    dir_path = os.path.dirname(os.path.abspath(__file__))
    vid_clean = load_video(os.path.join(dir_path, video_path), H=h, W=w)[0]

    vid_corr, mask = corrupt_image(vid_clean,
                                   frac_noise=frac_noise,
                                   noise_seed=noise_seed,
                                   noise_dist=noise_dist or 'uniform',
                                   get_noise_mask=get_noise_mask)
    return vid_corr.numpy(), vid_clean, mask
