# coding=utf-8
"""Training GBP continually on a sequence of video frames"""
import os
from contextlib import redirect_stdout
import sys
import tensorflow as tf

from core.inference.gbp.learner import GBPLearner, filtering_on_weights, filtering_on_biases
from core.utils.utils import set_allow_gpu_mem_growth
from experiments.utils.run_exp import exp_main
from core.utils.utils import denoise_eval
from core.utils.plot import plot_denoising

from experiments.video.config import get_config
from experiments.video.data import load_and_corrupt_video
from experiments.utils.utils import pretty_print_args, create_results_dir, create_or_concat_npy

set_allow_gpu_mem_growth()


def plot_model_denoising(model,
                         plot_dir,
                         train_batch_id,
                         train_itr,
                         clean_frames,
                         test_batch_id=None,
                         test_itr=None,
                         max_n_imgs=4):
    plot_denoising(img_corrupt=model.layers[0].pixel_obs_factor.obs,
                   img_denoised=model.layers[0].input_vars.mu,
                   itr=train_itr,
                   img=clean_frames[train_batch_id:train_batch_id + 1],
                   plotfile=os.path.join(plot_dir, 'denoise', f'img{train_batch_id}.png'))


def main():
    # Set up results dir
    conf = get_config()
    conf.experiment.results_dir = create_results_dir(conf.experiment.results_dir)

    print(f'\n\nOutput to logdir: {conf.experiment.results_dir}\n\n')

    if conf.tf_deterministic:
        tf.config.experimental.enable_op_determinism()

    train_data_corr, train_data_clean, corr_mask = \
        load_and_corrupt_video(dataset_name=conf.experiment.dataset_name,
                               noise_seed=conf.corruption_seed,
                               noise_dist=conf.corruption_type,
                               frac_noise=conf.frac_noise,
                               get_noise_mask=True)

    if conf.experiment.doing_validation:
        # Only take first two frames for validation
        nval = conf.experiment.n_validation_data
        train_data_corr = train_data_corr[:nval]
        train_data_clean = train_data_clean[:nval]
        if corr_mask is not None:
            corr_mask = corr_mask[:nval]

    conf.n_train_batches = conf.n_train_batches or train_data_corr.shape[0]
    conf.experiment.examples_per_class_train = train_data_corr.shape[0]
    conf.experiment.n_classes = 1

    score_filepath = os.path.join(conf.experiment.results_dir, 'denoise_scores.npy')
    if conf.experiment.save_denoised_video:
        video_denoised_filepath = os.path.join(conf.experiment.results_dir, 'video_denoised.npy')
        video_denoised_std_filepath = os.path.join(conf.experiment.results_dir, 'video_denoised_std.npy')
        video_noisy_filepath = os.path.join(conf.experiment.results_dir, 'video_noisy.npy')
        create_or_concat_npy(arr=train_data_corr,
                             npy_path=video_noisy_filepath)
        video_clean_filepath = os.path.join(conf.experiment.results_dir, 'video_clean.npy')
        create_or_concat_npy(arr=train_data_clean,
                             npy_path=video_clean_filepath)

    def eval_fn(model, train_itr, train_batch_id, args_test=None, test_data=None):
        to_prepend = [train_batch_id, train_itr]
        psnr, ll, mse = denoise_eval(denoised=model.layers[0].input_vars,
                                frame_clean=train_data_clean[train_batch_id],
                                crop_border=conf.eval_crop_border)
        psnr_masked, ll_masked, mse_masked = \
            denoise_eval(denoised=model.layers[0].input_vars,
                         frame_clean=train_data_clean[train_batch_id],
                         mask=corr_mask[train_batch_id:train_batch_id + 1, ..., None],
                         crop_border=conf.eval_crop_border)
        create_or_concat_npy(arr=[to_prepend + [psnr.numpy(),
                                                ll.numpy(),
                                                psnr_masked.numpy(),
                                                ll_masked.numpy()]],
                             npy_path=score_filepath)
        if conf.experiment.save_denoised_video:
            create_or_concat_npy(arr=model.layers[0].input_vars.mu,
                                 npy_path=video_denoised_filepath)
            create_or_concat_npy(arr=model.layers[0].input_vars.sigma,
                                 npy_path=video_denoised_std_filepath)
        print(f'Train batch {train_batch_id}, itr {train_itr}, '
              f'LL {ll}, PSNR {psnr}, '
              f'LL (mask) {ll_masked}, PSNR (mask) {psnr_masked}')
        return None, None, None

    def plot_denoise_extra(*args, **kwargs):
        return plot_model_denoising(*args, clean_frames=train_data_clean, **kwargs)

    def run_fn(args, tr_data, gbp_model=None, evalfn=None):
        print(f'\n\nOutput to logdir: {args.experiment.results_dir}\n\n')
        sys.stderr = open(os.path.join(args.experiment.results_dir, 'err.txt'), 'w', buffering=1)
        with open(os.path.join(args.experiment.results_dir, 'out.txt'), 'w', buffering=1) as fo:
            with redirect_stdout(fo):
                return exp_main(args=args,
                                gbp_net_init=gbp_model,
                                train_data=tr_data,
                                eval_fn=evalfn,
                                plot_subdirs=['denoise'],
                                plot_extra=plot_denoise_extra if args.experiment.plot_denoise_recon else None)

    run_fn(args=conf,
           tr_data=train_data_corr,
           evalfn=eval_fn)


if __name__ == '__main__':
    main()
