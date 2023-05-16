# coding=utf-8
import argparse

from core.factors.utils import get_nonlin_fns


def get_argparser(inc_filtering=False,
                  inc_classification=False,
                  inc_testing=False,
                  inc_plotting=False,
                  inc_segmentation=False,
                  inc_validation=False):
    ap = argparse.ArgumentParser()

    # Core options
    ap.add_argument('--profile', action='store_true',  # Tensorboard profiling
                    help='Generates tensorboard profile. '
                         'Will only run for 3 iters if profiling.')
    ap.add_argument('--logdir', type=str, required=True,  # Plotting
                    help='Where to store results and plots')
    ap.add_argument('--config-filepath', type=str)
    ap.add_argument('--load-checkpoint-dir', type=str)
    ap.add_argument('--not-static-graph', action='store_true')

    ap.add_argument('--examples-per-class-train')     # How many imgs in train set to subsample
    ap.add_argument('--batchsize-train', type=int)
    ap.add_argument('--n-iters-per-train-batch', type=int, default=1000)   # Number of iters per batch
    ap.add_argument('--n-train-batches', type=int)
    ap.add_argument('--n-train-eval-breaks', type=int, default=1)       # How many times to stop and eval during training
    ap.add_argument('--n-train-plot-breaks', type=int)                  # How many times to stop and plot during training

    # Model/gbp/inference core config
    ap.add_argument('--layer-schedule', nargs='+', required=False, type=int)
    ap.add_argument('--random-layer-schedule', action='store_true')
    ap.add_argument('--momentum', type=float, default=0.8)
    ap.add_argument('--relin-freq', type=int, default=1)
    ap.add_argument('--inference', type=str, default='gbp', choices=('gbp', 'gd', 'backprop'))

    # Model saving frequency
    ap.add_argument('--checkpoint-frequency-batches', type=int)

    # For exact repeatability
    ap.add_argument('--tf-deterministic', action='store_true')

    ap = add_init_args(ap)
    ap = add_factor_args(ap)

    if inc_filtering:
        ap = add_filtering_args(ap)
    if inc_classification:
        ap = add_classification_args(ap)
    if inc_validation:
        ap = add_validation_args(ap)
    if inc_testing:
        ap = add_testing_args(ap)
    if inc_plotting:
        ap = add_plotting_args(ap)
    if inc_segmentation:
        ap = add_segmentation_args(ap)

    return ap


def add_filtering_args(argp):
    argp.add_argument('--no-filtering', action='store_true')
    argp.add_argument('--copy-lin-points', action='store_true')
    argp.add_argument('--no-filter-biases', action='store_true',
                      help='If specified, filtering between train and '
                           'test does not include bias variables.')
    argp.add_argument('--precision-rescaling', type=float, default=1.,
                      help='Used when filtering. Factor to rescale prev. posterior precision by, '
                           'before adding it as prior as new graph.')
    argp.add_argument('--precision-rescaling-test', type=float, default=1.,
                      help='As with `--precision-rescaling` but for test time')
    argp.add_argument('--precision-rescaling-conv-only', action='store_true',
                    help='Only apply the precision rescaling to conv layers, not dense layers')
    return argp


def add_factor_args(argp):
    argp.add_argument('--nonlin', type=str,
                      choices=list(get_nonlin_fns().keys()),
                      default='leaky_relu')
    argp.add_argument('--factors-recon-feedforward', action='store_true')
    argp.add_argument('--factors-recon-sigma', type=float, default=0.5)  # Factor strengths
    argp.add_argument('--factors-recon-sigma-layers', type=float, nargs='+',
                      help='σs for recon factors in each layer')
    argp.add_argument('--factors-recon-N-rob', type=float, default=4.)
    argp.add_argument('--factors-recon-no-bias', action='store_true')
    argp.add_argument('--factors-pixel-obs-sigma', type=float, default=0.02)
    argp.add_argument('--factors-pixel-obs-N-rob', type=float, default=10.)
    argp.add_argument('--factors-avg-pool-sigma', type=float, default=0.05)
    argp.add_argument('--factors-softmax-obs-sigma', type=float, default=0.02)
    argp.add_argument('--factors-dense-sigma', type=float, default=0.1)
    argp.add_argument('--factors-dense-coeff-prior-sigma', type=float, default=1.)
    argp.add_argument('--factors-dense-weight-prior-sigma', type=float, default=1.)
    argp.add_argument('--factors-bias-prior-sigma', type=float, default=1.)
    argp.add_argument('--factors-coeff-prior-sigma', type=float, default=1.)
    argp.add_argument('--factors-recon-coeff-prior-N-rob', type=float)
    argp.add_argument('--factors-dense-coeff-prior-N-rob', type=float)
    argp.add_argument('--factors-weight-prior-sigma', type=float, default=1.)
    argp.add_argument('--factors-segment-weight-prior-sigma', type=float)
    argp.add_argument('--factors-coeff-prior-N-rob-layers', type=float, nargs='+')
    argp.add_argument('--factors-coeff-prior-sigma-layers', type=float, nargs='+')
    argp.add_argument('--factors-recon-weight-prior-sigma-layers', type=float, nargs='+')
    argp.add_argument('--factors-recon-fac-to-var-chunksize', type=int, default=4)
    argp.add_argument('--factors-dense-fac-to-var-chunksize', type=int, default=32)
    return argp


def add_init_args(argp):
    # Random seeds
    argp.add_argument('--weight-seed', type=int, default=267)
    argp.add_argument('--coeff-seed', type=int, default=176)
    argp.add_argument('--data-seed', type=int, default=678)
    argp.add_argument('--shuffle-batches', action='store_true')

    # Init distribution stds
    argp.add_argument('--coeff-init-std', type=float, default=0.2)
    argp.add_argument('--weight-init-std', type=float, default=0.2)
    return argp


def add_testing_args(argp):
    # If there is train/test split, these args may be useful
    argp.add_argument('--n-test-batches', type=int)
    argp.add_argument('--n-test-eval-breaks', type=int, default=1)        # How many times to stop and eval during test time inference
    argp.add_argument('--batchsize-test', type=int)
    argp.add_argument('--examples-per-class-test')
    argp.add_argument('--n-iters-per-test-batch', type=int, default=1000)
    argp.add_argument('--profile-test', action='store_true')
    argp.add_argument('--test-only', action='store_true')
    return argp


def add_validation_args(argp):
    argp.add_argument('--validation', action='store_true')
    argp.add_argument('--n-validation-data', type=int)
    return argp


def add_classification_args(argp):
    argp.add_argument('--no-dense-softmax-layer', action='store_true',
                      help='No dense projection + softmax layers. If given - '
                           'classification done by external linear classifier.')
    argp.add_argument('--n-classes', type=int, default=10)
    return argp


def add_plotting_args(argp):
    argp.add_argument('--plot-coeffs', action='store_true')
    argp.add_argument('--plot-weights', action='store_true')
    argp.add_argument('--plot-convergence', action='store_true')
    argp.add_argument('--plot-every-batch', action='store_true')
    argp.add_argument('--plot-train-batch-freq', type=int)
    argp.add_argument('--plot-test-batch-freq', type=int)
    return argp


def add_segmentation_args(argp):
    argp.add_argument('--segment-class-and-background', action='store_true')
    argp.add_argument('--every-pixel-img-classification', action='store_true')
    argp.add_argument('--factors-segment-obs-sigma', type=float, default=0.01)
    argp.add_argument('--factors-segment-prior-sigma', type=float, default=2.)
    return argp
