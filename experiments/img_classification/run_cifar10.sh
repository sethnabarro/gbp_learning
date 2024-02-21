#!/usr/bin/env bash
N_GPUS=1
N_REPEATS=1
ENV_DIR="/home/$(whoami)/envs/anon_gbp_env/"

scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $scriptdir
echo "$(dirname ${scriptdir})/utils/"
source "$(dirname ${scriptdir})/utils/bash_utils.sh"   # Includes functions called below

python_exec="${ENV_DIR}/bin/python"
repo_dir="$(get_repo_dir)"
gpus="$(get_gpu_ids $N_GPUS)"


for rep in $(seq $N_REPEATS)
do
  seed1=$RANDOM
  seed2=$RANDOM
  seed3=$RANDOM
  logdir="${repo_dir}/experiments/img_classification/results/standard_cifar10/"
  mkdir -p "${logdir}/"

  cmd="CUDA_VISIBLE_DEVICES='${gpus}' TF_GPU_ALLOCATOR=cuda_malloc_async XLA_FLAGS=--xla_gpu_cuda_data_dir=/rds/general/user/$(whoami)/home/anaconda3/pkgs/cuda-nvcc-12.2.140-0 PYTHONPATH=${PYTHONPATH}:$repo_dir:$repo_dir/experiments/img_classification/ $python_exec $repo_dir/experiments/img_classification/run.py
    --batchsize-train=50
    --batchsize-test=200
    --n-iters-per-train-batch=500
    --n-iters-per-test-batch=300
    --n-train-eval-breaks=5
    --n-test-eval-breaks=1
    --fix-params-for-testing
    --dataset='cifar10'
    --experiment-type='standard'
    --shuffle-batches
    --plot-train-batch-freq=100
    --plot-test-batch-freq=100
    --input-rescale='zero_one'
    --not-class-balanced-batches-train
    --factors-softmax-obs-sigma=0.01
    --factors-dense-sigma=0.01
    --factors-recon-sigma-layers 0.01
    --factors-avg-pool-sigma=0.01
    --factors-pixel-obs-sigma=0.03
    --factors-recon-N-rob=500000.
    --factors-pixel-obs-N-rob=50000000.
    --factors-dense-weight-prior-sigma=0.15
    --factors-dense-coeff-prior-sigma=2.
    --factors-weight-prior-sigma=0.1
    --factors-bias-prior-sigma=0.1
    --factors-coeff-prior-sigma=3.
    --precision-rescaling=1.
    --precision-rescaling-test=1.
    --xla-compile
    --momentum=0.9
    --dropout=0.5
    --inference='gbp'
    --nonlin='leaky_relu'
    --weight-init-std=0.2
    --coeff-init-std=0.2
    --checkpoint-frequency-batches=50
    --architecture='three_layer_k5_ff_fixed_in_16'
    --logdir=${logdir}
    --weight-seed=${seed1}
    --coeff-seed=${seed2}
    --data-seed=${seed3}"
#  cmd="${cmd} --n-validation-data=9000 --validation"

  if [ "$rep" -eq "1" ]; then
    # On the first run make plots of diagnostics
    cmd="${cmd} --plot-weights --plot-convergence --plot-coeffs"
  fi

  printf "PID of this script: %s\n" "$$"
  printf "Repeat %s of %s" "$rep" "$N_REPEATS"
  printf "\nRunning %s \n\n" "${cmd//[$'\t\r\n']}"
  eval "${cmd//[$'\t\r\n']}"
done