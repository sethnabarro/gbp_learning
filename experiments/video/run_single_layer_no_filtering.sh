#!/usr/bin/env bash
N_GPUS=1
N_REPEATS=10
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

  logdir="${repo_dir}/experiments/video/bear_258_454/"
  mkdir -p "${logdir}/final_conv_transpose"
  logdir="${logdir}/final_conv_transpose/no_filtering_single_layer/"
  cmd="CUDA_VISIBLE_DEVICES='${gpus}' TF_GPU_ALLOCATOR=cuda_malloc_async OMP_NUM_THREADS=4 PYTHONPATH=${PYTHONPATH}:$repo_dir:$repo_dir/experiments/video/ $python_exec $repo_dir/experiments/video/run.py
  --n-iters-per-train-batch=300
  --plot-train-batch-freq=1
  --dataset-name='bear_258_454'
  --no-filtering
  --n-iters-final-frame=300
  --n-train-eval-breaks=82
  --factors-pixel-obs-N-rob=0.2
  --factors-pixel-obs-sigma=0.2
  --factors-recon-sigma-layers 0.1
  --factors-recon-N-rob=1.4
  --factors-upsample-sigma=0.03
  --factors-recon-weight-prior-sigma-layers 0.018
  --factors-coeff-prior-sigma-layers 0.5
  --factors-bias-prior-sigma=0.3
  --precision-rescaling=1.
  --precision-rescaling-test=1.
  --plot-every-batch
  --plot-denoise-recon
  --plot-generative
  --precision-rescaling-conv-only
  --momentum=0.8
  --dropout=0.6
  --xla-compile
  --architecture='conv_tpose_single_layer_4'
  --weight-init-std=0.1
  --eval-crop-border=0
  --coeff-init-std=0.1
  --corruption-seed=${seed1}
  --weight-seed=${seed2}
  --coeff-seed=${seed3}
  --logdir=${logdir}
  --n-train-eval-breaks=82"  # 4 for validation, 82 for testing
#  cmd="${cmd}              # Uncomment for validation
#  --validation
#  --n-validation-data=4
#  "
# --save-denoised-video   # For images of denoised frames

  printf "PID of this script: %s\n" "$$"
  printf "Repeat %s of %s" "$rep" "$N_REPEATS"
  printf "\nRunning %s \n\n" "${cmd//[$'\t\r\n']}"
  eval "${cmd//[$'\t\r\n']}"
done