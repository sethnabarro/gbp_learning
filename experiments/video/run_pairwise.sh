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
  mkdir -p "${logdir}"
  logdir="${logdir}/final_pairwise/"
  cmd="CUDA_VISIBLE_DEVICES='${gpus}' PYTHONPATH=${PYTHONPATH}:$repo_dir:$repo_dir/experiments/video/ $python_exec $repo_dir/experiments/video/run.py
    --n-iters-per-train-batch=200
    --plot-train-batch-freq=1
    --dataset-name='bear_258_454'
    --no-filtering
    --pairwise-smoothing
    --n-iters-final-frame=200
    --n-train-eval-breaks=82
    --factors-pixel-obs-N-rob=0.14
    --factors-pixel-obs-sigma=0.2
    --factors-recon-sigma=1.3
    --factors-recon-N-rob=0.35
    --plot-every-batch
    --plot-denoise-recon
    --plot-train-batch-freq=1
    --momentum=0.7
    --architecture='pairwise'
    --logdir=${logdir}
    --weight-seed=${seed1}
    --coeff-seed=${seed2}
    --corruption-seed=${seed3}
    --n-train-eval-breaks=4"  # 4 for validation, 82 for testing
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