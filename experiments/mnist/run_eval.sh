#!/usr/bin/env bash
# Evaluates a checkpoint from a trained model
N_REPEATS=10
ENV_DIR="/home/$(whoami)/envs/anon_gbp_env/"
RESDIR="/home/$(whoami)/code/gbp_learning/experiments/mnist/results/standard_mnist/ff_cnn_non_asynch/8"

scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $scriptdir
echo "$(dirname ${scriptdir})/utils/"
source "$(dirname ${scriptdir})/utils/bash_utils.sh"   # Includes functions called below

python_exec="${ENV_DIR}/bin/python"
gpus="$(get_gpu_ids $N_GPUS)"

for rep in $(seq $N_REPEATS)
do
  cmd="CUDA_VISIBLE_DEVICES='${gpus}' PYTHONPATH=${PYTHONPATH}:../../ ${python_exec} run.py
    --fix-params-for-testing
    --batchsize-test=200
    --n-test-batches=50
    --mnist-experiment-type='standard_mnist'
    --n-iters-per-test-batch=500
    --plot-test-batch-freq=1
    --test-only
    --inference='gbp'
    --inference-test='gbp'
    --n-test-eval-breaks=1
    --config-filepath=${RESDIR}/conf.json
    --logdir=${RESDIR}
    --load-checkpoint-dir=${RESDIR}/train/models/batch_300
    "

  if [ "$rep" -eq "1" ]; then
    # On the first run make plots of diagnostics
    cmd="${cmd} --plot-weights --plot-coeffs --plot-convergence"
  fi

  printf "PID of this script: %s\n" "$$"
  printf "Repeat %s of %s" "$rep" "$N_REPEATS"
  printf "\nRunning %s \n\n" "${cmd//[$'\t\r\n']}"
  eval "${cmd//[$'\t\r\n']}"
done