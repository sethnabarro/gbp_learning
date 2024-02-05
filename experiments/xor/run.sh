#!/usr/bin/env bash
ENV_DIR="/opt/homebrew/Caskroom/miniforge/base/"

scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $scriptdir
echo "$(dirname ${scriptdir})/utils/"
source "$(dirname ${scriptdir})/utils/bash_utils.sh"   # Includes functions called below
python_exec="${ENV_DIR}/bin/python"
repo_dir="$(get_repo_dir)"

seed1=$RANDOM
seed2=$RANDOM
logdir=$(get_results_dir ${repo_dir} "xor" "")
logdir="${logdir}/"

cmd="PYTHONPATH=${PYTHONPATH}:$repo_dir:$repo_dir/experiments/xor/ $python_exec $repo_dir/experiments/xor/run.py
  --batchsize-train=4
  --batchsize-test=400
  --inference='gbp'
  --n-iters-per-train-batch=600
  --n-iters-per-test-batch=300
  --n-train-eval-breaks=1
  --n-test-batches=1
  --n-test-eval-breaks=1
  --fix-params-for-testing
  --plot-train-batch-freq=1
  --plot-test-batch-freq=1
  --factors-weight-prior-sigma=3.
  --fix-params-for-testing
  --factors-dense-coeff-prior-sigma-layers 5. 5.
  --factors-last-weight-prior-sigma=3.
  --factors-last-coeff-prior-sigma=2.
  --factors-bias-prior-sigma=1.
  --factors-softmax-obs-sigma=0.1
  --factors-dense-sigma=0.1
  --momentum=0.7
  --dropout=0.
  --architecture='two_layer_mlp_8'
  --logdir=${logdir}
  --coeff-init-std=0.1
  --weight-init-std=0.1
  --weight-seed=${seed1}
  --coeff-seed=${seed2}
  "

if [ "$rep" -eq "1" ]; then
  # On the first run make plots of diagnostics
  cmd="${cmd} --plot-weights --plot-coeffs --plot-convergence"
fi

printf "PID of this script: %s\n" "$$"
printf "Repeat %s of %s" "$rep" "$N_REPEATS"
printf "\nRunning %s \n\n" "${cmd//[$'\t\r\n']}"
eval "${cmd//[$'\t\r\n']}"
