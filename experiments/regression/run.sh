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
seed3=$RANDOM
logdir=$(get_results_dir ${repo_dir} "regression" "")
logdir="${logdir}/"

cmd="PYTHONPATH=${PYTHONPATH}:$repo_dir:$repo_dir/experiments/regression/ $python_exec $repo_dir/experiments/regression/run.py
    --batchsize-train=90
    --batchsize-test=225
    --inference='gbp'
    --n-iters-per-train-batch=2000
    --n-iters-per-test-batch=1000
    --n-train-eval-breaks=1
    --n-test-batches=1
    --n-test-eval-breaks=1
    --fix-params-for-testing
    --plot-train-batch-freq=1
    --plot-test-batch-freq=1
    --factors-regression-obs-sigma=0.03
    --factors-dense-weight-prior-sigma=7.
    --factors-dense-coeff-prior-sigma-layers 5. 5.
    --factors-weight-prior-sigma=7.
    --factors-last-weight-prior-sigma=7.
    --factors-last-coeff-prior-sigma=1.
    --factors-bias-prior-sigma=2.
    --factors-dense-sigma=0.007
    --momentum=0.8
    --dropout=0.6
    --architecture='two_layer_mlp_16'
    --logdir=${logdir}
    --coeff-init-std=0.5
    --weight-init-std=0.5
    --weight-seed=${seed1}
    --coeff-seed=${seed2}
    --data-seed=${seed3}
    "

printf "PID of this script: %s\n" "$$"
printf "Repeat %s of %s" "$rep" "$N_REPEATS"
printf "\nRunning %s \n\n" "${cmd//[$'\t\r\n']}"
eval "${cmd//[$'\t\r\n']}"
