#!/usr/bin/env bash
N_REPEATS=1
GPU='0'


for rep in $(seq $N_REPEATS)
do
  cmd="CUDA_VISIBLE_DEVICES='${GPU}' PYTHONPATH=${PYTHONPATH}:../../ python run.py
    --n-iters-per-test-batch=1500
    --n-test-eval-breaks=1
    --test-only
    --load-checkpoint-dir=/home/$(whoami)/code/gbp_learning_anon/experiments/mnist/results/standard_mnist/final_fix_test_params_bs10/2/train/models/batch_2400
    --config-filepath=/home/$(whoami)/code/gbp_learning_anon/experiments/mnist/results/standard_mnist/final_fix_test_params_bs10/2/conf.json
    --logdir=None"


#  if [ "$rep" -eq "1" ]; then
#    # On the first run make plots of diagnostics
#    cmd="${cmd} --plot-weights --plot-coeffs --plot-convergence"
#  fi

  printf "PID of this script: %s\n" "$$"
  printf "Repeat %s of %s" "$rep" "$N_REPEATS"
  printf "\nRunning %s \n\n" "${cmd//[$'\t\r\n']}"
  eval "${cmd//[$'\t\r\n']}"
done