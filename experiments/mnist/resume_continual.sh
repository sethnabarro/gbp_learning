#!/usr/bin/env bash
# Queue config - ignored if not running on queue
# declare a name for this job to be sample_job
#PBS -N mnist_continual_resum
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -lwalltime=24:00:00

N_GPUS=1
N_REPEATS=1
#module purge
module load anaconda3/personal
source activate gbp_env
#module load cuda/11.2.2
#module load cudnn/8.2.4

scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $scriptdir
echo "$(dirname ${scriptdir})/utils/"

if [[ ! -f "$(dirname ${scriptdir})/utils/bash_utils.sh" ]]; then
    # On queue system, need to copy utils.sh to working dir
    source $PBS_O_WORKDIR/experiments/utils/bash_utils.sh
    source $PBS_O_WORKDIR/utils/bash_utils.sh
    source $PBS_O_WORKDIR/../utils/bash_utils.sh
  else
    source "$(dirname ${scriptdir})/utils/bash_utils.sh"   # Includes functions called below
fi
#
python_exec="$(get_python_exec)"
repo_dir="$(get_repo_dir)"
gpus="$(get_gpu_ids $N_GPUS)"


for rep in $(seq $N_REPEATS)
do
#  seed1=123
#  seed2=321
#  seed3=222
  seed1=$RANDOM
  seed2=$RANDOM
  seed3=$RANDOM
  logdir=$(get_results_dir ${repo_dir} "mnist" "split_mnist")
  logdir="${logdir}/validation_no_fix_test_params"

  cmd="CUDA_VISIBLE_DEVICES='${gpus}' PYTHONPATH=${PYTHONPATH}:$repo_dir:$repo_dir/experiments/mnist/ $python_exec $repo_dir/experiments/mnist/run.py
    --config-filepath=/rds/general/user/$(whoami)/home/code/gbp_learning_anon//experiments/mnist/results/split_mnist/final_fix_test_params/7/tr_6_7/conf.json
    --load-checkpoint-dir=/rds/general/user/$(whoami)/home/code/gbp_learning_anon//experiments/mnist/results/split_mnist/final_fix_test_params/7/tr_4_5/train/models/batch_113/
    --remaining-tasks-train='[(6, 7), (8, 9)]'
    --fix-params-for-testing
    --logdir=/rds/general/user/$(whoami)/home/code/gbp_learning//experiments/mnist/results/split_mnist/final_fix_test_params/7/
    "

#  if [ "$rep" -eq "1" ]; then
#    # On the first run make plots of diagnostics
#    cmd="${cmd} --plot-weights --plot-convergence --plot-coeffs"
#  fi

  printf "PID of this script: %s\n" "$$"
  printf "Repeat %s of %s" "$rep" "$N_REPEATS"
  printf "\nRunning %s \n\n" "${cmd//[$'\t\r\n']}"
  eval "${cmd//[$'\t\r\n']}"
done