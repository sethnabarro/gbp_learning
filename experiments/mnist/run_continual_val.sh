#!/usr/bin/env bash
# Queue config - ignored if not running on queue
# declare a name for this job to be sample_job
#PBS -N mnist_continual_val
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -lwalltime=14:00:00

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
  logdir="${logdir}/validation_fix_test_params/"

  cmd="CUDA_VISIBLE_DEVICES='${gpus}' PYTHONPATH=${PYTHONPATH}:$repo_dir:$repo_dir/experiments/mnist/ $python_exec $repo_dir/experiments/mnist/run.py
    --n-train-batches=30
    --n-test-batches=15
    --examples-per-class-train=1500
    --examples-per-class-test=750
    --fix-params-for-testing
    --validation
    --n-validation-data=10000
    --n-iters-per-train-batch=1000
    --n-iters-per-test-batch=1000
    --n-train-eval-breaks=1
    --n-test-eval-breaks=1
    --shuffle-batches
    --plot-train-batch-freq=200
    --plot-test-batch-freq=200
    --factors-dense-weight-prior-sigma=1.
    --factors-dense-coeff-prior-sigma=1.5
    --factors-weight-prior-sigma=1.
    --factors-coeff-prior-sigma=1.
    --precision-rescaling=1.
    --precision-rescaling-test=1.
    --momentum=0.85
    --nonlin='elu'
    --checkpoint-frequency-batches=50
    --architecture='bignet_k5_32'
    --logdir=${logdir}
    --weight-seed=${seed1}
    --coeff-seed=${seed2}
    --data-seed=${seed3}
    --weight-init-std=0.2
    --coeff-init-std=0.2
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