#!/usr/bin/env bash


# Queue config - ignored if not running on queue
# declare a name for this job to be sample_job
#PBS -N video_denoise_pairwise
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -lwalltime=06:00:00

N_GPUS=1
N_REPEATS=10
module load anaconda3/personal
source activate gbp_env

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
  logdir=$(get_results_dir ${repo_dir} "video" "bear_256")
  logdir="${logdir}/final_pairwise/"
  cmd="CUDA_VISIBLE_DEVICES='${gpus}' PYTHONPATH=${PYTHONPATH}:$repo_dir:$repo_dir/experiments/video/ $python_exec $repo_dir/experiments/video/run.py
    --n-iters-per-train-batch=200
    --plot-train-batch-freq=1
    --dataset-name='bear_256'
    --no-filtering
    --pairwise-smoothing
    --n-iters-final-frame=200
    --n-train-eval-breaks=82
    --factors-pixel-obs-N-rob=0.14
    --factors-pixel-obs-sigma=0.2
    --factors-recon-sigma=1.3
    --factors-recon-N-rob=0.35
    --plot-every-batch
    --plot-coeffs
    --plot-weights
    --plot-denoise-recon
    --plot-train-batch-freq=1
    --save-denoised-video
    --momentum=0.7
    --architecture='convnet_single_layer'
    --logdir=${logdir}
    --corruption-seed=${seed1}"

#    if [ "$rep" -eq "1" ]; then
#      # On the first run make plots of diagnostics
#      cmd="${cmd} --plot-weights --plot-coeffs --plot-convergence"
#    fi

  printf "PID of this script: %s\n" "$$"
  printf "Repeat %s of %s" "$rep" "$N_REPEATS"
  printf "\nRunning %s \n\n" "${cmd//[$'\t\r\n']}"
  eval "${cmd//[$'\t\r\n']}"
done