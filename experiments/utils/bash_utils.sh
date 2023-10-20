#!/bin/bash


function get_repo_dir(){
  candidates=("/home/$(whoami)/code/gbp_learning_anon/"
              "/Users/$(whoami)/Desktop/code/gbp_learning_anon/")
  for cand in "${candidates[@]}";
  do
    if [[ -d "$cand" ]]; then
        echo "$cand"
        break
    fi
  done
}


function get_results_dir(){
  repo_dir=$1
  exp_dir=$2
  exp_subdir=$3
  candidates=("/home/$(whoami)/code/gbp_learning_anon/experiments/${exp_dir}/results/${exp_subdir}/"
              "$repo_dir/experiments/${exp_dir}/results/${exp_subdir}")
  for cand in "${candidates[@]}";
  do
    if [[ -d "$cand" ]]; then
        echo "$cand"
        break
    fi
  done
}


function get_gpu_ids {
  if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    # No $CUDA_VISIBLE_DEVICES
    busy_gpus_str="$( nvidia-smi pmon -s -u -c 1 | awk '$8 ~ /python/ { print $1} ' | xargs | sed -e 's/\s\+/,/g')"
    IFS=',' read -r -a busy_gpus_arr <<< "$busy_gpus_str"  # Convert comma-delim string to arr
    n_gpus=$(nvidia-smi --list-gpus | wc -l)
    available_gpus=()
    for gpuid in $(seq 0 "$(expr $n_gpus - 1)"); do
#      echo $gpuid
      if [[ "${busy_gpus_arr[*]}" =~ $gpuid ]]; then
        l=2  # do nothing
      else
#        echo "ADDING $gpuid"
        available_gpus+=("$gpuid")
      fi
    done
    n_gpus_needed=$1
    selected_available_gpus="$(echo ${available_gpus[*]:0:$n_gpus_needed} | sed -e 's/\s\+/,/g')"
    echo "$selected_available_gpus"
  else
    # If CUDA_VISIBLE_DEVICES already set, use these ids
    # Needed in queue system
    echo "$CUDA_VISIBLE_DEVICES"
  fi
}
