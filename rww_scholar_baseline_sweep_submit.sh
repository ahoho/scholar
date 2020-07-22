#!/bin/bash

#SBATCH --array=0-3
#SBATCH --job-name=rww_scholar_baseline_sweep
#SBATCH --output=/workspace/sweep_outs_and_errors/rww_baseline_sweep_output_%A_%a.txt
#SBATCH --error=/workspace/sweep_outs_and_errors/rww_baseline_sweep_error_%A_%a.txt
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

################################################################
# Hyperparameter sweep for Real World Worry (rww) dataset 
#
#   PSR doing this mainly to get up to speed.
#
#   Note that the SBATCH --array parameter needs to be set
#   to 0-{N-1} where N is the number of parameter combinations.
#
#   Also make sure that the dataset prefix is used consistently,
#   e.g. rww for this dataset, 20ng for 20 Newsgroups, etc.
#
# Once this .sh script is ready, run sbatch <submit-script>.sh
#
################################################################
topics_values=( 10 15 )
lr_values=( 0.002 )
alpha_values=( 0.01 0.1  )

trial=${SLURM_ARRAY_TASK_ID}
topics=${topics_values[$(( trial % ${#topics_values[@]} ))]}
trial=$(( trial / ${#topics_values[@]} ))
lr=${lr_values[$(( trial % ${#lr_values[@]} ))]}
trial=$(( trial / ${#lr_values[@]} ))
alpha=${alpha_values[$(( trial % ${#alpha_values[@]} ))]}

##source ../../prep.sh
##export OMP_NUM_THREADS=4
##export OMP_PROC_BIND=spread

## use ${topics}, ${lr}, ${alpha} below
## 7/20/2020: reversed 0/1 from previous version
array_contains () {
	local array="$1[@]"
	local seeking=$2
	#	local in=1
	local in=0
	for element in "${!array}"; do
		if [[ $element == "$seeking" ]]; then
		    # in=0
		    in=1
		    break
		fi
	done
	return $in
}
declare -a arr=("topics-10_lr-0.002_alpha-0.01" "topics-10_lr-0.002_alpha-0.1" "topics-15_lr-0.002_alpha-0.01" "topics-15_lr-0.002_alpha-0.1" )
VAR="topics-${topics}_lr-${lr}_alpha-${alpha}"


# if [[ ${topics} == 10 ]]; then
# 	min_npmi=0.33
# else
# 	min_npmi=0.26
# fi
min_npmi=0.0

# First argument should be the preprocessor output from Scholar
# Using --dev-prefix test (i.e. using test set as dev data)
array_contains arr $VAR && echo "For rww baseline, skipping ${VAR}" || /workspace/.conda/envs/scholar/bin/python multiple_run_scholar.py ../data/real_world_worry/processed --dev-metric npmi -k ${topics} --epochs 500 --patience 500 --batch-size 200 --background-embeddings --device 0 --global-seed 42 --store-all --dev-prefix test --runs 5 -l ${lr} --alpha ${alpha} --min-acceptable-npmi ${min_npmi} -o ../results/sweep/rww_scholar_baseline/output_topics-${topics}_lr-${lr}_alpha-${alpha}/
