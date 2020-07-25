#!/bin/bash

#SBATCH --array=0-26
#SBATCH --job-name=cord19_scholar_baseline_sweep
#SBATCH --output=/workspace/sweep_outs_and_errors/cord19_baseline_sweep_output_%A_%a.txt
#SBATCH --error=/workspace/sweep_outs_and_errors/cord19_baseline_sweep_error_%A_%a.txt
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

################################################################
# Hyperparameter sweep for CORD-19 dataset 
#
#
#   Note that the SBATCH --array parameter needs to be set
#   to 0-{N-1} where N is the number of parameter combinations.
#
#   Also make sure that the dataset prefix is used consistently,
#   e.g. cord19 for this dataset, 20ng for 20 Newsgroups, etc.
#
# Once this .sh script is ready, run sbatch <submit-script>.sh
#
################################################################

################################################################
# Record of preprocessing, where $D is the data directory
################################################################
#  > python preprocess_data.py $D/train.jsonlist $D/processed --test $D/test.jsonlist --label dummy_label --tokenized --min-doc-count 10 --max-doc-freq .9
# Using snowball stopwords
# Reading data files
# Found 40000 training documents
# Found 10000 test documents
# Parsing documents
# Using tokenized_text element as the source of text to process
# Train set processing complete
# Test set processing complete
# Size of full vocabulary=280773
# Found label dummy_label with 1 classes
# Selecting the vocabulary
# Excluding words with frequency > 0.90: []
# Vocab size after filtering = 21254
# Final vocab size = 21254
# Most common words remaining: coronavirus disease pandemic patients sars infection respiratory results severe health
# Converting to count representations
# Size of train document-term matrix: (40000, 21254)
# Converting to count representations
# Size of test document-term matrix: (10000, 21254)
# 0 words missing from training data
# 492 words missing from test data
# Done!


topics_values=( 50 100 150 )
lr_values=( 0.001 0.002 0.005 )
alpha_values=( 0.005 0.01 0.1  )

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

declare -a arr=( "topics-50_lr-0.001_alpha-0.005" "topics-50_lr-0.001_alpha-0.01" "topics-50_lr-0.001_alpha-0.1" "topics-100_lr-0.001_alpha-0.005" "topics-100_lr-0.001_alpha-0.01" "topics-100_lr-0.001_alpha-0.1" "topics-150_lr-0.001_alpha-0.005" "topics-150_lr-0.001_alpha-0.01" "topics-150_lr-0.001_alpha-0.1" "topics-50_lr-0.002_alpha-0.005" "topics-50_lr-0.002_alpha-0.01" "topics-50_lr-0.002_alpha-0.1" "topics-100_lr-0.002_alpha-0.005" "topics-100_lr-0.002_alpha-0.01" "topics-100_lr-0.002_alpha-0.1" "topics-150_lr-0.002_alpha-0.005" "topics-150_lr-0.002_alpha-0.01" "topics-150_lr-0.002_alpha-0.1" "topics-50_lr-0.005_alpha-0.005" "topics-50_lr-0.005_alpha-0.01" "topics-50_lr-0.005_alpha-0.1" "topics-100_lr-0.005_alpha-0.005" "topics-100_lr-0.005_alpha-0.01" "topics-100_lr-0.005_alpha-0.1" "topics-150_lr-0.005_alpha-0.005" "topics-150_lr-0.005_alpha-0.01" "topics-150_lr-0.005_alpha-0.1" )

VAR="topics-${topics}_lr-${lr}_alpha-${alpha}"


# if [[ ${topics} == 10 ]]; then
# 	min_npmi=0.33
# else
# 	min_npmi=0.26
# fi
min_npmi=0.0

# First argument should be the preprocessor output from Scholar
array_contains arr $VAR && echo "For cord19 baseline, skipping ${VAR}" || /workspace/.conda/envs/scholar/bin/python multiple_run_scholar.py /workspace/kd-topic-modeling/data/cord19/processed --dev-metric npmi -k ${topics} --epochs 500 --patience 500 --batch-size 200 --background-embeddings --device 0 --global-seed 13 --store-all --dev-prefix test --runs 5 -l ${lr} --alpha ${alpha} --min-acceptable-npmi ${min_npmi} -o ../results/sweep/cord19_scholar_baseline/output_topics-${topics}_lr-${lr}_alpha-${alpha}/

