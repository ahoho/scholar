import os
import pandas as pd
import numpy as np
import json
import scipy
from scipy import stats
import argparse
########################################################################################
# Align topics from two topic models using their topic-word distributions.
#
# Applies competitive linking (Melamed, 2000) using Jensen-Shannon divergence as in
# Hoyle, Alexander, Pranav Goel, and Philip Resnik, "Improving Neural Topic Models using
# Knowledge Distillation", EMNLP 2020.
#
# python topic_alignment.py
#    <path to topic_word_file_from_model1.csv>
#    <path to topic_word_file_from_model2.csv>
#    -o <path to output file>
#    -n <num_topwords>
#
# Input CSV input files should contain the following header:
#  Word, label_for_topic0, ..., label_for_topicN
# and each cell i,j should contain Pr(word_i|topic_j)
#
# For each matched topic pair, output CSV shows topic index (from 0), topic label, and
# top N words.  Unmatched topics appear paired with empty cells on the other side.
########################################################################################

def main(call=None):

    # Handle command line
    parser = argparse.ArgumentParser()
    parser.add_argument("topic_word_file_from_model1")
    parser.add_argument("topic_word_file_from_model2")
    parser.add_argument(
        "-o",
        dest="output_path",
        type=str,
        default="topic_aligned_pairs.csv",
        help="Output file path (.csv)",
    )
    parser.add_argument(
        "-n",
        dest="N",
        type=int,
        default=10,
        help="Number of top words to show per topic",
    )
    options = parser.parse_args(call)
    path_to_file1 = options.topic_word_file_from_model1
    path_to_file2 = options.topic_word_file_from_model2
    output_path   = options.output_path
    N             = options.N

    # Read topic-word matrices from the two models
    topic_word1 = pd.read_csv(path_to_file1)
    topic_word2 = pd.read_csv(path_to_file2)

    # Matching is done only on the basis of vocabulary in common in the two models
    # (So mileage may vary if the two models have really different vocabularies)
    common_vocab = get_common_vocab(topic_word1, topic_word2)
    topic_word_final1, topic_word_final2, topic_ind_to_topic_label1, topic_ind_to_topic_label2, label_to_ind1, label_to_ind2 \
      = get_same_vocab_topic_word_dists_and_topic_index_label_dict(topic_word1, topic_word2, common_vocab)    
    
    #renormalize the above matrices obtained and return these topic-word matrices with the same vocab for both models
    #renormalization is needed since js divergence works on two probability distributions
    topic_word_final1 = topic_word_final1/topic_word_final1.sum(axis=1).reshape((len(topic_ind_to_topic_label1), 1))
    topic_word_final2 = topic_word_final2/topic_word_final2.sum(axis=1).reshape((len(topic_ind_to_topic_label2), 1))
    
    topic_match_tuples, topic_match_scores = get_topic_matched_pairs(topic_word_final1, topic_word_final2)
    
    topic_match_results_df = create_output_csv_for_matched_topic_pairs(topic_match_tuples,
                                                                         topic_match_scores,
                                                                         topic_ind_to_topic_label1,
                                                                         topic_ind_to_topic_label2,
                                                                         label_to_ind1,
                                                                         label_to_ind2,
                                                                         topic_word1,
                                                                         topic_word2,
                                                                         N)

    # Add columns with top words from (original, pre-vocabulary-intersection) topics 
    final_topic_match_results_df = add_topwords_to_aligned_pairs(topic_match_results_df, topic_word1, topic_word2,
                                                                     topic_ind_to_topic_label1, topic_ind_to_topic_label2,
                                                                     label_to_ind1, label_to_ind2)

    #print(final_topic_match_results_df)
    final_topic_match_results_df.to_csv(output_path, index=False)

def add_topwords_to_aligned_pairs(topic_match_results_df, topic_word1, topic_word2, ind_to_label1, ind_to_label2, label_to_ind1, label_to_ind2, numwords=20):
    
    return topic_match_results_df
    
    
def get_common_vocab(tw1, tw2, word_col_name = 'Word'):
    #input: the two dataframes from the two models (tw1, tw2) with a column for 'Word'
    #output: list of words that are present in vocabulary of both the models
    
    words1 = set(list(tw1[word_col_name]))
    words2 = set(list(tw2[word_col_name]))
    return list(words1.intersection(words2))

#using the list of common words, get the values for those words for every topic for both the models
def get_same_vocab_topic_word_dists_and_topic_index_label_dict(tw1, tw2, common_words, word_col_name = 'Word'):
    #Inputs: the two dataframes (topic-word) from the two models (tw1, tw2), list of common vocab words
    #Output 1: renormalized topic-word distributions for both models sharing same vocab
    #Output 2: dictionaries for each model - index in the topic-word distribution array mapped to original topic label
    
    #assuming each dataframe has first column for word and remaining columns are topic labels
    topic_labels1 = list(tw1.columns[1:])
    topic_labels2 = list(tw2.columns[1:])

    # dicts mapping topic index to topic label in each model, and vice-versa
    topic_ind_to_topic_label1 = {}
    topic_ind_to_topic_label2 = {}
    label_to_ind1 = {}
    label_to_ind2 = {}
    
    word_to_index1 = {}
    word_to_index2 = {}
    
    words1 = list(tw1[word_col_name])
    words2 = list(tw2[word_col_name])
    
    for i, w in enumerate(words1):
        word_to_index1[w] = i
    for i, w in enumerate(words2):
        word_to_index2[w] = i    
    
    indices_to_keep1 = []
    indices_to_keep2 = []
    for word in common_words:
        indices_to_keep1.append(word_to_index1[word])
        indices_to_keep2.append(word_to_index2[word])
    
    topic_word_final1 = []
    topic_word_final2 = []
    
    for i, topic_label in enumerate(topic_labels1):
        topic_ind_to_topic_label1[i] = topic_label
        label_to_ind1[topic_label]   = i
        topic_word_vals = np.array(tw1[topic_label]) 
        topic_word_vals = topic_word_vals[indices_to_keep1] #select indices of words in common vocab
        topic_word_final1.append(topic_word_vals)
    topic_word_final1 = np.array(topic_word_final1)
    
    for i, topic_label in enumerate(topic_labels2):
        topic_ind_to_topic_label2[i] = topic_label
        label_to_ind2[topic_label]   = i
        topic_word_vals = np.array(tw2[topic_label])
        topic_word_vals = topic_word_vals[indices_to_keep2] #select indices of words in common vocab
        topic_word_final2.append(topic_word_vals)
    topic_word_final2 = np.array(topic_word_final2)
    
    return topic_word_final1, topic_word_final2, topic_ind_to_topic_label1, topic_ind_to_topic_label2, label_to_ind1, label_to_ind2


## Function for computing JS divergence between two vectors, in our case, V-dimensional (two topic vectors).
## Note that lower JS divergence score means more similar distributions.
def jsd(p, q, base=np.e):
    '''
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
   
    ## normalize p, q to probabilities - not needed since input is already prob dists
    #p, q = np.array(torch.softmax(torch.from_numpy(p), dim=0)), np.array(torch.softmax(torch.from_numpy(q), dim=0))
    m = (p + q)/2
    return stats.entropy(p, m, base=base)/2. +  scipy.stats.entropy(q, m, base=base)/2.


## JS Divergence score matrix, originally written for two betas of the same shape but below should allow for
## two difference number of topics. Let beta1 be the matrix for the model with LOWER number of topics. For a beta1 with K1 topics
## and beta2 with K2 topics, the resulting matrix will be K1 x K2 - each cell (i,j) carrying JS divergence score between
## topic i of model 1 (beta1) and topic j of model 2 (beta2).
def js_divergence(beta1, beta2):
    #assert beta1.shape==beta2.shape
    x, _ = beta1.shape
    y, _ = beta2.shape
    js_div_score_matrix = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            js_div_score_matrix[i][j] = round(jsd(beta1[i], beta2[j]), 4)
    return js_div_score_matrix


## Use the JS Divergence score matrix to get matched topic pairs. Simple algorithm - get the pair with minimum divergence
## score, add it to the matched pairs (hence final list is sorted by match value by default), remove those topics from
## consideration (replacing row and column corresponding to topic 1 from beta1, topic2 from beta2
## by a larger than max possible value), and repeat the process
def get_topic_matched_pairs(beta1, beta2):

    js_div_scores = js_divergence(beta1, beta2)
    x, y = js_div_scores.shape
    N = min(x, y)

    topic_match_tuples = [] #where every tuple has form (index1, index2) where index1 is topic num from model 1, and index2 is topic num from model 2
    topic_match_scores = []
    while len(topic_match_tuples) < N:
        z = np.argmin(js_div_scores)
        i = z//js_div_scores.shape[1]
        j = z%js_div_scores.shape[1]
        topic_match_tuples.append((i,j))
        topic_match_scores.append(np.min(js_div_scores))
        js_div_scores[i, :] = 2.0
        js_div_scores[:, j] = 2.0
    return topic_match_tuples, topic_match_scores

def get_topic_top_words(df, label, N=10):
    # In df first column is Word, the rest are the topic labels
    # Returns top N words for this topic label
    sorted_df = df.sort_values(by = label, ascending = False)
    words    = sorted_df['Word'].tolist()
    topwords = words[:N]
    return topwords

def create_output_csv_for_matched_topic_pairs(topic_match_tuples, topic_match_scores,
                                                  topic_ind_to_topic_label1, topic_ind_to_topic_label2,
                                                  label_to_ind1, label_to_ind2,
                                                  topic_word1, topic_word2, N):
    # Creates output spreadsheet with matched topics
    # For each matched topic pair show topic index (from 0), topic label, and topic top N words
    # Add empty cells for unmatched topics in the larger model
    
    out = pd.DataFrame(columns = ['divergence_score',
                                  'topic_ind1', 'topic_from_model1', 'topic_words_model1', 
                                  'topic_ind2', 'topic_from_model2', 'topic_words_model2'])
    all_topic_labels1 = list(topic_ind_to_topic_label1.values())
    all_topic_labels2 = list(topic_ind_to_topic_label2.values())
    
    topic_labels_model1, topic_labels_model2, divergence_scores = [], [], []
    topic_inds_model1, topic_inds_model2   = [], []
    topic_words_model1, topic_words_model2 = [], []

    # Create rows for matched topic pairs
    for topic_pair, pair_score in zip(topic_match_tuples, topic_match_scores):
        topic_inds_model1.append(topic_pair[0])
        topic_inds_model2.append(topic_pair[1])
        topic_labels_model1.append(topic_ind_to_topic_label1[topic_pair[0]])
        topic_labels_model2.append(topic_ind_to_topic_label2[topic_pair[1]])
        topic_words_model1.append(" ".join(get_topic_top_words(topic_word1, topic_ind_to_topic_label1[topic_pair[0]], N)))
        topic_words_model2.append(" ".join(get_topic_top_words(topic_word2, topic_ind_to_topic_label2[topic_pair[1]], N)))
        divergence_scores.append(pair_score)

    # Create rows if one or the other topic is unmatched, padding columns as needed
    # Note use of np.nan for empty cell
    if len(divergence_scores)<len(topic_ind_to_topic_label1):
        for label in all_topic_labels1:
            if label not in topic_labels_model1:
                topic_labels_model1.append(label)
                topic_inds_model1.append(label_to_ind1[label])
                topic_words_model1.append(" ".join(get_topic_top_words(topic_word1, label, N)))
                topic_labels_model2.append(np.nan)
                divergence_scores.append(np.nan)
    elif len(divergence_scores)<len(topic_ind_to_topic_label2):
        for label in all_topic_labels2:
            if label not in topic_labels_model2:
                topic_labels_model2.append(label)
                topic_inds_model2.append(label_to_ind2[label])
                topic_words_model2.append(" ".join(get_topic_top_words(topic_word2, label, N)))
                topic_labels_model1.append(np.nan)
                divergence_scores.append(np.nan)
    if (len(topic_inds_model1) < len(topic_inds_model2)):
        # https://stackoverflow.com/questions/3438756/some-built-in-to-pad-a-list-in-python
        topic_inds_model1  += [np.nan]  * (len(topic_inds_model2) - len(topic_inds_model1))
        topic_words_model1 += [np.nan]  * (len(topic_words_model2) - len(topic_words_model1))
    else:
        topic_inds_model2  += [np.nan] * (len(topic_inds_model1) - len(topic_inds_model2))
        topic_words_model2 += [np.nan] * (len(topic_words_model1) - len(topic_words_model2))
        
    out['topic_from_model1']  = topic_labels_model1
    out['topic_from_model2']  = topic_labels_model2
    out['topic_ind1']         = topic_inds_model1
    out['topic_ind2']         = topic_inds_model2
    out['topic_words_model1'] = topic_words_model1
    out['topic_words_model2'] = topic_words_model2
    out['divergence_score']   = divergence_scores
    return out

if __name__ == "__main__":
    main()
