#!/usr/bin/env python
# coding: utf-8
#################################################################
# Compare topics in two models.
#
# Uses Jensen-Shannon divergence and competitive (greedy) linking
# to align topics in the two models.
#
# Note: requires that the two models have the same vocabulary.
#################################################################

# Adding (hard-wired) path for imports
import sys
sys.path.append('/workspace/kd-topic-modeling/kd-scholar')


import os
import pandas as pd
import numpy as np
import json
from run_scholar import print_top_words
import scipy
import torch
import pickle
import file_handling as fh


# Hard-coding for now
# Number of topics in first model must be <= number in second model
#   model_path1 = "/workspace/kd-topic-modeling/results/cord19_40K_1aug2020"
#   model_path2 = "/workspace/kd-topic-modeling/results/cord19_40K_1aug2020-TEST"
model_path1 = "/workspace/kd-topic-modeling/results/sweep/rww_scholar_baseline_phrases/output_topics-10_lr-0.002_alpha-0.01/121958"
model_path2 = "/workspace/kd-topic-modeling/results/sweep/rww_scholar_baseline_phrases/output_topics-10_lr-0.002_alpha-0.01/131932"
n_words     = 10


## Function for computing JS divergence between two vectors, in our case, V-dimensional (two topic vectors).
## Note that lower JS divergence score means more similar distributions.
def jsd(p, q, base=np.e):
    '''
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    
    ## normalize p, q to probabilities
    p, q = np.array(torch.softmax(torch.from_numpy(p), dim=0)), np.array(torch.softmax(torch.from_numpy(q), dim=0))
    m = (p + q)/2
    return scipy.stats.entropy(p, m, base=base)/2. +  scipy.stats.entropy(q, m, base=base)/2.

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
    #assert beta1.shape==beta2.shape
    js_div_scores = js_divergence(beta1, beta2)
    x, y = js_div_scores.shape
    #print(js_div_scores.shape)
    topic_match_tuples = []
    topic_match_scores = []
    while len(topic_match_tuples)<min(x, y):
        z = np.argmin(js_div_scores)
        i = z//js_div_scores.shape[0]
        j = z%js_div_scores.shape[1]
        topic_match_tuples.append((i,j))
        topic_match_scores.append(np.min(js_div_scores))
        js_div_scores[i, :] = 2.0
        js_div_scores[:, j] = 2.0
    return topic_match_tuples, topic_match_scores

# Show top n words for topic
def get_top_n_topic_words(beta, vocab, n=30):
    K, V = beta.shape
    out = []
    for i in range(K):
        topic = []
        vocab_dist = beta[i]
        top_word_indices = vocab_dist.argsort()[-n:][::-1]
        for ind in top_word_indices:
            topic.append(vocab[ind])
        out.append(topic)
    return out


def main():
    print("Reading model 1")
    beta1   = np.load(os.path.join(model_path1, 'beta.npz'))['beta']
    vocab1  = fh.read_json(os.path.join(model_path1, 'vocab.json'))
    topics1 = get_top_n_topic_words(beta1, vocab1, n_words)

    print("Reading model 2")
    beta2 = np.load(os.path.join(model_path2, 'beta.npz'))['beta']
    vocab2  = fh.read_json(os.path.join(model_path2, 'vocab.json'))
    topics2 = get_top_n_topic_words(beta2, vocab2, n_words)

    
    print("Matching topics")
    topic_match_tuples, topic_match_scores = get_topic_matched_pairs(beta1, beta2)


    for pair, score in zip(topic_match_tuples, topic_match_scores):
        print(str(score) + "\t" + str(pair))
        topicnum1    = pair[0]
        topicnum2    = pair[1]
        topicstring1 = ' '.join(topics1[topicnum1])
        topicstring2 = ' '.join(topics2[topicnum2])
        print(topicstring1)
        print(topicstring2)


    
if __name__ == "__main__":
    main()

