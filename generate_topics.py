################################################################
# Generate top-N words for topics, one per line, to stdout
################################################################
import os
import sys
import argparse
import numpy as np
import file_handling as fh


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


def main(call=None):
    
    # handle command line
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path for model directory")
    parser.add_argument(
        "-n",
        dest="n_words",
        type=int,
        default=30,
        help="number of words to show in each topic"
    )
    options    = parser.parse_args(call)
    model_path = options.model_path
    n_words    = options.n_words

    ## load Beta
    beta = np.load(os.path.join(model_path, 'beta.npz'))['beta']

    ## load vocab 
    vocab = fh.read_json(os.path.join(model_path, 'vocab.json'))

    # get and print topics
    topics = get_top_n_topic_words(beta, vocab, n_words)
    for topic in topics:
        topicstring = ' '.join(topic)
        print(topicstring)

    
if __name__ == "__main__":
    main()

