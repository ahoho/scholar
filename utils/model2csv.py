################################################################
#
#  Converts topic model output to CSV suitable as input for
#  utilities for topic visualization and model comparison
#
#  Example:
#    python model2csv.py
#    --package   scholar
#    --docfile   /Users/resnik/Misc/projects/rapid2020_nsf/modeling/real_world_worry/inputs/rwwd-train.jsonlist
#    --modeldir  /Users/resnik/Misc/projects/rapid2020_nsf/modeling/real_world_worry/vanilla_scholar_10topics/
#    --vocabfile /Users/resnik/Misc/projects/rapid2020_nsf/modeling/real_world_worry/vanilla_scholar_10topics/vocab.json
#
################################################################
import argparse
import pandas as pd
import numpy as np
import json
import os
import sys
import csv

def convert_scholar(modeldir, docfile, vocabfile, word_topics_file):
    
    # Read in original documents and the vocabulary file
    docs_df = pd.read_json(docfile, lines=True)
    with open(vocabfile) as f:
        vocab_list = json.load(f)

    # Load beta (topic-word distribution) matrix, transpose to get word-topic matrix,
    # convert to a dataframe with 'Word' as the first column,
    # output to CSV file.
    beta          = np.load(os.path.join(modeldir, 'beta.npz'))['beta']
    betaT         = np.transpose(beta)
    V, K          = betaT.shape
    labels        = ["Topic {}".format(str(i+1)) for i in range(K)]
    betaT_df      = pd.DataFrame(betaT, columns = labels)
    betaT_df.insert(0, 'Word', vocab_list)
    betaT_df.to_csv(word_topics_file, index=False)
    sys.stderr.write("Wrote {}\n".format(word_topics_file))

    # Load theta (document-topic proportions) matrix (training docs only, not dev/test),
    # and document jsonlines.
    npz   = np.load(os.path.join(modeldir, 'theta.train.npz')) 
    ids   = npz['ids']
    theta = npz['theta']
    n_docs, n_topics = theta.shape
    with open(docfile) as f:
        doc_jsonl = f.readlines()
    
    # For CSV output header and one row per document
    with open(document_topics_file, 'w', encoding="utf-8", errors="replace") as csvfile:
        writer = csv.writer(csvfile, dialect='excel', delimiter=',', quoting=csv.QUOTE_ALL)
        row = ['docID'] + labels + ['text']
        writer.writerow(row)
        for i in range(n_docs):
            doc         =  json.loads(doc_jsonl[i])
            text        = doc['text'].replace("\n", " ")
            row         = [doc['id']] + list(theta[i]) + [text]
            writer.writerow(row)
    sys.stderr.write("Wrote {}\n".format(document_topics_file))

    
# Handle command line
parser = argparse.ArgumentParser(description='Converts topic model output into CSV files')
parser.add_argument('-p','--package',
                        help='Topic model package: scholar, mallet, scholar', dest='package',   default='scholar')
parser.add_argument('-m','--modeldir',
                        help='Directory containing model output',             dest='modeldir',  default='.')
parser.add_argument('-d','--docfile',
                        help='File containing documents that were modeled',   dest='docfile',   default=None)
parser.add_argument('-v','--vocabfile',
                        help='File containing vocabulary',                    dest='vocabfile', default=None)
parser.add_argument('-w','--word_topics_file',
                        help='Output CSV for topic-words distribution',       dest='word_topics_file', default="word_topics.csv")
parser.add_argument('-D','--document_topics_file',
                        help='Output CSV for doc-topics distribution',        dest='document_topics_file', default="document_topics.csv")


args = vars(parser.parse_args())
package              = args['package']
modeldir             = args['modeldir']
docfile              = args['docfile']
vocabfile            = args['vocabfile']
word_topics_file     = args['word_topics_file']
document_topics_file = args['document_topics_file']

if (package == 'scholar'):
    convert_scholar(modeldir, docfile, vocabfile, word_topics_file)
else:
    sys.stderr.write("Not yet handling package '{}'\n".format(package))
