import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
# Include Scholar as directory for imports
# Use . if running in scholar code directory
sys.path.append('.')
from run_scholar import print_top_words



# Hardwired info. TO DO: turn into commandline arguments
indir           = "./example/inputs"
scholar_outdir  = "./example/vanilla_scholar_10topics"
csv_outfile     = "./example/vanilla_10topics_out.csv"
K_to_show       = 20 

# read file with document json
print("Reading document file.\n")
with open(os.path.join(indir, 'rwwd-train.jsonlist')) as f:
    lines = f.readlines()

# load the vocab
with open(os.path.join(scholar_outdir,'vocab.json')) as f:
    vocab = json.load(f)
        
# load the stored (K x V) topic matrix (stored in a compressed numpy format)
print("Topics")
beta = np.load(os.path.join(scholar_outdir, 'beta.npz'))['beta']
print_top_words(beta, vocab, n_pos=K_to_show, n_neg=0);

# load the matrix with topic proportions for each document (note that this excludes those in the dev set).
npz   = np.load(os.path.join(scholar_outdir, 'theta.train.npz')) 
ids   = npz['ids']
theta = npz['theta']
n_docs, n_topics = theta.shape

# Construct CSV output
# for i in range(n_docs):
print("\nWriting CSV to " + csv_outfile)
with open(csv_outfile, 'w', encoding="utf-8", errors="replace") as csvfile:
    
    writer = csv.writer(csvfile, dialect='excel', delimiter=',', quoting=csv.QUOTE_ALL)

    # Create header
    row = ['docID']
    for i in range(n_topics):
        row.append("Topic " + str(i))
    row.append('text')
    writer.writerow(row)
    
    for i in range(n_docs):
        doc           =  json.loads(lines[i])
        text         = doc['text'].replace("\n", " ")
        row         = [doc['id']] + list(theta[i]) + [text]
        writer.writerow(row)



############################################################################
# Useful examples
# Also see https://github.com/dallascard/scholar/blob/master/tutorial.ipynb
############################################################################

# Example of extracting info from a document
if False:
    print("First document...")
    first_doc = json.loads(lines[0])
    for key, value in first_doc.items():
        print(key, ':', value)

# Example: showing how to display the most common words
if False:
    # load the background log-frequencies
    bg = np.load(os.path.join(scholar_outdir,'bg.npz'))['bg']
    # sort terms by log-frequency
    order = np.argsort(bg)
    # print the most common words
    for i in range(1, 16):
        index = order[-i]
        print(vocab[index], np.exp(bg[index]))

# Example: find the original line corresponding to a single doc, and display the text
if False:
  index = 1
  print(ids[index])
  for line in lines:
      doc = json.loads(line)
      if doc['id'] == ids[index]:
          print(doc['text'])
          break
    







