######################################################################################
#
# python tokenize_with_phrases.py < in.jsonl > out.jsonl
#
#  Input: same jsonlines format required by preprocess_data.py
#
#  Output: adds a 'tokenized' element to each json line,
#          doing spaCy-based tokenization in which multi-word
#          phrases have been appended to the unigram tokens.
#
#  NOTE: Hardwires relative path to stopword list
#        This should be turned into a command line argument
#
#  Example:
#     {"id": "1",
#      "text": "This is a test of the emergency broadcast system",
#       ...}
#  Output is the original json line with this element added:
#     "tokenized_text": "test emergency broadcast system emergency_broadcast_system"
#
######################################################################################

import sys
import json
import spacy
from phrase_tokenization_subs import *

# Hard-wired stopword list
sys.stderr.write("Reading MALLET stopword list\n")
stoplist = load_wordlist("./stopwords/mallet_stopwords.txt")

# Initialize spacy
sys.stderr.write("Initializing spacy\n")
nlp = spacy.load("en_core_web_sm")

# Main loop
for line in sys.stdin:
    obj                   = json.loads(line)
    text                  = obj['text']
    tokens                = tokenize_string_adding_phrases(nlp, text, stoplist, 3)
    obj['tokenized_text'] = " ".join(tokens)
    new = json.dumps(obj)
    print(new)
    




