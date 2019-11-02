import os
import re
import sys
import string
import json
import multiprocessing
import itertools
from optparse import OptionParser
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import savemat
from gensim.utils import chunkize

import file_handling as fh

"""
Convert a dataset into the required format (as well as formats required by other tools).
Input format is one line per item.
Each line should be a json object.
At a minimum, each json object should have a "text" field, with the document text.
Any other field can be used as a label (specified with the --label option).
If training and test data are to be processed separately, the same input directory should be used
Run "python preprocess_data -h" for more options.
If an 'id' field is provided, this will be used as an identifier in the dataframes, otherwise index will be used 
"""

# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = "".join(punct_chars)
replace = re.compile("[%s]" % re.escape(punctuation))
alpha = re.compile("^[a-zA-Z_]+$")
alpha_or_num = re.compile("^[a-zA-Z_]+|[0-9_]+$")
alphanum = re.compile("^[a-zA-Z0-9_]+$")


def main(args):
    usage = "%prog train.jsonlist output_dir"
    parser = OptionParser(usage=usage)
    label_parser = parser.add_mutually_exclusive_group()
    label_parser.add_option(
        "--label",
        dest="label",
        default=None,   
        help="field(s) to use as label (comma-separated): default=%default",
    )
    label_parser.add_option(
        # c.f. https://stackoverflow.com/a/18609361/5712749
        '--label_dicts',
        dest="label_dicts",
        type=json.loads,
        default=None,
        help="""
        field(s) to use as label along with their values, format as json dict, e.g.,
        "{'label_1': ['class_1_a', 'class_1_b'], 'label_2': ['class_2_a', 'class_2_b']}"
        """
    )
    parser.add_option(
        "--test",
        dest="test",
        default=None,
        help="Test data (test.jsonlist): default=%default",
    )
    parser.add_option(
        "--train-prefix",
        dest="train_prefix",
        default="train",
        help="Output prefix for training data: default=%default",
    )
    parser.add_option(
        "--test-prefix",
        dest="test_prefix",
        default="test",
        help="Output prefix for test data: default=%default",
    )
    parser.add_option(
        "--stopwords",
        dest="stopwords",
        default="snowball",
        help="List of stopwords to exclude [None|mallet|snowball]: default=%default",
    )
    parser.add_option(
        "--min-doc-count",
        dest="min_doc_count",
        default=0,
        help="Exclude words that occur in less than this number of documents",
    )
    parser.add_option(
        "--max-doc-freq",
        dest="max_doc_freq",
        default=1.0,
        help="Exclude words that occur in more than this proportion of documents",
    )
    parser.add_option(
        "--keep-num",
        action="store_true",
        dest="keep_num",
        default=False,
        help="Keep tokens made of only numbers: default=%default",
    )
    parser.add_option(
        "--keep-alphanum",
        action="store_true",
        dest="keep_alphanum",
        default=False,
        help="Keep tokens made of a mixture of letters and numbers: default=%default",
    )
    parser.add_option(
        "--strip-html",
        action="store_true",
        dest="strip_html",
        default=False,
        help="Strip HTML tags: default=%default",
    )
    parser.add_option(
        "--no-lower",
        action="store_true",
        dest="no_lower",
        default=False,
        help="Do not lowercase text: default=%default",
    )
    parser.add_option(
        "--min-length",
        dest="min_length",
        default=3,
        help="Minimum token length: default=%default",
    )
    parser.add_option(
        "--ngram_min",
        dest="ngram_min",
        default=1,
        help="n-grams lower bound",
    )
    parser.add_option(
        "--ngram_max",
        dest="ngram_max",
        default=1,
        help="n-grams upper bound",
    )
    parser.add_option(
        "--vocab-size",
        dest="vocab_size",
        default=None,
        help="Size of the vocabulary (by most common, following above exclusions): default=%default",
    )    
    parser.add_option(
        "--seed",
        dest="seed",
        default=42,
        help="Random integer seed (only relevant for choosing test set): default=%default",
    )

    (options, args) = parser.parse_args(args)

    train_infile = args[0]
    output_dir = args[1]

    test_infile = options.test
    train_prefix = options.train_prefix
    test_prefix = options.test_prefix
    label_fields = options.label or options.label_dicts
    min_doc_count = int(options.min_doc_count)
    ngram_range = int(options.ngram_min), int(options.ngram_max)
    min_doc_count = int(options.min_doc_count)    
    max_doc_freq = float(options.max_doc_freq)
    vocab_size = options.vocab_size
    stopwords = options.stopwords
    if stopwords == "None":
        stopwords = None
    keep_num = options.keep_num
    keep_alphanum = options.keep_alphanum
    strip_html = options.strip_html
    lower = not options.no_lower
    min_length = int(options.min_length)
    seed = options.seed
    if seed is not None:
        np.random.seed(int(seed))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocess_data(
        train_infile,
        test_infile,
        output_dir,
        train_prefix,
        test_prefix,
        min_doc_count,
        max_doc_freq,
        ngram_range,
        vocab_size,
        stopwords,
        keep_num,
        keep_alphanum,
        strip_html,
        lower,
        min_length,
        label_fields=label_fields,
    )


def preprocess_data(
    train_infile,
    test_infile,
    output_dir,
    train_prefix,
    test_prefix,
    min_doc_count=0,
    max_doc_freq=1.0,
    ngram_range=(1, 1),
    vocab_size=None,
    stopwords=None,
    keep_num=False,
    keep_alphanum=False,
    strip_html=False,
    lower=True,
    min_length=3,
    label_fields=None,
    workers=1,
    proc_multiplier=100,
):

    if stopwords == "mallet":
        print("Using Mallet stopwords")
        stopword_list = fh.read_text(os.path.join("stopwords", "mallet_stopwords.txt"))
    elif stopwords == "snowball":
        print("Using snowball stopwords")
        stopword_list = fh.read_text(
            os.path.join("stopwords", "snowball_stopwords.txt")
        )
    elif stopwords is not None:
        print("Using custom stopwords")
        stopword_list = fh.read_text(
            os.path.join("stopwords", stopwords + "_stopwords.txt")
        )
    else:
        stopword_list = []
    stopword_set = {s.strip() for s in stopword_list}

    print("Reading data files")
    train_items = fh.read_jsonlist(train_infile)
    n_train = len(train_items)
    print("Found {:d} training documents".format(n_train))

    if test_infile is not None:
        test_items = fh.read_jsonlist(test_infile)
        n_test = len(test_items)
        print("Found {:d} test documents".format(n_test))
    else:
        test_items = []
        n_test = 0

    all_items = itertools.chain(train_items, test_items)
    n_items = n_train + n_test

    if isinstance(label_fields, str):
        label_lists = {}
        if "," in label_fields:
            label_fields = label_fields.split(",")
        else:
            label_fields = [label_fields]
        for label_name in label_fields:
            label_set = set()
            for i, item in enumerate(all_items):
                if label_name is not None:
                    label_set.add(item[label_name])
            label_list = list(label_set)
            label_list.sort()
            n_labels = len(label_list)
            print("Found label %s with %d classes" % (label_name, n_labels))
            label_lists[label_name] = label_list
    if isinstance(label_fields, dict):
        label_lists = {k: sorted(list(set(v))) for k, v in label_fields.items()}
        label_fields = list(label_lists.keys())
    if label_fields is None:
            label_fields = []

    # to pass to pool.imap
    def _process_item(item):
        text = item['text']
        tokens, _ = tokenize(
            text,
            strip_html=strip_html,
            lower=lower,
            keep_numbers=keep_num,
            keep_alphanum=keep_alphanum,
            min_length=min_length,
            stopwords=stopword_set,
            ngram_range=ngram_range,
            vocab=vocab,
        )
        labels = None
        if label_fields:
            labels = {label_field: item[label_field] for label_field in label_fields}
            
        return item.get('id', None), tokens, labels
    
    # make vocabulary
    train_ids, train_parsed, train_labels = [], [], []
    test_ids, test_parsed, test_labels = [], [], []

    print("Parsing documents")
    word_counts = Counter()
    doc_counts = Counter()

    vocab = None

    # process in blocks
    pool = multiprocessing.Pool(workers)
    chunksize = proc_multiplier * workers
    for i, group in enumerate(i, chunkize(_process_item, chunksize=chunksize)):
        print(f'On group {i} of {len(all_items) // chunksize}')
        for ids, tokens, labels in pool.imap(_process_item, group):
            # store the parsed documents
            if i < n_train:
                if ids is not None:
                    train_ids.append(ids)
                if labels is not None:
                    train_labels.append(labels)
                train_parsed.append(tokens)
            else:
                if ids is not None:
                    test_ids.append(ids)
                if labels is not None:
                    test_labels.append(labels)
                test_parsed.append(tokens)

            # keep track fo the number of documents with each word
            word_counts.update(tokens)
            doc_counts.update(set(tokens))

    print("Size of full vocabulary=%d" % len(word_counts))

    print("Selecting the vocabulary")
    most_common = doc_counts.most_common()
    words, doc_counts = zip(*most_common)
    doc_freqs = np.array(doc_counts) / float(n_items)
    vocab = [
        word
        for i, word in enumerate(words)
        if doc_counts[i] >= min_doc_count and doc_freqs[i] <= max_doc_freq
    ]
    most_common = [word for i, word in enumerate(words) if doc_freqs[i] > max_doc_freq]
    if max_doc_freq < 1.0:
        print(
            "Excluding words with frequency > {:0.2f}:".format(max_doc_freq),
            most_common,
        )

    print("Vocab size after filtering = %d" % len(vocab))
    if vocab_size is not None:
        if len(vocab) > int(vocab_size):
            vocab = vocab[: int(vocab_size)]

    vocab_size = len(vocab)
    print("Final vocab size = %d" % vocab_size)

    print("Most common words remaining:", " ".join(vocab[:10]))
    vocab.sort()

    fh.write_to_json(vocab, os.path.join(output_dir, train_prefix + ".vocab.json"))

    train_X_sage, tr_aspect, tr_no_aspect, tr_widx, vocab_for_sage = process_subset(
        train_items,
        train_parsed,
        train_labels,
        label_fields,
        vocab,
        output_dir,
        train_prefix,
    )
    if n_test > 0:
        test_X_sage, te_aspect, te_no_aspect, _, _ = process_subset(
            test_items,
            test_parsed,
            test_labels,
            label_fields,
            vocab,
            output_dir,
            test_prefix,
        )

    train_sum = np.array(train_X_sage.sum(axis=0))
    print("%d words missing from training data" % np.sum(train_sum == 0))

    if n_test > 0:
        test_sum = np.array(test_X_sage.sum(axis=0))
        print("%d words missing from test data" % np.sum(test_sum == 0))

    sage_output = {
        "tr_data": train_X_sage,
        "tr_aspect": tr_aspect,
        "widx": tr_widx,
        "vocab": vocab_for_sage,
    }
    if n_test > 0:
        sage_output["te_data"] = test_X_sage
        sage_output["te_aspect"] = te_aspect
    savemat(os.path.join(output_dir, "sage_labeled.mat"), sage_output)
    sage_output["tr_aspect"] = tr_no_aspect
    if n_test > 0:
        sage_output["te_aspect"] = te_no_aspect
    savemat(os.path.join(output_dir, "sage_unlabeled.mat"), sage_output)

    print("Done!")


def process_subset(
    items, ids, parsed, labels, label_fields, vocab, output_dir, output_prefix
):
    n_items = len(items)
    vocab_size = len(vocab)
    vocab_index = dict(zip(vocab, range(vocab_size)))

    if not ids or len(ids) != n_items:
        ids = [str(i) for i in range(n_items)]

    # create a label index using string representations
    if labels:
        labels_df = pd.DataFrame.from_records(labels, index=ids)

        for label_field in label_fields:
            labels_df_subset = pd.get_dummies(labels_df[label_field])
            labels_df_subset.to_csv(
                os.path.join(output_dir, output_prefix + "." + label_field + ".csv")
            )
            if labels_df[label_field].nunique() == 2:
                labels_df_subset.iloc[0, :].to_csv(
                    os.path.join(
                        output_dir, output_prefix + "." + label_field + "_vector.csv"
                    )
                )

    X = np.zeros([n_items, vocab_size], dtype=int)

    dat_strings = []
    dat_labels = []
    mallet_strings = []
    fast_text_lines = []

    counter = Counter()
    word_counter = Counter()
    doc_lines = []
    print("Converting to count representations")
    for i, words in enumerate(parsed):
        # get the vocab indices of words that are in the vocabulary
        indices = [vocab_index[word] for word in words if word in vocab_index]
        word_subset = [word for word in words if word in vocab_index]

        counter.clear()
        counter.update(indices)
        word_counter.clear()
        word_counter.update(word_subset)

        if len(counter.keys()) > 0:
            # udpate the counts
            mallet_strings.append(str(i) + "\t" + "en" + "\t" + " ".join(word_subset))

            dat_string = str(int(len(counter))) + " "
            dat_string += " ".join(
                [
                    str(k) + ":" + str(int(v))
                    for k, v in zip(list(counter.keys()), list(counter.values()))
                ]
            )
            dat_strings.append(dat_string)

            # for dat formart, assume just one label is given
            if len(label_fields) > 0:
                label = items[i][label_fields[-1]]
                dat_labels.append(str(label_index[str(label)]))

            values = list(counter.values())
            X[
                np.ones(len(counter.keys()), dtype=int) * i, list(counter.keys())
            ] += values

    # convert to a sparse representation
    sparse_X = sparse.csr_matrix(X)
    fh.save_sparse(sparse_X, os.path.join(output_dir, output_prefix + ".npz"))

    print("Size of {:s} document-term matrix:".format(output_prefix), sparse_X.shape)

    fh.write_to_json(ids, os.path.join(output_dir, output_prefix + ".ids.json"))

    # save output for Mallet
    fh.write_list_to_text(
        mallet_strings, os.path.join(output_dir, output_prefix + ".mallet.txt")
    )

    # save output for David Blei's LDA/SLDA code
    fh.write_list_to_text(
        dat_strings, os.path.join(output_dir, output_prefix + ".data.dat")
    )
    if len(dat_labels) > 0:
        fh.write_list_to_text(
            dat_labels,
            os.path.join(output_dir, output_prefix + "." + label_field + ".dat"),
        )

    # save output for Jacob Eisenstein's SAGE code:
    sparse_X_sage = sparse.csr_matrix(X, dtype=float)
    vocab_for_sage = np.zeros((vocab_size,), dtype=np.object)
    vocab_for_sage[:] = vocab

    # for SAGE, assume only a single label has been given
    if len(label_fields) > 0:
        # convert array to vector of labels for SAGE
        sage_aspect = np.argmax(np.array(labels_df.values, dtype=float), axis=1) + 1
    else:
        sage_aspect = np.ones([n_items, 1], dtype=float)
    sage_no_aspect = np.array([n_items, 1], dtype=float)
    widx = np.arange(vocab_size, dtype=float) + 1

    return sparse_X_sage, sage_aspect, sage_no_aspect, widx, vocab_for_sage


def tokenize(
    text,
    strip_html=False,
    lower=True,
    keep_emails=False,
    keep_at_mentions=False,
    keep_numbers=False,
    keep_alphanum=False,
    min_length=3,
    stopwords=None,
    ngram_range=(1, 1),
    vocab=None,
):
    text = clean_text(text, strip_html, lower, keep_emails, keep_at_mentions)
    tokens = text.split()

    if stopwords is not None:
        tokens = ["_" if t in stopwords else t for t in tokens]

    # remove tokens that contain numbers
    if not keep_alphanum and not keep_numbers:
        tokens = [t if alpha.match(t) else "_" for t in tokens]

    # or just remove tokens that contain a combination of letters and numbers
    elif not keep_alphanum:
        tokens = [t if alpha_or_num.match(t) else "_" for t in tokens]

    # drop short tokens
    if min_length > 0:
        tokens = [t if len(t) >= min_length else "_" for t in tokens]

    counts = Counter()

    unigrams = [t for t in tokens if t != "_"]
    counts.update(unigrams)

    if vocab is not None:
        tokens = [token for token in unigrams if token in vocab]
    else:
        tokens = unigrams
    
    tokens = [
        '_'.join(tokens[j:j+i])
        for i in range(min(ngram_range), max(ngram_range) + 1)
        for j in range(len(tokens) - i + 1)
    ]
    
    return tokens, counts


def clean_text(
    text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False
):
    # remove html tags
    if strip_html:
        text = re.sub(r"<[^>]+>", "", text)
    else:
        # replace angle brackets
        text = re.sub(r"<", "(", text)
        text = re.sub(r">", ")", text)
    # lower case
    if lower:
        text = text.lower()
    # eliminate email addresses
    if not keep_emails:
        text = re.sub(r"\S+@\S+", " ", text)
    # eliminate @mentions
    if not keep_at_mentions:
        text = re.sub(r"\s@\S+", " ", text)
    # replace underscores with spaces
    text = re.sub(r"_", " ", text)
    # break off single quotes at the ends of words
    text = re.sub(r"\s\'", " ", text)
    text = re.sub(r"\'\s", " ", text)
    # remove periods
    text = re.sub(r"\.", "", text)
    # replace all other punctuation (except single quotes) with spaces
    text = replace.sub(" ", text)
    # remove single quotes
    text = re.sub(r"\'", "", text)
    # replace all whitespace with a single space
    text = re.sub(r"\s", " ", text)
    # strip off spaces on either end
    text = text.strip()
    return text


if __name__ == "__main__":
    main(sys.argv[1:])

