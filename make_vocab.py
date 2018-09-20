from hyperparameter import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import os
import regex
from collections import Counter
import pandas as pd
import pickle

def make_vocab(caption_string, fname):
    '''
    Constructs vocabulary.

    Args:
      caption_string: A string with all captions.
      fname: A string. Output file name.

    Writes vocabulary line by line to `preprocessed/fname`

    '''

    text = regex.sub("[^\s\p{Latin}']", "", caption_string)
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))

def pickle_to_string(fpath):
    df = pd.read_pickle(fpath)
    cap = ''
    for index, row in df.iterrows():
        cap = cap + ' ' + row['caption']
        if index % 100 == 0:
            print(index, "words converted to string")

    return cap

if __name__ == '__main__':
    caption_string_train = pickle_to_string("./train_lookup_table.pkl")
    caption_string_val = pickle_to_string("./val_lookup_table.pkl")
    caption_string_all = caption_string_train + " " + caption_string_val
    make_vocab(caption_string_all, "./vocab.tsv")

    print("Done")
