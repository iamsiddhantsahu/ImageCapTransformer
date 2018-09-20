from __future__ import print_function
from hyperparameter import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex
import pandas as pd
import PIL

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('./preprocessed/vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_train_data():
    word2idx, idx2word = load_en_vocab()

    x_list, y_list = [], []
    df = pd.read_pickle('./train_lookup_table.pkl')
    for index, row in df.iterrows():
        #load image for each row in DataFrame
        image = np.asarray(PIL.Image.open(row['file_name']))
        x_list.append(image) #appending each image to x_list
        #load caption for each row in DataFrame
        caption = row['caption']
        y = [word2idx.get(word, 1) for word in (caption + u" </S>").split()] #add </S> end of each caption
        y_list.append(np.array(y)) #append caption to y_list

        if index % 1000 == 0:
            print(index, "images and captions are loaded")

    #padding of y_list
    y_padded = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, y in enumerate(y_list):
        y_padded[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))

    return x_list, y_padded



def get_batch_data():
    # Load data
    X, Y = load_train_data()

    # calc total batch count
    num_batch = len(X) // hp.batch_size

    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)

    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])

    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size,
                                capacity=hp.batch_size*64,
                                min_after_dequeue=hp.batch_size*32,
                                allow_smaller_final_batch=False)

    return x, y, num_batch # (N, T), (N, T), ()
