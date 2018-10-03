# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf
from transformer_modules import *


class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[196, 512], dim_embed=512, n_time_step=16,
                 num_blocks = 6, num_heads = 8, dropout=0.1, is_training = True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        #self.prev2out = prev2out
        #self.ctx2out = ctx2out
        #self.alpha_c = alpha_c
        #self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        #self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        #self.weight_initializer = tf.contrib.layers.xavier_initializer()
        #self.const_initializer = tf.constant_initializer(0.0)
        #self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])

    def build_model(self):

        enc = self.features
        N = self.features[0]

        with tf.variable_scope("encoder"):
            enc += positional_encoding(N,
                        self.L,
                        num_units=self.D,
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe")


            ## Dropout
            enc = tf.layers.dropout(enc,
                                    rate=self.dropout,
                                    training=tf.convert_to_tensor(is_training))

            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                ### Multihead Attention
                enc = multihead_attention(queries=enc,
                                            keys=enc,
                                            num_units=self.D,
                                            num_heads=self.num_heads,
                                            dropout_rate=self.dropout,
                                            is_training=is_training,
                                            causality=False)

                ### Feed Forward
                enc = feedforward(enc, num_units=[4*self.D, self.D])

            # Decoder
        with tf.variable_scope("decoder"):
            ## Embedding
            dec = embedding(self.captions,
                              vocab_size=self.V,
                              num_units=self.M,
                              scale=True,
                              scope="dec_embed")

            ## Positional Encoding
            dec += positional_encoding(N,
                                  self.T,
                                  num_units=self.M,
                                  zero_pad=False,
                                  scale=False,
                                  scope="dec_pe")

            ## Dropout
            dec = tf.layers.dropout(dec,
                                    rate=self.dropout,
                                    training=tf.convert_to_tensor(is_training))

            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    dec = multihead_attention(queries=dec,
                                                keys=dec,
                                                num_units=self.M,
                                                num_heads=self.num_heads,
                                                dropout_rate=self.dropout,
                                                is_training=is_training,                                                    causality=True,
                                                scope="self_attention")

                    ## Multihead Attention ( vanilla attention)
                    dec = multihead_attention(queries=dec,
                                                keys=enc,
                                                num_units=self.M,
                                                num_heads=self.num_heads,
                                                dropout_rate=self.dropout,
                                                is_training=is_training,
                                                causality=False,
                                                scope="vanilla_attention")

                    ## Feed Forward
                    dec = feedforward(self.dec, num_units=[4*self.M, self.M])

        # Final linear projection
        logits = tf.layers.dense(dec, self.V)
        preds = tf.to_int32(tf.arg_max(logits, dimension=-1))
        istarget = tf.to_float(tf.not_equal(self.captions, 0))
        acc = tf.reduce_sum(tf.to_float(tf.equal(preds, self.captions))*istarget)/ (tf.reduce_sum(istarget))
        tf.summary.scalar('acc', acc)

        if is_training:
            # Loss
            y_smoothed = label_smoothing(tf.one_hot(self.captions, depth=self.V)))
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_smoothed)
            mean_loss = tf.reduce_sum(loss*istarget) / (tf.reduce_sum(istarget))

        return mean_loss

    def build_sampler(self, max_len=20):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat( [x, context],1), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=(t!=0))
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return alphas, betas, sampled_captions
