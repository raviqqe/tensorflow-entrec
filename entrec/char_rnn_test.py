import tensorflow as tf

from .char_rnn import char_rnn


def test_char_rnn():
    for mode in [tf.contrib.learn.ModeKeys.TRAIN,
                 tf.contrib.learn.ModeKeys.EVAL,
                 tf.contrib.learn.ModeKeys.INFER]:
        with tf.variable_scope(mode):
            char_rnn(tf.placeholder(tf.int32, [None, 64, 8]),
                     tf.placeholder(tf.int32, [None, 64]),
                     mode=mode,
                     num_classes=7,
                     char_space_size=128,
                     char_embedding_size=128,
                     word_embedding_size=128,
                     context_vector_size=128)
