import tensorflow as tf

from .char_rnn import char_rnn


def test_char_rnn():
    for i, (sentence_shape, labels_shape) in enumerate([
            [[11, 64, 8], [11, 64]],
            [[None, 64, 8], [None, 64]],
            [[None, None, 8], [None, None]],
            [[None, None, None], [None, None]]]):
        for mode in [tf.contrib.learn.ModeKeys.TRAIN,
                     tf.contrib.learn.ModeKeys.EVAL,
                     tf.contrib.learn.ModeKeys.INFER]:
            with tf.variable_scope('model_{}_{}'.format(i, mode)):
                char_rnn(tf.placeholder(tf.int32, sentence_shape),
                         tf.placeholder(tf.int32, labels_shape),
                         mode=mode,
                         num_classes=7,
                         char_space_size=128,
                         char_embedding_size=128,
                         word_embedding_size=128,
                         context_vector_size=128)
