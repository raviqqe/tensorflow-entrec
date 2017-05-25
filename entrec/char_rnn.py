import extenteten as ex
import qnd
import tensorflow as tf


@ex.func_scope()
def char_rnn(sentence,
             labels=None,
             *,
             mode,
             num_classes,
             char_space_size,
             char_embedding_size,
             word_embedding_size,
             context_vector_size):
    assert ex.static_rank(sentence) == 3
    assert ex.static_rank(labels) == 2 if labels is not None else True

    char_embeddings = ex.embeddings(id_space_size=char_space_size,
                                    embedding_size=char_embedding_size)

    word_embeddings = tf.reshape(
        ex.bidirectional_id_vector_to_embedding(
            tf.reshape(sentence,
                       [tf.shape(sentence)[0] * tf.shape(sentence)[1],
                        tf.shape(sentence)[2]]),
            char_embeddings,
            output_size=word_embedding_size,
            context_vector_size=context_vector_size,
            dynamic_length=True),
        [tf.shape(sentence)[0],
         tf.shape(sentence)[1],
         2 * word_embedding_size])

    sentence_length = ex.id_tensor_to_length(sentence)

    rnn_outputs = ex.bidirectional_rnn(
        word_embeddings,
        output_size=word_embedding_size,
        sequence_length=sentence_length)

    logits = tf.reshape(
        ex.mlp(tf.reshape(rnn_outputs,
                          [tf.shape(rnn_outputs)[0] * tf.shape(rnn_outputs)[1],
                           ex.static_shape(rnn_outputs)[2]]),
               layer_sizes=[word_embedding_size, num_classes]),
        [tf.shape(sentence)[0], tf.shape(sentence)[1], num_classes])

    sentence_mask = tf.sequence_mask(sentence_length,
                                     maxlen=ex.static_shape(sentence)[1],
                                     dtype=tf.float32)

    loss = tf.reduce_mean(
        tf.reduce_sum(
            (tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits)
             * sentence_mask),
            axis=1)
        / tf.to_float(sentence_length)) if labels is not None else None

    predictions = tf.to_int32(tf.argmax(logits, axis=2))
    num_words = tf.to_float(tf.reduce_sum(sentence_length))

    return tf.contrib.learn.ModelFnOps(
        mode,
        eval_metric_ops={
            'accuracy': tf.contrib.metrics.streaming_mean(
                tf.reduce_sum(tf.to_float(tf.equal(labels, predictions))
                              * sentence_mask)
                / num_words,
                num_words)[1],
        } if labels is not None else None,
        predictions=predictions,
        loss=loss,
        train_op=ex.minimize(loss) if labels is not None else None)


def def_char_rnn():
    adder = qnd.FlagAdder()
    adder.add_required_flag('num_classes', type=int)
    adder.add_required_flag('char_space_size', type=int)
    adder.add_flag('char_embedding_size', type=int, default=64)
    adder.add_flag('word_embedding_size', type=int, default=128)
    adder.add_flag('context_vector_size', type=int, default=128)

    def model(sentence, labels=None, *, mode, key=None):
        return char_rnn(sentence, labels, mode=mode, **adder.flags)

    return model
