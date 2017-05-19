import extenteten as ex
import tensorflow as tf


@ex.func_scope()
def char_rnn(sentence,
             labels,
             *,
             mode,
             num_classes,
             char_space_size,
             char_embedding_size,
             word_embedding_size,
             context_vector_size):
    assert ex.static_rank(sentence) == 3
    assert ex.static_rank(labels) == 2

    char_embeddings = ex.embeddings(id_space_size=char_space_size,
                                    embedding_size=char_embedding_size)

    word_embeddings = tf.reshape(
        ex.bidirectional_id_vector_to_embedding(
            tf.reshape(sentence, [-1, ex.static_shape(sentence)[2]]),
            char_embeddings,
            output_size=word_embedding_size,
            context_vector_size=context_vector_size,
            dynamic_length=True),
        [-1, ex.static_shape(sentence)[1], word_embedding_size])

    sentence_lengths = ex.id_tensor_to_length(sentence)

    rnn_outputs = ex.bidirectional_rnn(
        word_embeddings,
        output_size=word_embedding_size,
        sequence_length=sentence_lengths)

    logits = tf.reshape(
        ex.mlp(tf.reshape(rnn_outputs, [-1, ex.static_shape(rnn_outputs)[2]]),
               layer_sizes=[word_embedding_size, num_classes]),
        [-1, ex.static_shape(sentence)[1], num_classes])

    loss = tf.reduce_mean(
        tf.reduce_sum(
            (tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits)
             * tf.sequence_mask(sentence_lengths,
                                maxlen=ex.static_shape(sentence)[1],
                                dtype=tf.float32)),
            axis=1)
        / tf.to_float(sentence_lengths))

    return tf.contrib.learn.ModelFnOps(
        mode,
        predictions=tf.argmax(logits, axis=2),
        loss=loss,
        train_op=ex.minimize(loss))
