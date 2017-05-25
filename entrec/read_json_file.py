import json

import listpad
import numpy as np
import qnd
import qndex
import tensorflow as tf


def def_convert_json_example():
    """Define json example converter.

    An example is the following.

    ```json
    [
        { "word": "I",   "label": 0 },
        { "word": "am",  "label": 0 },
        { "word": "Bob", "label": 1 },
        { "word": ".",   "label": 0 }
    ]
    ```
    """
    chars = qndex.nlp.def_chars()

    def convert_json_example(string):
        char_indices = {char: index for index, char in enumerate(chars())}

        def convert(string):
            words, labels = zip(*[(pair['word'], pair['label'])
                                  for pair in json.loads(string.decode())])
            word_length = max(len(word) for word in words)

            return tuple(map(
                lambda x: np.array(x, np.int32),
                [listpad.ListPadder([word_length, None], qndex.nlp.NULL_INDEX)
                 .pad([[(char_indices[char]
                         if char in char_indices else
                         qndex.nlp.UNKNOWN_INDEX)
                        for char in word]
                       for word in words]),
                 labels,
                 len(words),
                 word_length]))

        sentence, labels, sentence_length, word_length = tf.py_func(
            convert,
            [string],
            [tf.int32, tf.int32, tf.int32, tf.int32],
            name="convert_json_example")

        sentence_length.set_shape([])
        word_length.set_shape([])

        return (tf.reshape(sentence, [sentence_length, word_length]),
                tf.reshape(labels, [sentence_length]))

    return convert_json_example


def def_read_json_file():
    convert_json_example = def_convert_json_example()

    def read_json_file(filename_queue):
        key, value = tf.WholeFileReader().read(filename_queue)
        sentence, labels = convert_json_example(value)
        return {'key': key, 'sentence': sentence}, {'labels': labels}

    return read_json_file
