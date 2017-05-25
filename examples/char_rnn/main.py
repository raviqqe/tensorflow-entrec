#!/usr/bin/env python

import logging

import entrec
import qnd
import qndex


model = entrec.def_char_rnn()
read_file = entrec.def_read_json_file()
train_and_evaluate = qnd.def_train_and_evaluate()


def main():
    logging.getLogger().setLevel(logging.INFO)
    train_and_evaluate(model, read_file)


if __name__ == '__main__':
    main()
