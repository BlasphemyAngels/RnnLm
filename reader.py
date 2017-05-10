# !/usr/bin/python3
# _*_coding: utf-8_*_

import os
import jieba
import collections
import tensorflow as tf

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return list(jieba.cut(f.read().decode("utf-8").replace("\n",  "<eos>"), cut_all=False))

def _build_vocab(filename):

    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))

    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

def _file_to_word_ids(filename, word_to_id):

    data = _read_words(filename)
    return [word_to_id[w] for w in data if w in word_to_id]



def raw_data(data_path=None):
    train_path = os.path.join(data_path, "a.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)

    vocabulary = len(word_to_id)
    return train_data, vocabulary



if __name__ == "__main__":
    train_data, vocabulary = raw_data("")
    print(train_data)
