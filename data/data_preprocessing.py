# -*- coding:utf-8 -*-
from collections import Counter
import pickle
import numpy as np
import tensorflow.contrib.keras as kr

# tag dict
tag2label = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}


def read_data(filename):
    content, label, sentences, tag = [], [], [], []
    with open(filename, encoding='utf-8') as file:
        lines = file.readlines()
    for eachline in lines:
        if eachline != '\n':
            [char, tag_] = eachline.strip().split()
            sentences.append(char)
            tag.append(tag_)
        else:
            content.append(sentences)
            label.append(tag)
            sentences, tag = [], []
    return content, label



def build_vocab(filenames, vocab_size=5000):
    wordlist = []
    word = {}
    word['<PAD>'] = 0
    j = 1
    for filename in filenames:
        content, _ = read_data(filename)
        for sen_ in content:
            wordlist.extend(sen_)
    counter = Counter(wordlist)
    count_pari = counter.most_common(vocab_size)
    word_, _ = list(zip(*count_pari))
    for key in word_:
        if key.isdigit():
            key = '<NUM>'
        if key not in word:
            word[key] = j
        j += 1

    word['<UNK>'] = j
    with open('word2id.pkl', 'wb') as fw:  # 将建立的字典 保存
        pickle.dump(word, fw)
    return word



def sequence2id(filename, word2id_file):
    content2id, label2id = [], []
    content, label = read_data(filename)
    with open(word2id_file, 'rb') as fr:
        word = pickle.load(fr)
    for i in range(len(label)):
        label2id.append([tag2label[x] for x in label[i]])
    for j in range(len(content)):
        w = []
        for key in content[j]:
            if key.isdigit():
                key = '<NUM>'
            elif key not in word:
                key = '<UNK>'
            w.append(word[key])
        content2id.append(w)
    return content2id, label2id



if __name__ == '__main__':
    sentences, labels = read_data('train_data')
    print("corpus including {0} sentences".format(len(sentences)))

    file_names = ['train_data', 'test_data']
    word2id = build_vocab(file_names)
    print("words in corpus: {}".format(len(word2id)))

    content2id, label2id = sequence2id('train_data')
    print('sentence example: {}'.format(content2id[1]))
    print('label example: {}'.format(label2id[1]))