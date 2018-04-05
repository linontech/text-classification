# -*- coding: utf-8 -*-
# @Time    : 2018/4/3 18:18
# @Author  : lzw
import random
from collections import namedtuple
from itertools import tee
import jieba
import numpy as np
import os

from utils import accept_sentence

jieba.load_userdict('../dataset/digital forum comments/digital_selfdict.txt')
Document = namedtuple('Document', 'words tags split label')

"""
    read text data
"""
def load_text_label(text_path, classes_num, portion, shuffle=True):
    """
    classes_num: dict of num of each class
    split: portion [train_portion, dev_portion, ...]
    p.s you should know how many texts each classes got.
    p.s you should shuffle texts of each classes first, due to farly splitting dataset.
    """
    train_lens, dev_lens, train_len_total, dev_len_total = {}, {}, 0, 0
    for _class, num in classes_num.items():
        train_lens[_class] = int(num * portion[0])
        dev_lens[_class] = num - train_lens[_class]
        train_len_total += train_lens[_class]
        dev_len_total += dev_lens[_class]

    documents=[]
    files = os.listdir(text_path)
    for file in files:  # filename=labelname
        classes_name = file.split('.')[0]
        with open(text_path+file, encoding='utf-8') as texts:
            for line_no, line in enumerate(texts):
                text, label = line.split('\t')
                text = accept_sentence(text)
                words = jieba.lcut(''.join(text), HMM=False)
                tags = [line_no]
                label = 1 if label.rstrip()=='负面' else 0
                split = 'train' if line_no<train_lens[classes_name] else 'dev'
                documents.append(Document(words, tags, split, label))  # yield Document(words, tags, label)
    assert train_len_total==len([d for d in documents if d.split=='train']), 'train text load error!'
    if shuffle: random.shuffle(documents)
    return documents, train_len_total, dev_len_total

def load_data_without_store_ixs_labeled(filepath, data_size=350000):
    """
    random_sampler
    read a large text file with random line access
    """
    # random_sampler for big file which does not fit in memory
    # random_nums = set(random.sample(range(44001245), data_size))
    # if you want to generate len of samples which equal the len of text, this method just fails.
    with open(filepath, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()
        random_set = sorted(random.sample(range(filesize), data_size))
        count=0
        for i in range(data_size):
            f.seek(random_set[i])
            f.readline()  # Skip current line (because we might be in the middle of a line)
            line = f.readline().rstrip()
            line = line.decode('utf-8')

            text, label = line.split('\t')
            text = accept_sentence(text)
            # print (' '.join(text))
            words = jieba.lcut(''.join(text), HMM=False)
            tag = [count]
            count += 1
            label = 1 if label=='负面' else 0
            yield Document(words, tag, int(label))


def copy_gen(_gen):
    """
        copy a generator
    """
    return tee(_gen)