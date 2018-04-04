# -*- coding: utf-8 -*-
# @Time    : 2018/4/3 18:18
# @Author  : lzw
import random
from collections import namedtuple
from itertools import tee
import jieba
import numpy as np

from utils import accept_sentence

jieba.load_userdict('../dataset/digital forum comments/digital_selfdict.txt')
Document = namedtuple('Document', 'words tags label')

"""
    read text data
"""
def load_text_label(text_path):
    """
    scan through the texts one time, and return how many texts each class has.
    one class per this function.
    p.s you should know how many texts each classes got.
    p.s you should shuffle texts of each classes first, due to farly splitting dataset.
    """
    count = 0
    with open(text_path, encoding='utf-8') as texts:
        for line_no, line in enumerate(texts):
            text, label = line.split('\t')
            words = jieba.lcut(text, HMM=False)
            tags = [line_no]
            count += 1
            yield Document(words, tags, label)


def load_data_without_store_ixs_labeled(filepath, data_size=350000):
    """
    random_sampler
    read a large text file with random line access
    """
    # random_sampler for big file which does not fit in memory
    # generate random numer to sample data, size=350000 due to limit of machine
    # random_nums = set(random.sample(range(44001245), data_size))
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


def getVecs(model, corpus_size, vec_size):
    vecs = [np.array(model.docvecs[z].reshape(1, vec_size)) for z in range(corpus_size)]
    return np.concatenate(vecs)

def copy_gen(_gen):
    """
        copy a generator
    """
    return tee(_gen)