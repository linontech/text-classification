# -*- coding: utf-8 -*-
# @Time    : 2018/4/3 18:18
# @Author  : lzw
import random
import re
from collections import namedtuple
from contextlib import contextmanager
from itertools import tee
from timeit import default_timer
import jieba
import os

jieba.load_userdict('../dataset/digital forum comments/digital_selfdict.txt')
Document = namedtuple('Document', 'words tags split label')

def Chinese_word_extraction(content_raw):
    # chinese_pattern = u"([\u4e00-\u9fa5]+)"
    chinese_pattern = u"([\u4e00-\u9fa5a-zA-Z]+)"
    chi_pattern = re.compile(chinese_pattern)
    listOfwords = chi_pattern.findall(content_raw)
    return listOfwords


def accept_sentence(sentnece):
    return Chinese_word_extraction(sentnece)

def split_file_classes(path):
    """
    helper to split data into classes of files
    e.g:
        class_name_count = split_file_clsses('../dataset/digital forum comments/split_20180403/all.txt')
        for i,j in class_name_count.items():
            print (i, j)
    """
    class_name_count={}
    class_name_file={}
    label_index={}
    _label=0
    with open(path, encoding='utf-8') as texts:
        for line_no, line in enumerate(texts):
            text, label = line.split('\t')
            label=label.rstrip()
            if label_index.get(label,None) not in class_name_count.keys():
                label_index[label] = _label
                class_name_count[label_index[label]]=1
                class_name_file[label]=open('../dataset/digital forum comments/split_20180403/classes/'+label+'.txt', 'w', encoding='utf-8')
                class_name_file[label].write(text+'\t'+label+'\n')
                _label+=1
            else:
                class_name_count[label_index[label]] = class_name_count.get(label_index[label],0) + 1
                class_name_file[label].write(text+'\t'+label+'\n')
    for f in class_name_file.values():
        f.close()
    return label_index, class_name_count


"""
    read text data
"""
def load_text_label(text_path, classes_num_count, label_index, portion, shuffle=True):
    """
    classes_num: dict of num of each class
    split: portion [train_portion, dev_portion, ...]
    label_index: labelname to index
    p.s you should know how many texts each classes got.
    p.s you should shuffle texts of each classes first, due to farly splitting dataset.
    """
    train_lens, dev_lens, train_len_total, dev_len_total = {}, {}, 0, 0
    for _class, num in classes_num_count.items():
        if _class==label_index['未知']:   continue
        train_lens[_class] = int(num * portion[0])
        dev_lens[_class] = num - train_lens[_class]
        train_len_total += train_lens[_class]
        dev_len_total += dev_lens[_class]

    documents=[]
    files = os.listdir(text_path)
    tag_id=0
    for file in files:  # filename=labelname
        # classes_name = file.split('.')[0]
        with open(text_path+file, encoding='utf-8') as texts:
            for line_no, line in enumerate(texts):
                if line == '': continue
                text, label = line.split('\t')
                label = label.rstrip()
                text = accept_sentence(text)
                words = jieba.lcut(''.join(text), HMM=False)
                tags = [tag_id]
                if label == '未知':
                    split = 'UNK'
                    label=-1
                else:
                    label = label_index[label]
                    split = 'train' if line_no < train_lens[label] else 'dev'
                documents.append(Document(words, tags, split, label))  # yield Document(words, tags, label)
                tag_id+=1

    assert train_len_total==len([d for d in documents if d.split=='train']), 'train text load error!'
    assert dev_len_total==len([d for d in documents if d.split=='dev']), 'train text load error!'

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

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


