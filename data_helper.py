# -*- coding: utf-8 -*-
# @Author  : lzw

"""
# description:
    1. background: you have some text data with multiple labels in one file.
        split_file_classes(), load_text_label()
    2. build word_dict.( word2count, word2index, index2word )
        build_vocabulary()
    3. generate batches from input texts for a doc2vec model.


"""
import datetime
import os
import random
import re
from collections import namedtuple
import jieba
import numpy as np
import logging
logging.getLogger().setLevel(logging.INFO)


def split_file_classes(source_path, output_path):
    """
    split big file into small files which contains different labels.
    e.g:
        class_name_count = split_file_clsses('../dataset/digital forum comments/split_20180403/all.txt')
        for i,j in class_name_count.items():
            print (i, j)
    """
    class_name_count = {}
    class_name_file = {}
    label_index = {}
    _label = 0
    with open(source_path, encoding='utf-8') as texts:
        for line_no, line in enumerate(texts):
            text, label = line.split('\t')
            label = label.rstrip()
            if label_index.get(label, None) not in class_name_count.keys():
                label_index[label] = _label
                class_name_count[label_index[label]] = 1
                class_name_file[label] = open(output_path + label + '.txt', 'w', encoding='utf-8')
                class_name_file[label].write(text + '\t' + label + '\n')
                _label += 1
            else:
                class_name_count[label_index[label]] = class_name_count.get(label_index[label], 0) + 1
                class_name_file[label].write(text + '\t' + label + '\n')
    for f in class_name_file.values():
        f.close()
    return label_index, class_name_count

"""
the newest one.
"""
def load_text_label(text_path, classes_num_count, label_index, portion, shuffle=True):
    """
    load multiple files which contains different labels into type "namedtuple('Document', 'words tags split label')"

    classes_num: dict of num of each class
    split: portion [train_portion, dev_portion, ...]
    label_index: labelname to index
    p.s you should shuffle texts of each classes first, due to farly splitting dataset.
    """
    Document = namedtuple('Document', 'words tags split label')
    train_lens, dev_lens, train_len_total, dev_len_total = {}, {}, 0, 0
    for _class, num in classes_num_count.items():
        if _class == label_index['未知']:   continue
        train_lens[_class] = int(num * portion[0])
        dev_lens[_class] = num - train_lens[_class]
        train_len_total += train_lens[_class]
        dev_len_total += dev_lens[_class]

    documents = []
    files = os.listdir(text_path)
    tag_id = 0
    for file in files:  # filename=labelname
        # classes_name = file.split('.')[0]
        with open(text_path + file, encoding='utf-8') as texts:
            for line_no, line in enumerate(texts):
                if line == '': continue
                text, label = line.split('\t')
                label = label.rstrip()
                text = accept_sentence(text)
                words = jieba.lcut(''.join(text), HMM=False)
                tags = [tag_id]
                if label == '未知':
                    split = 'UNK'
                    label = -1
                else:
                    label = label_index[label]
                    split = 'train' if line_no < train_lens[label] else 'dev'
                documents.append(Document(words, tags, split, label))  # yield Document(words, tags, label)
                tag_id += 1

    assert train_len_total == len([d for d in documents if d.split == 'train']), 'train text load error!'
    assert dev_len_total == len([d for d in documents if d.split == 'dev']), 'train text load error!'

    if shuffle: random.shuffle(documents)
    return documents, train_len_total, dev_len_total


def load_data_without_store_ixs_labeled(filepath, data_size=350000):
    """
    random_sampler for a large texts which can not fit in memory
    read a large text file with random line access
    p.s if you want to generate len of samples which equal the len of text, this method just fails.
    """
    Document = namedtuple('Document', 'words tags split label')
    with open(filepath, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()
        random_set = sorted(random.sample(range(filesize), data_size))
        count = 0
        split='None'
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
            label = 1 if label == '负面' else 0
            yield Document(words, tag, split, int(label))

def load_text_label_test(filepath, label_index):
    """
    for test text
    """
    Document = namedtuple('Document', 'words tags split label')
    file = open(filepath, encoding='utf-8')
    lines = file.read().split('\n')
    file.close()
    documents = []
    split='test'
    tag_id=0
    for line in lines:
        text = line.split('\t')[0]
        label = line.split('\t')[1]
        text = accept_sentence(text)
        text = ' '.join(jieba.lcut(''.join(text)))
        documents.append(Document(text,[tag_id],split,label_index[label]))
        tag_id+=1
    return documents

def load_data_with_store_ixs(filepath, data_size=350000):
    """
    read a large text file with random line access
    e.g:
     train_texts = utils.load_data(filepath='dataset/sentence_20180318_without_stopwords.txt',
                                   sample_file_path='dataset/sample_sentences.txt',
                                   data_size=100000) # 175000
    """
    # with open(sample_file_path, encoding='utf-8') as file:
    #     sample_count = 0
    #     for line in file:
    #         text = line.split('\t')[0].replace(' ', '').replace('\n', '')
    #         text = accept_sentence(text)
    #         text = jieba.lcut(''.join(text))
    #         yield ' '.join(text)
    #         sample_count += 1
    #
    # assert sample_count == 10

    try: # load indexes if exists
        ixs = load_obj('dataset/ix_for_big_data_' + str(data_size) + '.pkl')
        logging.info('Indexes file found. ')
        create_new_indexes = False
    except:
        create_new_indexes = True
        logging.info('No indexes file found. ')

    thefile = open('dataset/the100000file.txt', 'w')
    with open(filepath, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()
        if create_new_indexes:
            random_set = sorted(random.sample(range(filesize), data_size))
            logging.info('saving indexes file. ')
            save_obj(random_set, 'dataset/ix_for_big_data_' + str(data_size) + '.pkl')
        else:
            random_set = ixs

        for i in range(data_size):
            f.seek(random_set[i])
            f.readline()  # Skip current line (because we might be in the middle of a line)
            line = f.readline().rstrip()
            line = line.decode('utf-8')
            text = line.split('\t')[0].replace(' ', '').replace('\n', '')
            text = accept_sentence(text)
            text = ' '.join(jieba.lcut(''.join(text), HMM=False))
            thefile.write(text + '\n')
            yield text
            # if i%50000==0:
            #     print (i)
    thefile.close()


def build_vocabulary(input_data, stop_words_file='dataset/stopwords.txt', k=100, min_appear_freq=5,
                     to_save=False, to_save_file_info=''):
    """
    使用最常见的词语作为词库
    最常出现的k个词删除，因为它们很有可能是停用词
    要求词库中的词在语料库中出现次数大于频率prob，对于不在词库中的词语(停用词)使用 '__UNK__'
    :param input_data: listOfText，已经分词
    :param k 忽略 k most common words 中的停用词
    :param min_appear_freq 单词最小出现次数，参数根据语料库大小需要调整
    :return: dict
    """
    logging.info(str(datetime.now()).replace(':', '-') + '  Start building vocabulary. ')
    stop_words_file = open(stop_words_file, encoding='utf-8')
    stop_words = set(stop_words_file.read().split('\n'))
    stop_words_file.close()

    word2count, unk_count = {}, 0
    for text in input_data:
        list_of_words = accept_sentence(text)
        for word in list_of_words:
            if word in stop_words:
                unk_count += 1  # 这里用出现的停用词个数当作近似的unk_count，避免建立vocab后还要重新遍历，
            word2count[word] = word2count.get(word, 0) + 1  # 但是只把k_common中的停用词从词典中去掉

    sort_word2count = sorted(word2count.items(), key=lambda x: x[1], reverse=True)

    remove_sotp_words = []
    most_common_words, _ = zip(*sort_word2count[:k])
    for most_common_word in most_common_words:
        if most_common_word in stop_words:
            remove_sotp_words.append(most_common_word)
            word2count.pop(most_common_word)

    word2count['UNK'] = unk_count

    index = 2
    word2index = {'_UNK_': 0, '_NULL_': 1}  # '_<s>_': -1, '_</s>_': -2}
    remove_words = []
    for word, count in word2count.items():
        if count >= min_appear_freq:
            word2index[word] = index
            index += 1
        else:
            remove_words.append(word)
    for word in remove_words:
        word2count.pop(word)

    index2word = dict(zip(word2index.values(), word2index.keys()))
    remove_words = remove_words + remove_sotp_words
    logging.info('{} remove words: \n {}'.format(len(remove_words), str(remove_words)))
    logging.info(str(datetime.now()).replace(':', '-') + '  End building vocabulary. length of vocabulary: {}'.format(
        len(index2word)))

    if to_save:
        if not os.path.exists('vocabulary/'):
            os.makedirs('vocabulary/')
        save_obj(word2count, 'vocabulary/word2count' + to_save_file_info)
        save_obj(word2index, 'vocabulary/word2index' + to_save_file_info)
        save_obj(index2word, 'vocabulary/index2word' + to_save_file_info)

    return word2count, word2index, index2word


def generate_texts2indexes(input_data, word2index):
    """
    transform a splited already chinese sentence into list of numbers in a "LabelSen" format.
    """
    LabelSen = namedtuple('LabeledSen', ['sentence_index', 'word_indexes'])
    texts2indexes_list = []
    sentence_index = 0
    for text in input_data:
        # yield LabelSen(sentence_index, words2ids(word2index, text))
        texts2indexes_list.append(LabelSen(sentence_index, words2ids(word2index, text)))
        sentence_index += 1
    return texts2indexes_list


"""
    Mikolov 2014 ：如果句子的长度小于window_size，那么剩下的单词用__NULL__标签代替，且放在前面。
    DONE: '__NULL__' 表示句子还不够一个window_size那么长的部分
    TODO: add num_skip variables to limit samples generate from one window
         num_skip: 一个窗口最多采样多少样本; prevent overfitting issue
"""
def generate_pvdm_batches(LabeledSentences, n_epochs, batch_size, window_size, shuffle=True):
    """
    generate batches for pvdm model.
    """
    sample_label_s = []
    for LabeledSentence in LabeledSentences:
        indexes = LabeledSentence.word_indexes
        sen_len = len(indexes)
        if sen_len < window_size:  # if len of sentence smaller than window_size prepad __null__ in the front
            null_words = np.array([1] * (window_size - sen_len))
            indexes = np.concatenate((null_words, indexes))
            sen_len = len(indexes)
            assert sen_len == window_size

        if sen_len == window_size:
            poses = [0]
        else:
            poses = range(0, sen_len - window_size + 1)

        for pos in poses:
            sample_label_s.append(
                [np.concatenate((np.array([LabeledSentence.sentence_index]), indexes[pos: pos + window_size - 1])), \
                 indexes[pos + window_size - 1: pos + window_size]])

    data = np.array(sample_label_s)
    data_size = len(sample_label_s)
    num_batches_per_epoch = int(data_size / batch_size)+1  # num_of_step_per_epoch

    batches = []
    for epoch in range(n_epochs):  # batch_iter
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            # if len(shuffled_data[start_index:end_index]) != batch_size:
            #     continue # throw away the rest samples
            # yield shuffled_data[start_index:end_index]
            batches.append(shuffled_data[start_index:end_index])
    return batches


def generate_pvdbow_batches(LabeledSentences, n_epochs, batch_size, window_size, shuffle=True):
    """
    generate batches for pvdbow model.
    """
    sample_label_s = []
    for LabeledSentence in LabeledSentences:
        indexes = LabeledSentence.word_indexes
        sen_len = len(indexes)
        if sen_len < window_size:  # if len of sentence smaller than window_size
            null_words = np.array([1] * (window_size - sen_len))
            indexes = np.concatenate((null_words, indexes))
            sen_len = len(indexes)
            assert sen_len == window_size
        if sen_len == window_size:
            poses = [0]
        else:
            poses = range(0, sen_len - window_size + 1)
        for pos in poses:
            sample_label_s.append([np.array([LabeledSentence.sentence_index]), indexes[pos: pos + 1]])

    data = np.array(sample_label_s)
    data_size = len(sample_label_s)
    num_batches_per_epoch = int(data_size / batch_size)+1
    batches = []
    for epoch in range(n_epochs):  # batch_iter
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))  # memory error here
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):  # start from 0
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            # yield shuffled_data[start_index:end_index]
            batches.append(shuffled_data[start_index:end_index])
    return batches


def Chinese_word_extraction(content_raw):
    """
    rules to parse a sentence
    """
    chinese_pattern = u"([\u4e00-\u9fa5a-zA-Z]+)"
    chi_pattern = re.compile(chinese_pattern)
    listOfwords = chi_pattern.findall(content_raw)
    return listOfwords


def accept_sentence(sentnece):
    """
    """
    return Chinese_word_extraction(sentnece)


def words2ids(word2id, words):
    """
    transform list of words (sentence) into list of numbers based on word2index dict
    """

    def word_to_id(word):
        id = word2id.get(word)
        if id is None:  # 如果字典里面没有这个词的情况，变成_UNK_的index
            id = 0  # '__UNK__’ 的index
        return id

    x = list(map(word_to_id, words))
    return np.array(x)


def ids2words(id2word, ids):
    """
    transform list of numbers (sentence) into list of words based on index2word dict
    """

    def id_to_word(id):
        word = id2word.get(id)
        return word

    words = list(map(id_to_word, ids))
    return words  # list


def save_obj(obj, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def copy_gen(_gen):
    """
        copy a generator
    """
    from itertools import tee
    return tee(_gen)


def ovs_minority_text(texts, minority_texts, k, n=2,
                       stop_words_file='dataset/stopwords.txt'):
    """
        simple generate positive samples by traditional language model until meet </s>
        things to try:
            different len of sentences (median, majority)
            generate sentence from all sentences' bigram model
            take into account the frequency of bigrams
            take into account to ignore stopwords
            slow speed 1.5k sentence, about 1 hr

    :param texts: already cutted sentences
    :param minority_texts: LabeledSen list
    :param n: n=2 stands for unigram, bigram; try bigram first
    :param k: how many sentences to generate, k= num*len(minority_text)
    :return: generate sentences
    """
    if n != 2:
        logging.critical('Not implemented yet. :)')
    logging.info('Attempt to generate {} sentences'.format(k))
    # build bigram indexes for all sentences
    bigrams_dict = {}
    unigram_dict = {}
    idf_dict = {}
    for text in texts:
        text = '<s> ' + text + ' </s>'
        word_list = text.split(' ')
        indexes_len = len(word_list)
        tmp = set()
        for pos in range(indexes_len - 1):
            key = (word_list[pos], word_list[pos + 1])
            bigrams_dict[key] = bigrams_dict.get(key, 0) + 1
            if pos != 0:
                word = word_list[pos]
                unigram_dict[word] = unigram_dict.get(word, 0) + 1
                if (len(tmp) == 0) or ((len(tmp) != 0) and (word not in tmp)):
                    idf_dict[word] = idf_dict.get(word, 0) + 1  # idf value from all sentence

    sen_len = len(texts)
    tf_idf_rs = []
    minority_bigrams_dict = {}
    sentences_count = []
    for text in minority_texts:
        word_list = text.split(' ')
        indexes_len = len(word_list)
        sentences_count.append(indexes_len)
        rs = []
        for word in set(word_list):
            rs.append((word, (word_list.count(word) / indexes_len) * np.log(sen_len / idf_dict[word])))
        tf_idf_rs.append(rs)
        for pos in range(indexes_len - 1):
            key = (word_list[pos], word_list[pos + 1])
            minority_bigrams_dict[key] = minority_bigrams_dict.get(key, 0) + 1  # minority_bigrams_dict

    # calculate tf-idf of every words; how a word affect a sentence
    logging.info('len of bigram dict: {}'.format(len(bigrams_dict)))
    logging.info('len of minority bigram dict: {}'.format(len(minority_bigrams_dict)))
    logging.info('len of unigram dict: {}'.format(len(unigram_dict)))

    tf_idf_each_sentence, start_words = [], []
    for word_tf_idf in tf_idf_rs:
        tf_idf_sorted = sorted(word_tf_idf, key=lambda x: x[1], reverse=True)
        start_words += tf_idf_sorted[:3]

    # choose big tf-idf words for each minority sentences as the begin when generating sentenece
    # try: subsample k words from start_words
    median_sen_len = np.median(sentences_count)
    logging.info(
        'generating minority sentences with len {}, from {} start words.'.format(median_sen_len, len(start_words)))
    generated_sens = []
    random.shuffle(start_words)
    logging.info('start ' + str(datetime.now()).replace(':', '-'))
    for c, word in enumerate(start_words):
        generated_sen = '' + word[0]
        count = 0
        next_word = word[0]
        while (1):
            next_words = [(key, value) for key, value in minority_bigrams_dict.items() if key[0] == next_word]
            next_words_keys = [key[1] for key, value in next_words]
            next_words_value = [value for key, value in next_words]
            deno = sum(next_words_value)
            next_words_freq = [value / deno for value in next_words_value]
            next_words_num = len(next_words_keys)
            if next_words_num != 0:
                next_word = np.random.choice(next_words_keys,
                                             p=next_words_freq)  # condition with freq of minority_bigrams
            if (next_words_num == 0) or (next_word == '<\s>') or (
                        count > median_sen_len):  # count > average length sentences_count/sen_len
                generated_sen += ' <\s>'  # majority num or median
                break
            generated_sen += ' ' + next_word
            count += 1
        generated_sens.append(generated_sen)
        # print(generated_sen)
        if c > k:
            break
        if c % 1000 == 0:
            print(c)
        c += 1
        # print (generated_sen)
    logging.info('end ' + str(datetime.now()).replace(':', '-'))

    return generated_sens