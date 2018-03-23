# -*- coding: utf-8 -*-
# @Author  : lzw
import logging
import random
import re
from collections import namedtuple
from datetime import datetime

import jieba
import numpy as np

random.seed(3)
logging.getLogger().setLevel(logging.INFO)
jieba.load_userdict('dataset/digital forum comments/digital_selfdict.txt')

LabelSen = namedtuple('LabeledSen', ['sentence_index', 'word_indexes'])


def save_model(model, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_model(filename):
    import pickle
    with open(filename, 'rb') as f:
        model = pickle.load(f)
        # 测试读取后的Model
    return model


def Chinese_word_extraction(content_raw):
    chinese_pattern = u"([\u4e00-\u9fa5]+)"
    chi_pattern = re.compile(chinese_pattern)
    listOfwords = chi_pattern.findall(content_raw)
    return listOfwords


def accept_sentence(sentnece):
    return Chinese_word_extraction(sentnece)
# def accept_word(word):
#     return

def words2ids(word2id, words):
    def word_to_id(word):
        id = word2id.get(word)
        if id is None:  # 如果字典里面没有这个词的情况
            id = 0  # '__UNK__’ 的index
        return id

    x = list(map(word_to_id, words))
    return np.array(x)


def ids2words(id2word, ids):
    def id_to_word(id):
        word = id2word.get(id)
        return word

    words = list(map(id_to_word, ids))
    return words  # list


def build_vocabulary(input_data, stop_words_file='dataset/stopwords.txt', k=100, min_appear_freq=5, window_size=3):
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
    word2index = {'_UNK_': 0, '_NULL_': 1}
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

    return word2count, word2index, index2word


def generate_texts2indexes(input_data, word2index):
    """
    Mikolov 2014 ：如果句子的长度小于window_size，那么剩下的单词用__NULL__标签代替，且放在前面。
    num_skip: 一个窗口最多采样多少样本; prevent overfitting issue
    '__NULL__' 表示句子还不够一个window_size那么长
    :param input_data: listOfText，已经分词
    :param labels: 对应的标签
    :return: LabelDoc, [（编号，分词句子中词语index）]
    """
    texts2indexes_list = []
    sentence_index = 0
    for text in input_data:
        # text = accept_sentence(text)
        # yield LabelSen(sentence_index, words2ids(word2index, text))
        texts2indexes_list.append(LabelSen(sentence_index, words2ids(word2index, text)))
        sentence_index += 1
    return texts2indexes_list


def generate_pvdm_batches(LabeledSentences, n_epochs, batch_size, window_size, shuffle=True):
    """
    add num_skip variables to limit samples generate from one window
    :param batch_size:
    :param window_size:
    :param LabeledSentences:
    :return:
    """
    sample_label_s = []
    # bad=False
    for LabeledSentence in LabeledSentences:
        indexes = LabeledSentence.word_indexes
        sen_len = len(indexes)
        if sen_len < window_size:  # if len of sentence smaller than window_size prepad __null__ in the front
            null_words = np.array([1] * (window_size - sen_len))
            indexes = np.concatenate((null_words, indexes))
            sen_len = len(indexes)
            assert sen_len == window_size
        if sen_len == window_size:
            # bad=True
            poses = [0]
        else:
            poses = range(0, sen_len - window_size + 1)

        for pos in poses:
            sample_label_s.append(
                [np.concatenate((np.array([LabeledSentence.sentence_index]), indexes[pos: pos + window_size - 1])), \
                 indexes[pos + window_size - 1: pos + window_size]])

            # if bad:
            #     print (indexes)
            #     print (np.array([LabeledSentence.sentence_index]) + indexes[pos: pos+window_size])
            #     bad=False

    data = np.array(sample_label_s)
    data_size = len(sample_label_s)
    num_batches_per_epoch = int(data_size / batch_size) + 1  # num_of_step_per_epoch

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
            if len(shuffled_data[start_index:end_index]) != batch_size:
                continue
            # yield shuffled_data[start_index:end_index]
            batches.append(shuffled_data[start_index:end_index])
    return batches


def generate_pvdbow_batches(LabeledSentences, n_epochs, batch_size, window_size, shuffle=True):
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
    num_batches_per_epoch = int(data_size / batch_size) + 1
    batches = []
    for epoch in range(n_epochs):  # batch_iter
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))  # memory error here
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if len(shuffled_data[start_index:end_index]) != batch_size:
                continue
            # yield shuffled_data[start_index:end_index]
            batches.append(shuffled_data[start_index:end_index])
    return batches


def load_data_label(filepath):
    file = open(filepath, encoding='utf-8')
    lines = file.read().split('\n')
    file.close()
    texts, labels = [], []
    for line in lines:
        text = line.split('\t')[0]
        label = line.split('\t')[1]
        text = accept_sentence(text)
        text = jieba.lcut(''.join(text), HMM=False)
        texts.append(' '.join(text))
        if label == '中立':
            labels.append(0)
        elif label == '负面':
            labels.append(1)

    return texts, labels


def load_data(filepath, sample_file_path, data_size=350000):
    with open(sample_file_path, encoding='utf-8') as file:
        sample_count = 0
        for line in file:
            text = line.split('\t')[0].replace(' ', '').replace('\n', '')
            text = accept_sentence(text)
            text = jieba.lcut(''.join(text))
            yield ' '.join(text)
            sample_count += 1

    assert sample_count == 10
    # random_sampler for big file which does not fit in memory
    # generate random numer to sample data, size=350000 due to limit of machine
    # random_nums = set(random.sample(range(44001245), data_size))
    # need to remember the sample
    try:
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
            # print (' '.join(text))
            thefile.write(text + '\n')
            yield text
            # if i%50000==0:
            #     print (i)
    thefile.close()


def save_obj(obj, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def evaluate_analysis(x, y, y_pred, thresold, path):
    """
    :param x:
    :param y:
    :param y_pred:
    :param thresold:
    :param path: path of result out folder
    :return:
    """
    # assert y.shape == y_pred.shape
    logging.info('writing error analysis result to ' + path)
    from sklearn.metrics import classification_report
    y_predHat = [1 if i > thresold else 0 for i in y_pred]
    classification_report(y, y_predHat)
    f = open(path, 'w', encoding='utf-8')

    tmp = []
    f.write('ix\t' + 'real\t' + 'pred\t' + 'text\n')
    for ix in range(len(y_pred)):
        y_pred_single = 1 if y_pred[ix] > thresold  else 0
        if y_pred_single != y[ix]:  # write error first
            f.write(str(ix + 1) + '\t' + str(y[ix]) + '\t' +
                    str(y_pred[ix]) + '\t' +  str(x[ix]) + '\n')
        tmp.append(str(ix + 1) + '\t' + str(y[ix]) + '\t' +
                   str(y_pred[ix]) + '\t' + str(x[ix]) + '\n')
    f.write('-----------------------ALL RESULT------------------------')
    for text in tmp:
        f.write(text)

    f.close()
