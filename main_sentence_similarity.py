# -*- coding: utf-8 -*-
# @Author  : lzw
import numpy as np

import utils
import logging
from datetime import datetime
import jieba
import os
from train_utils import get_embeddings

def calSimilarity(matrix, ix_a):
    doc_embeddings_tags = matrix[ix_a:, :]
    doc_embeddings_comments = matrix[:ix_a, :]

    matrix = np.dot(doc_embeddings_comments, doc_embeddings_tags.T)
    norm_all = np.linalg.norm(doc_embeddings_comments, axis=1, keepdims=True)
    norm_top10 = np.linalg.norm(doc_embeddings_tags, axis=1, keepdims=True)
    norm_matrix_row = np.dot(norm_all, norm_top10.T)  # i列j行表示，第i(69138)个句子与第j(10)个句子之间各自norm相乘

    cos = matrix / norm_matrix_row
    cos = 0.5 + 0.5 * cos  # 归一化
    sim_matric_argmax_ix = np.argmax(cos, 0)

    return sim_matric_argmax_ix

jieba.load_userdict('dataset/car forum comments/carSelfDict.txt')

k_mostcommon_ignore=100
min_appear_freq=1

logging.info(str(datetime.now()).replace(':', '-') + '  start load_data.')
comments_file=open('dataset/car forum comments/comments.txt', encoding='utf-8')
comments=comments_file.read().split('\n')
comments_file.close()

tags_file=open('dataset/car forum comments/tags.txt', encoding='utf-8')
tags=tags_file.read().split('\n')
tags_file.close()
train_texts=comments[:10000]+tags
tags_start_ix = 10000


for i, text in enumerate(train_texts):
  train_texts[i] = ''.join(utils.accept_sentence(train_texts[i])).replace(' ', '').replace('\t','')
  train_texts[i] = ' '.join(jieba.lcut(train_texts[i], cut_all=False))

document_size = len(train_texts)
logging.info(str(datetime.now()).replace(':', '-') + '  end load_data.')
word2count, word2index, index2word = utils.build_vocabulary(train_texts,
                                                            stop_words_file='dataset/stopwords.txt',
                                                            k=k_mostcommon_ignore,
                                                            min_appear_freq=min_appear_freq)

if not os.path.exists('vocabulary/'):
  os.makedirs('vocabulary/')
utils.save_obj(word2count, 'vocabulary/word2count' + '_Vcar.pkl')
utils.save_obj(word2index, 'vocabulary/word2index' + '_Vcar.pkl')
utils.save_obj(index2word, 'vocabulary/index2word' + '_Vcar.pkl')
# get embeddings and specific index

retrained_model_pvdbow_path = 'dataset/car forum comments/trained_model_pvdbow_2018-03-25 11-47-31.693103/checkpoints'
retrained_model_pvdm_path = 'dataset/car forum comments/trained_model_pvdm_2018-03-25 11-53-06.646865/checkpoints'

logging.info('[LR] get embedding from a retrained model \n\t{}, {}. '.format(retrained_model_pvdm_path,
                                                                             retrained_model_pvdbow_path))
logging.info('[LR] Get embeddings from retrained.')

word_embeddings_dm, doc_embeddings_dm = get_embeddings(checkpoint_dir=retrained_model_pvdm_path,
                                                       model_type='pvdm')
word_embeddings_None, doc_embeddings_dbow = get_embeddings(checkpoint_dir=retrained_model_pvdbow_path,
                                                           model_type='pvdbow')

test_embeddings = np.concatenate((doc_embeddings_dm, doc_embeddings_dbow),
                                         axis=1)  # dm vec in the front

sim_matric_argmax_ix = calSimilarity(test_embeddings, tags_start_ix)  # 计算相似度，LabeledSentences_train前十个句子
tags_ix = range(tags_start_ix, len(train_texts))
logging.info('-' * 80)
i = 1
for t, sim_t in zip(tags_ix, sim_matric_argmax_ix):
    logging.info('第{}句话: '.format(i))
    logging.info(
        train_texts[t] + '\n'
        + ' \t\t'
        + ' '.join(utils.ids2words(index2word, utils.words2ids(word2index, train_texts[t].split(
            ' ')))) + '\n'
        + ' \t\t'
        + train_texts[sim_t] + '\n'
        + ' \t\t'
        + ' '.join(utils.ids2words(index2word,
                                   utils.words2ids(word2index, train_texts[sim_t].split(' ')))))
    i += 1
logging.info('-' * 80)

