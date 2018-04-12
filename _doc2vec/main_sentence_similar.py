# -*- coding: utf-8 -*-
# @Author  : lzw
import logging
from datetime import datetime

import numpy as np

import data_helper
from train_utils import get_embeddings

"""
    First thought to expect high similarity between short meaningful comment with a same-meaning long comment.
    but seems can't reach my goal.
    main fun: Given a doc id, return the k most similar doc ids. 
"""


def k_most_common(matrix, id, k=10):
    """
    calculate similarity between a sentence and all corpus, output k most similar.
    :param matrix:
    :param id:
    :return:
    """
    doc_embedding_id = matrix[id:id + 1, :]
    doc_embeddings_comments = np.concatenate((matrix[:id, :], matrix[id + 1:, :]), axis=0)

    matrix = np.dot(doc_embeddings_comments, doc_embedding_id.T)
    norm_all = np.linalg.norm(doc_embeddings_comments, axis=1, keepdims=True)
    norm_top10 = np.linalg.norm(doc_embedding_id, axis=1, keepdims=True)
    norm_matrix_row = np.dot(norm_all, norm_top10.T)  # i列j行表示，第i(69138)个句子与第j(10)个句子之间各自norm相乘

    cos = matrix / norm_matrix_row
    cos = 0.5 + 0.5 * cos  # 归一化 normalized
    sim_matric_argmax_ix = np.argsort(cos, axis=0)[::-1][:k]
    # print(cos[sim_matric_argmax_ix])
    return sim_matric_argmax_ix, cos[sim_matric_argmax_ix]


k_mostcommon_ignore = 100
min_appear_freq = 1
logging.info(str(datetime.now()).replace(':', '-') + '  start load_data.')
train_filepath = 'dataset/digital forum comments/20171211_train.txt'
train_texts, _ = data_helper.load_data_label(train_filepath)
document_size = len(train_texts)
logging.info(str(datetime.now()).replace(':', '-') + '  end load_data.')
#

word2index_path = 'vocabulary/word2index' + '_v1_min_count_1.pkl'
index2word_path = 'vocabulary/index2word' + '_v1_min_count_1.pkl'
word2index = data_helper.load_obj(word2index_path)
index2word = data_helper.load_obj(index2word_path)
trained_model_checkpoint_dir_pvdbow = 'model/trained_model_pvdbow_50d/checkpoints'
trained_model_checkpoint_dir_pvdm = 'model/trained_model_pvdm_50d/checkpoints'
logging.info('[LR] get embedding from a retrained model \n\t{}, {}. '.format(trained_model_checkpoint_dir_pvdm,
                                                                             trained_model_checkpoint_dir_pvdbow))
logging.info('[LR] Get embeddings from retrained.')

word_embeddings_dm, doc_embeddings_dm = get_embeddings(checkpoint_dir=trained_model_checkpoint_dir_pvdm,
                                                       model_type='pvdm')
word_embeddings_None, doc_embeddings_dbow = get_embeddings(checkpoint_dir=trained_model_checkpoint_dir_pvdbow,
                                                           model_type='pvdbow')

test_embeddings = np.concatenate((doc_embeddings_dm, doc_embeddings_dbow),
                                 axis=1)  # dm vec in the front

"""
    specific doc id here
"""
doc_id = 52898
doc_id_text = train_texts[doc_id]

sim_matric_argmax_ix, sim_matric_argmax_nums = k_most_common(test_embeddings, doc_id)
logging.info('-' * 100)
logging.info(
    'target sentence: \n'
    + '\t\t' + train_texts[doc_id] + '\n'
    + '\t\t' + ' '.join(data_helper.ids2words(index2word, data_helper.words2ids(word2index, train_texts[doc_id].split(
        ' ')))))

for sim_t, sim_num in zip(sim_matric_argmax_ix, sim_matric_argmax_nums):
    sim_t = int(sim_t)
    logging.info('-' * 100 + '\n'
                 + '\t\t' + str(float(sim_num)) + '\n'
                 + '\t\t' + train_texts[sim_t] + '\n'
                 + '\t\t' + ' '.join(data_helper.ids2words(index2word,
                                                           data_helper.words2ids(word2index, train_texts[sim_t].split(' ')))))
logging.info('-' * 100)
