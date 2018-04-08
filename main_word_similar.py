# -*- coding: utf-8 -*-
# @Author  : lzw
import logging
from datetime import datetime

import numpy as np

import data_helper
from train_utils import get_embeddings

"""
"""

def k_most_common(matrix, id, k=10):
    """
    calculate similarity between a word and vocabulary, output k most similar.
    :param matrix:
    :param id:
    :return:
    """
    doc_embedding_id = matrix[id:id + 1, :]
    doc_embeddings_comments = np.concatenate((matrix[:id, :], matrix[id + 1:, :]), axis=0)

    matrix = np.dot(doc_embeddings_comments, doc_embedding_id.T)
    norm_all = np.linalg.norm(doc_embeddings_comments, axis=1, keepdims=True)
    norm_top10 = np.linalg.norm(doc_embedding_id, axis=1, keepdims=True)
    norm_matrix_row = np.dot(norm_all, norm_top10.T)

    cos = matrix / norm_matrix_row
    cos = 0.5 + 0.5 * cos  # 归一化 normalized
    sim_matric_argmax_ix = np.argsort(cos, axis=0)[::-1][:k]
    return sim_matric_argmax_ix, cos[sim_matric_argmax_ix]


k_mostcommon_ignore = 100
min_appear_freq = 1
#
word2index_path = 'vocabulary/word2index' + '_v1.pkl'
index2word_path = 'vocabulary/index2word' + '_v1.pkl'

word2index = data_helper.load_obj(word2index_path)
index2word = data_helper.load_obj(index2word_path)
trained_model_checkpoint_dir_pvdm = 'model/trained_model_pvdm_50d/checkpoints'
logging.info('[word similar] get embedding from a trained model \n\t{}. '.format(trained_model_checkpoint_dir_pvdm,
                                                                             ))
word_embeddings_dm, doc_embeddings_dm = get_embeddings(checkpoint_dir=trained_model_checkpoint_dir_pvdm,
                                                       model_type='pvdm')

"""
    specific word id.
"""
word = '白屏'
word_id = word2index[word]
word_id_text = index2word[word_id]

sim_matric_argmax_ix, sim_matric_argmax_nums = k_most_common(word_embeddings_dm, word_id)
logging.info('-' * 100)
logging.info(
    'target word: \n' + '\t\t' + index2word[word_id] + '\n')

for sim_t, sim_num in zip(sim_matric_argmax_ix, sim_matric_argmax_nums):
    sim_t = int(sim_t)
    logging.info('-' * 100 + '\n'
                 + '\t\t' + str(float(sim_num)) + '\n'
                 + '\t\t' + index2word[sim_t] + '\n')
logging.info('-' * 100)
