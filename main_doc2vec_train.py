# -*- coding: utf-8 -*-
# @Author  : lzw
import logging
from datetime import datetime
import jieba
import utils
import os
import numpy as np
from sklearn.metrics import classification_report
from train_utils import train, get_embeddings, infer
logging.getLogger().setLevel(logging.INFO)
jieba.load_userdict('dataset/digital forum comments/digital_selfdict.txt')

train_filepath = 'dataset/digital forum comments/20171211_train.txt'
dev_filepath = 'dataset/digital forum comments/20171211_dev.txt'
logdir = 'logs/tmp/'

# parameters for building vocabulary
min_appear_freq = 1  # 1
k_mostcommon_ignore = 100  # consider stopwords here
# parameters for doc2vec model
model_type_pvdbow = 'pvdbow'
model_type_pvdm = 'pvdm'
batch_size = 256
n_epochs = 100  # big batches causes memory. # 200
eval_every_epochs = 20  # 10
# cal_sim_every_epochs = 30
tolerance = 10
eplison = 1e-4
learning_rate = 0.01
window_size = 5  # 5-12
num_neg_samples = 20  # 64
embedding_size_w = 50  # 200 300 400
embedding_size_d = 50  # 200 300 400

"""
train doc2vec model, pvdm and pvdbow, with train_filepath.
"""
model_info = '50d'
logging.info(str(datetime.now()).replace(':', '-') + '  start load_data.')
train_texts, train_labels = utils.load_data_label(train_filepath)
train_texts = list(train_texts)
document_size = len(train_texts)
logging.info(str(datetime.now()).replace(':', '-') + '  end load_data.')
word2count, word2index, index2word = utils.build_vocabulary(train_texts,
                                                            stop_words_file='dataset/stopwords.txt',
                                                            k=k_mostcommon_ignore,
                                                            min_appear_freq=min_appear_freq)
utils.save_obj(word2count, 'vocabulary/word2count' + '_v1.pkl')
utils.save_obj(word2index, 'vocabulary/word2index' + '_v1.pkl')
utils.save_obj(index2word, 'vocabulary/index2word' + '_v1.pkl')

logging.info(str(datetime.now()).replace(':', '-') + '  start generate_texts2indexes.')
train_data = utils.generate_texts2indexes(input_data=train_texts,
                                          word2index=word2index)
logging.info(str(datetime.now()).replace(':', '-') + '  end generate_texts2indexes.')

# generate train batches
train_batches = utils.generate_pvdbow_batches(batch_size=batch_size,
                                              window_size=window_size,
                                              LabeledSentences=train_data,
                                              n_epochs=n_epochs)
logging.info(str(datetime.now()).replace(':', '-') + '  Train pvdbow start - ')
train(model_type=model_type_pvdbow,
      word2index=word2index,
      index2word=index2word,
      train_texts=train_texts,
      train_batches=train_batches,
      vocabulary_size=len(word2index),
      document_size=document_size,
      embedding_size_w=embedding_size_w,
      embedding_size_d=embedding_size_d,
      batch_size=batch_size,
      n_epochs=n_epochs,
      eval_every_epochs=eval_every_epochs,
      tolerance=tolerance,
      learning_rate=learning_rate,
      num_neg_samples=num_neg_samples,
      model_info=model_info,
      eplison=eplison)

logging.info(str(datetime.now()).replace(':', '-') + '  Train pvdbow finished - ')
train_batches = utils.generate_pvdm_batches(batch_size=batch_size,
                                            window_size=window_size,
                                            LabeledSentences=train_data,
                                            n_epochs=n_epochs)

logging.info(str(datetime.now()).replace(':', '-') + '  Train pvdm start - ')
train(model_type=model_type_pvdm,
      word2index=word2index,
      index2word=index2word,
      train_texts=train_texts,
      train_batches=train_batches,
      vocabulary_size=len(word2index),
      document_size=document_size,
      embedding_size_w=embedding_size_w,
      embedding_size_d=embedding_size_d,
      batch_size=batch_size,
      eval_every_epochs=eval_every_epochs,
      tolerance=tolerance,
      learning_rate=learning_rate,
      num_neg_samples=num_neg_samples,
      eplison=eplison,
      model_info=model_info,
      n_epochs=n_epochs)

logging.info(str(datetime.now()).replace(':', '-') + '  Train pvdm finished - ')
