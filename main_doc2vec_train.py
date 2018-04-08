# -*- coding: utf-8 -*-
# @Author  : lzw
import logging
from datetime import datetime
import jieba
import data_helper
import os
import numpy as np
from train_utils import train
logging.getLogger().setLevel(logging.INFO)

"""
    train doc2vec model, pvdm and pvdbow, with train_filepath.
    1. dbow performs better than dm. (Jey Han Lau, IBM Research 2016), 
       what's more, using a pretrained word2vec in dbow improves model.

"""
jieba.load_userdict('dataset/digital forum comments/digital_selfdict.txt')
train_filepath = 'dataset/digital forum comments/20171211_train.txt'
logdir = 'logs/tmp/'

# parameters for building vocabulary
min_appear_freq = 1  # 1
k_mostcommon_ignore = 100  # consider stopwords here
# parameters for doc2vec model
model_type_pvdbow = 'pvdbow'
model_type_pvdm = 'pvdm'
batch_size = 256
n_epochs = 100  # big batches causes memory. # 200
eval_every_epochs = 10  # 10
# cal_sim_every_epochs = 30
tolerance = 10
eplison = 1e-4
learning_rate = 0.01
window_size = 5  # 5-12
num_neg_samples = 64  # 64
embedding_size_w = 100  # 200 300 400
embedding_size_d = 100  # 200 300 400
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_info = 'pre300d-100d'
logging.info(str(datetime.now()).replace(':', '-') + '  start load_data.')
train_texts, train_labels = data_helper.load_data_label(train_filepath)
train_texts = list(train_texts)
document_size = len(train_texts)
logging.info(str(datetime.now()).replace(':', '-') + '  end load_data.')


"""
    using a corpus or a built vocabulary.
"""
use_pretrained_word_embeddings=True
if use_pretrained_word_embeddings:
    pretrained_word_embeddings_path = 'vocabulary/word2vec_dict_pretrained_cbow_100.pkl'
    logging.info("using pre-trained word embeddings, word2vec_model_path: {}".format(pretrained_word_embeddings_path))
    word2vec_dict=data_helper.load_obj(pretrained_word_embeddings_path)
    word2index = {'_UNK_': 0, '_NULL_': 1}
    for i, k_v in enumerate(word2vec_dict.items()):
        word2index[k_v[0]] = i + 2
    index2word = dict(zip(word2index.values(), word2index.keys()))

else:
    word2count, word2index, index2word = data_helper.build_vocabulary(train_texts,
                                                                stop_words_file='dataset/stopwords.txt',
                                                                k=k_mostcommon_ignore,
                                                                min_appear_freq=min_appear_freq)
    word2vec_dict=None
    pretrained_word_embeddings_path = None

logging.info(str(datetime.now()).replace(':', '-') + '  start generate_texts2indexes.')
train_data = data_helper.generate_texts2indexes(input_data=train_texts,
                                          word2index=word2index)
logging.info(str(datetime.now()).replace(':', '-') + '  end generate_texts2indexes.')

# generate train batches
train_batches = data_helper.generate_pvdm_batches(batch_size=batch_size,
                                            window_size=window_size,
                                            LabeledSentences=train_data,
                                            n_epochs=n_epochs)

logging.info(str(datetime.now()).replace(':', '-') + '  Train pvdm start - ')
train(model_type=model_type_pvdm,
      word2index=word2index,
      index2word=index2word,
      word2vec_dict=word2vec_dict,
      train_texts=train_texts,
      train_batches=train_batches,
      vocabulary_size=len(word2index),
      document_size=document_size,
      embedding_size_w=embedding_size_w,
      embedding_size_d=embedding_size_d,
      use_pretrained_word_embeddings=use_pretrained_word_embeddings,
      batch_size=batch_size,
      eval_every_epochs=eval_every_epochs,
      tolerance=tolerance,
      learning_rate=learning_rate,
      num_neg_samples=num_neg_samples,
      eplison=eplison,
      model_info=model_info,
      n_epochs=n_epochs)

logging.info(str(datetime.now()).replace(':', '-') + '  Train pvdm finished - ')

# train_batches = utils.generate_pvdbow_batches(batch_size=batch_size,
#                                               window_size=window_size,
#                                               LabeledSentences=train_data,
#                                               n_epochs=n_epochs)
# logging.info(str(datetime.now()).replace(':', '-') + '  Train pvdbow start - ')
# train(model_type=model_type_pvdbow,
#       word2index=word2index,
#       index2word=index2word,
#       train_texts=train_texts,
#       train_batches=train_batches,
#       vocabulary_size=len(word2index),
#       document_size=document_size,
#       embedding_size_w=embedding_size_w,
#       embedding_size_d=embedding_size_d,
#       use_pretrained_word_embeddings=use_pretrained_word_embeddings,
#       pretrained_word_embeddings_path=pretrained_word_embeddings_path,
#       batch_size=batch_size,
#       n_epochs=n_epochs,
#       eval_every_epochs=eval_every_epochs,
#       tolerance=tolerance,
#       learning_rate=learning_rate,
#       num_neg_samples=num_neg_samples,
#       model_info=model_info,
#       eplison=eplison)
#
# logging.info(str(datetime.now()).replace(':', '-') + '  Train pvdbow finished - ')