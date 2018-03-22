# -*- coding: utf-8 -*-
# @Author  : lzw
import logging
import os
from datetime import datetime

import jieba
import numpy as np

import utils
from lr_model import lr_model
from train_utils import train, get_embeddings, infer

# import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)
jieba.load_userdict('dataset/digital_selfdict.txt')

train_filepath = 'dataset/20171211_train.txt'
dev_filepath = 'dataset/20171211_dev.txt'
test_filepath = 'dataset/测试集3_联想_杨欣标注15000_20171116 (2).txt'
logdir = 'logs/tmp/'
# 建立词库
min_appear_freq = 2
k_mostcommon_ignore = 100  # 这里考虑停用词

# 训练模型
model_type_pvdbow = 'pvdbow'
model_type_pvdm = 'pvdm'
batch_size = 256
n_epochs = 100  # big batches causes memory.
eval_every_epochs = 20
# cal_sim_every_epochs = 30
tolerance = 10
eplison = 1e-4
learning_rate = 0.01
window_size = 5  # 5-12
embedding_size_w = 100
embedding_size_d = 100

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
logging.critical(str(datetime.now()).replace(':', '-') + '  start load_data.')
# train_texts = utils.load_data(filepath='dataset/sentence_20180318_without_stopwords.txt',
#                               sample_file_path='dataset/sample_sentences.txt',
#                               data_size=100000) # 175000
train_texts, train_labels = utils.load_data_label(train_filepath)

# from itertools import tee
try:
    train_texts = list(train_texts)
    # tee_gen = [train_texts,train_texts]/
except:
    logging.critical('create list fail. ')
    exit()
    # from itertools import tee
    # tee_gen       = tee(train_texts, 3)  # 复制生成器
    # document_size = sum(1 for i in tee_gen[2])

document_size = len(train_texts)
logging.critical(str(datetime.now()).replace(':', '-') + '  end load_data.')
word2count, word2index, index2word = utils.build_vocabulary(train_texts,
                                                            stop_words_file='dataset/stopwords.txt',
                                                            k=k_mostcommon_ignore,
                                                            min_appear_freq=min_appear_freq)
utils.save_obj(word2count, 'vocabulary/word2count' + '_v1.pkl')
utils.save_obj(word2index, 'vocabulary/word2index' + '_v1.pkl')
utils.save_obj(index2word, 'vocabulary/index2word' + '_v1.pkl')

logging.critical(str(datetime.now()).replace(':', '-') + '  start generate_texts2indexes.')
train_data = utils.generate_texts2indexes(input_data=train_texts,
                                          word2index=word2index)
# tee_gen_train      = tee(train_data, 2)  # 复制生成器
logging.critical(str(datetime.now()).replace(':', '-') + '  end generate_texts2indexes.')

train_already = True
if not train_already:

    train_batches = utils.generate_pvdbow_batches(batch_size=batch_size,
                                                  window_size=window_size,
                                                  LabeledSentences=train_data,
                                                  n_epochs=n_epochs)
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
          eplison=eplison)

    logging.critical(str(datetime.now()).replace(':', '-') + '  Train pvdbow finished - ')
    train_batches = utils.generate_pvdm_batches(batch_size=batch_size,
                                                window_size=window_size,
                                                LabeledSentences=train_data,
                                                n_epochs=n_epochs)
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
          eplison=eplison,
          n_epochs=n_epochs)

    logging.critical(str(datetime.now()).replace(':', '-') + '  Train pvdm finished - ')
else:
    """
    Get embedding, and using it to do sentiment classify
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    word2index_path = 'vocabulary/word2index' + '_v1.pkl'
    index2word_path = 'vocabulary/index2word' + '_v1.pkl'
    word2index = utils.load_obj(word2index_path)
    id2word = utils.load_obj(index2word_path)

    logging.critical('Get embeddings.')
    trained_model_checkpoint_dir_pvdbow = 'model/trained_model_pvdbow_2018-03-22 05-22-01.830643/checkpoints'
    trained_model_checkpoint_dir_pvdm = 'model/trained_model_pvdm_2018-03-22 06-21-33.759069/checkpoints'

    _, doc_embeddings_dbow = get_embeddings(checkpoint_dir=trained_model_checkpoint_dir_pvdbow,
                                            model_type=model_type_pvdbow)

    word_embeddings_dm, doc_embeddings_dm = get_embeddings(checkpoint_dir=trained_model_checkpoint_dir_pvdm,
                                                           model_type=model_type_pvdm)
    embedding = np.concatenate((doc_embeddings_dbow, doc_embeddings_dm), axis=1)

    """
    训练 logistic regression
    """
    train_texts, train_labels = utils.load_data_label(train_filepath)
    new_LabeledSentences_train = utils.generate_texts2indexes(input_data=train_texts,
                                                              word2index=word2index)
    logging.critical('Num of train sentences: {}'.format(len(train_texts)))

    assert embedding.shape[0] == len(train_labels)

    trained_model_checkpoint_dir_pvdbow = 'model/trained_model_pvdbow_2018-03-22 05-22-01.830643/checkpoints'
    trained_model_checkpoint_dir_pvdm = 'model/trained_model_pvdm_2018-03-22 06-21-33.759069/checkpoints'

    word_embeddings_None, doc_embeddings_dbow = get_embeddings(checkpoint_dir=trained_model_checkpoint_dir_pvdbow,
                                                               model_type='pvdbow')

    word_embeddings_dm, doc_embeddings_dm = get_embeddings(checkpoint_dir=trained_model_checkpoint_dir_pvdm,
                                                           model_type='pvdm')
    embedding = np.concatenate((doc_embeddings_dbow, doc_embeddings_dm), axis=1)

    # compare with sklearn logistic optimazation. looks like my logistic regression still not perfect...
    # from sklearn import linear_model
    # lr = linear_model.SGDClassifier(max_iter=500, verbose=1, tol=100, epsilon=0.01, learning_rate='constant', eta0=0.00001)
    # lr.fit(embedding, train_labels)
    #
    # classifedLabels = lr.predict(embedding)
    lr = lr_model(epsilon=1e-5,
                  llambda=0.01,
                  alpha=0.1,
                  num_iters=8000,
                  batch_size=embedding.shape[0],  # embedding.shape[0]
                  tolerance=5,
                  normalization='l2')
    learntParameters, final_costs = lr.train(embedding, train_labels, np.unique(train_labels))

    """
    评估 logistic regression 在训练集
    """
    classifedLabels = lr.classify(embedding)
    classifedProbs, classifedLabels = zip(*classifedLabels)
    classifedProbs = np.array(classifedProbs).flatten()
    classifedLabels = np.array(classifedLabels).flatten()
    # print (train_labels)
    # print (classifedProbs)
    logging.critical('[[ LR ]] Accuracy on training data: ', (np.sum(classifedLabels == train_labels) / len(train_labels)) * 100, '%')

    ################################################################################################################
    """
    评估 logistic regression 在测试集
    """
    logging.critical('Evaluate lr model which training by doc2vec using dataset: {}'.format(test_filepath))
    test_texts, test_labels = utils.load_data_label(test_filepath)
    new_LabeledSentences_test = utils.generate_texts2indexes(input_data=test_texts,
                                                             word2index=word2index)
    logging.critical('Num of test sentences: {}'.format(len(test_texts)))

    test_embeddings = infer(checkpoint_dir_pvdm=trained_model_checkpoint_dir_pvdm,
                            checkpoint_dir_pvdbow=trained_model_checkpoint_dir_pvdbow,
                            new_LabeledSentences_train=new_LabeledSentences_test,
                            train_texts=test_texts,
                            window_size=window_size,
                            embedding_size_w=embedding_size_w,
                            embedding_size_d=embedding_size_d,
                            word2index=word2index,
                            index2word=index2word,
                            batch_size=batch_size,
                            vocabulary_size=len(word2index),
                            document_size=len(new_LabeledSentences_test),
                            eval_every_epochs=eval_every_epochs,
                            tolerance=tolerance,
                            learning_rate=learning_rate,
                            eplison=eplison,
                            n_epochs=n_epochs,
                            logdir=None)

    assert test_embeddings.shape[0] == len(test_labels)
    classifedLabels = lr.classify(test_embeddings)
    classifedProbs, classifedLabels = zip(*classifedLabels)
    # classifedProbs = np.array(classifedProbs).flatten()
    classifedLabels = np.array(classifedLabels).flatten()

    print('Accuracy on test data: ', (np.sum(classifedLabels == test_labels) / len(test_labels)) * 100, '%')
