# -*- coding: utf-8 -*-
# @Author  : lzw
import logging
import os

import jieba
import numpy as np
from sklearn.metrics import classification_report

import utils
from lr_model import lr_model
from train_utils import get_embeddings, infer, evaluate_analysis

logging.getLogger().setLevel(logging.INFO)
jieba.load_userdict('dataset/digital forum comments/digital_selfdict.txt')

train_filepath = 'dataset/digital forum comments/20171211_train.txt'
dev_filepath = 'dataset/digital forum comments/20171211_dev.txt'
logdir = 'logs/tmp/'

# parameters for building vocabulary
min_appear_freq = 2  # 1
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
# embedding_size_w = 100  # 200 300 400
# embedding_size_d = 100  # 200 300 400

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
word2index_path = 'vocabulary/word2index' + '_v1_min_count_1.pkl'
index2word_path = 'vocabulary/index2word' + '_v1_min_count_1.pkl'
word2index = utils.load_obj(word2index_path)
index2word = utils.load_obj(index2word_path)

"""
    given a doc2vec model, 
    different: reinfer train text with another set, using the re-inferred doc embeddings to train classifier. 
    does infer train and test texts together/separate matter? 
    first thought a separate one will be welcome, for effciency consideration (incrementing docs).
"""
train_texts, train_labels = utils.load_data_label(train_filepath)
# oversampling using a traditional language model bigram or another dataset
texts_file = open('dataset/digital forum comments/20171211_dev.txt', encoding='utf-8')
texts = texts_file.read().split('\n')
dev_texts = []
for i in texts:  # load ovs
    t, l = i.split('\t')
    if l == '负面':
        dev_texts.append(' '.join(jieba.lcut(t.strip())))

ovs_texts = dev_texts
inp = train_texts + ovs_texts
inp_len = len(train_labels + [1] * len(ovs_texts))
new_LabeledSentences_ovs = utils.generate_texts2indexes(input_data=inp[:],
                                                        word2index=word2index)
use_exist_retrained_model = True
# reinfer together
model_info = '300d_reinferTrain'
if not use_exist_retrained_model:
    trained_model_checkpoint_dir_pvdbow = 'model/trained_model_pvdbow_2018-03-24 01-49-16.072438_300/checkpoints'
    trained_model_checkpoint_dir_pvdm = 'model/trained_model_pvdm_2018-03-24 03-18-25.909381_300/checkpoints'
    infer_lr = 3.0  # lr need to be slightly big for convergence 0.1-0.5, but here 3.0 is not big.
    infer_eplison = 1e-5
    infer_num_neg_samples = 20
    infer_tolerance = 5
    infer_embedding_size_w = 300
    infer_embedding_size_d = 300
    embeddings = infer(checkpoint_dir_pvdm=trained_model_checkpoint_dir_pvdm,
                       checkpoint_dir_pvdbow=trained_model_checkpoint_dir_pvdbow,
                       infer_LabeledSentences=new_LabeledSentences_ovs,
                       trained_texts=train_texts,
                       infer_texts=ovs_texts,
                       window_size=window_size,
                       embedding_size_w=infer_embedding_size_w,
                       embedding_size_d=infer_embedding_size_d,
                       word2index=word2index,
                       index2word=index2word,
                       batch_size=batch_size,
                       vocabulary_size=len(word2index),
                       document_size=len(new_LabeledSentences_ovs),
                       eval_every_epochs=eval_every_epochs,
                       tolerance=infer_tolerance,
                       learning_rate=infer_lr,
                       num_neg_samples=infer_num_neg_samples,
                       eplison=infer_eplison,
                       model_info=model_info,
                       n_epochs=n_epochs,
                       logdir=None)

else:
    # retrained_model_checkpoint_dir_pvdbow = 'model/retrained_infer_model_pvdbow_200d_reinferTrain/checkpoints'
    # retrained_model_checkpoint_dir_pvdm = 'model/retrained_infer_model_pvdm_200d_reinferTrain/checkpoints'

    retrained_model_checkpoint_dir_pvdbow = 'model/retrained_infer_model_pvdbow_' + model_info + '/checkpoints'
    retrained_model_checkpoint_dir_pvdm = 'model/retrained_infer_model_pvdm_' + model_info + '/checkpoints'

    word_embeddings_dm, doc_embeddings_dm = get_embeddings(checkpoint_dir=retrained_model_checkpoint_dir_pvdm,
                                                           model_type='pvdm')
    word_embeddings_None, doc_embeddings_dbow = get_embeddings(checkpoint_dir=retrained_model_checkpoint_dir_pvdbow,
                                                               model_type='pvdbow')
    embeddings = np.concatenate((doc_embeddings_dm[-inp_len:], doc_embeddings_dbow[-inp_len:]),
                                axis=1)  # dm vec in the front
    assert embeddings.shape[0] == inp_len, '' + str(embeddings.shape[0]) + ', ' + str(inp_len)

X = embeddings
Y = train_labels + [1] * len(ovs_texts)
# Y=train_labels[:]
"""
    specific classifier here
"""
from sklearn import linear_model
clf = linear_model.SGDClassifier(max_iter=3000, verbose=1, tol=20, learning_rate='constant', eta0=0.01,
                                 class_weight={1: 3, 0: 1})
# X, Y = utils.smote_to_go(X, Y)
clf.fit(X, Y)

# using gbdt handling high dimension data, for clf seems to can't handle the features for sentences generate by doc2vec
# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier(n_estimators=80,
#                                  max_leaf_nodes=30,
#                                  subsample=0.5,
#                                  learning_rate=0.1,
#                                  loss='deviance',
#                                  verbose=1,
#                                  max_depth=3,
#                                  max_features=10, random_state=10)
# clf.fit(X, Y)

"""
    logistic regression on train set
"""
path = 'lr_models/'
train_error_analysis_file = 'train_error_analysis_300_reinferTrain_gbdt.txt'
test_error_analysis_file = 'test_error_analysis_300_reinferTrain_gbdt.txt'
classifedLabels = clf.predict(X)
# classifedLabels = clf.classify(X)
# classifedProbs, classifedLabels = zip(*classifedLabels)
classifedLabels = np.array(classifedLabels).flatten()

evaluate_analysis(train_texts + ovs_texts, Y, classifedLabels, 0.5,
                        path + train_error_analysis_file)
logging.info('[clf] \n' + str(classification_report(Y, classifedLabels)))
logging.info('[clf] Accuracy on training data: {} %'.format(
    np.sum(classifedLabels == Y) / len(Y) * 100))

################################################################################################################
"""
    evaluate logistic regression on test set (test text from different months of text.)
"""
# test_filepath_0 = 'dataset/digital forum comments/20171211_dev.txt'  # 测试集3_联想_杨欣标注15000_20171116.txt
test_filepath_0 = 'dataset/digital forum comments/测试集3_联想_杨欣标注15000_20171116.txt'
test_texts, test_labels = utils.load_data_label(test_filepath_0)
test_texts_len = len(test_texts)
logging.info('[clf] Evaluate clf model which training by doc2vec using dataset: {}'.format(test_filepath_0))
logging.info('[clf] Num of test sentences: {}'.format(test_texts_len))

use_exist_inferred_model = True
test_model_info = '300d_test'
if not use_exist_inferred_model:
    """
    use the reinferred train texts model to infer test texts
    """
    trained_model_checkpoint_dir_pvdbow = 'model/retrained_infer_model_pvdbow_' + model_info + '/checkpoints'
    trained_model_checkpoint_dir_pvdm = 'model/retrained_infer_model_pvdm_' + model_info + '/checkpoints'

    new_LabeledSentences_test = utils.generate_texts2indexes(input_data=test_texts,
                                                             word2index=word2index)
    # parameters for inferring vector (retrained a model using specific models' weights, bias, embeddings.)
    infer_lr = 3.0  # lr need to be slightly big for convergence 0.1-0.5, but here 3.0 is not big.
    infer_eplison = 1e-5
    infer_num_neg_samples = 64
    infer_tolerance = 5
    infer_embedding_size_w = 300
    infer_embedding_size_d = 300

    test_embeddings = infer(checkpoint_dir_pvdm=trained_model_checkpoint_dir_pvdm,
                            checkpoint_dir_pvdbow=trained_model_checkpoint_dir_pvdbow,
                            infer_LabeledSentences=new_LabeledSentences_test,
                            trained_texts=train_texts + train_texts + ovs_texts,  # for reinfer train texts
                            infer_texts=test_texts,
                            window_size=window_size,
                            embedding_size_w=infer_embedding_size_w,
                            embedding_size_d=infer_embedding_size_d,
                            word2index=word2index,
                            index2word=index2word,
                            batch_size=batch_size,
                            vocabulary_size=len(word2index),
                            document_size=len(new_LabeledSentences_test),
                            eval_every_epochs=eval_every_epochs,
                            tolerance=infer_tolerance,
                            learning_rate=infer_lr,
                            num_neg_samples=infer_num_neg_samples,
                            eplison=infer_eplison,
                            n_epochs=n_epochs,
                            model_info=test_model_info,
                            logdir=None)

else:
    """
        load a already inferred model of test texts
    """
    retrained_infer_model_test_checkpoint_dir_pvdbow = 'model/retrained_infer_model_pvdbow_' + test_model_info + '/checkpoints'
    retrained_infer_model_test_checkpoint_dir_pvdm = 'model/retrained_infer_model_pvdm_' + test_model_info + '/checkpoints'
    word_embeddings_dm, doc_embeddings_dm = get_embeddings(
        checkpoint_dir=retrained_infer_model_test_checkpoint_dir_pvdm,
        model_type='pvdm')
    word_embeddings_None, doc_embeddings_dbow = get_embeddings(
        checkpoint_dir=retrained_infer_model_test_checkpoint_dir_pvdbow,
        model_type='pvdbow')
    test_embeddings = np.concatenate((doc_embeddings_dm[-test_texts_len:], doc_embeddings_dbow[-test_texts_len:]),
                                     axis=1)  # dm vec in the front

assert test_embeddings.shape[0] == test_texts_len, '' + str(test_embeddings.shape[0]) + ', ' + str(test_texts_len)
classifedLabels = clf.predict(test_embeddings)
# classifedLabels = clf.classify(test_embeddings)
# classifedProbs, classifedLabels = zip(*classifedLabels)
classifedLabels = np.array(classifedLabels).flatten()
evaluate_analysis(test_texts, test_labels, classifedLabels, 0.5, path + test_error_analysis_file)
logging.info('[clf] \n' + str(classification_report(test_labels, classifedLabels)))
logging.info(
    '[clf] Accuracy on test data: {} %'.format((np.sum(classifedLabels == test_labels) / len(test_labels)) * 100))
