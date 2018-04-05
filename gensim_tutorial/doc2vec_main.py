# -*- coding: utf-8 -*-
# @Author  : lzw
# description: using gensim doc2vec to handle text classification( doc2vec as text representation )
# reference : tutorial on gensim official (https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb)
# trick: 1. using iterator to handle large file problem;
#        2. use an external metric to evaluate doc2vec embedding.(sentiment classification)

import datetime
import multiprocessing
from collections import OrderedDict
from collections import defaultdict
from random import shuffle

import jieba
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn import linear_model
from sklearn.metrics import classification_report

from gensim_t.data_helper import load_text_label
from gensim_t.train_utils import elapsed_timer

jieba.load_userdict('../dataset/digital forum comments/digital_selfdict.txt')
# logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
cores = multiprocessing.cpu_count()

corpus_path = '../dataset/digital forum comments/split_20180403/classes/'
# path = '../dataset/digital forum comments/split_20180403/classes/'
classes_num = {'0': 84145, '1': 22057}  # get these nums first
split = [0.8, 0.2]
corpus, train_len_total, dev_len_total = load_text_label(corpus_path, classes_num, split, shuffle=True)
corpus_train = [d for d in corpus if d.split == 'train']
corpus_dev = [d for d in corpus if d.split == 'dev']
# corpus_train_label = np.array([d.label for d in corpus if d.split == 'train'])

corpus_len = len(corpus)

simple_models = [  # 参数调整，模型比较
    # PV-DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=100, window=10, negative=20, hs=0, min_count=2, workers=cores, compute_loss=True),
    # PV-DBOW
    Doc2Vec(dm=0, size=100, negative=20, hs=0, min_count=2, workers=cores, compute_loss=True),
    # PV-DM w/ average
    Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=20, hs=0, min_count=2, workers=cores, compute_loss=True),
]
models_by_name = OrderedDict((str(model), model) for model in simple_models)

# 只要初始化一次词库
simple_models[0].build_vocab(corpus)  # PV-DM w/ concat requires one special NULL word so it serves as template
print(simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print(model)

best_error = defaultdict(lambda: 1.0)  # To selectively print only best errors achieved
alpha, min_alpha, epoches = 2.5, 0.1, 20
alpha_delta = (alpha - min_alpha) / epoches
print("START %s" % datetime.datetime.now())
infer_steps = 20
infer_alpha = 0.1
for epoch in range(1, epoches + 1):
    shuffle(corpus)
    for name, train_model in models_by_name.items():
        duration = 'na'
        train_model.alpha, train_model.min_alpha = alpha, alpha
        with elapsed_timer() as elapsed:
            train_model.train(corpus, total_examples=corpus_len, epochs=1)  # Train
            duration = '%.1f' % elapsed()
        print('-%d epoch, duration=%s,' % (epoch, duration))

        if (epoch % 5 == 0) or (epoch == epoches):
            dev_labels = [doc.label for doc in corpus if doc.split == 'dev']
            clf = linear_model.SGDClassifier(max_iter=3000, verbose=1, tol=20, learning_rate='constant', eta0=0.01,
                                             class_weight={1: 1, 0: 1})

            clf.fit([train_model.docvecs[doc.tags[0]] for doc in corpus if doc.split == 'train'],
                    [doc.label for doc in corpus if doc.split == 'train'])

            classifedLabels_dev = clf.predict(
                [train_model.docvecs[doc.tags[0]] for doc in corpus if doc.split == 'dev'])
            print(
                ' [clf  ' + name + ' ]' + '\n' + str(classification_report(dev_labels, classifedLabels_dev)))

            classifedLabels_dev_infer = clf.predict(np.array(
                [train_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in corpus if
                 doc.split == 'dev']))
            print(
                ' [clf  ' + name + ' infer ]' + '\n' + str(
                    classification_report(dev_labels, classifedLabels_dev_infer)))

            alpha -= alpha_delta
            print('[complete epoch %d ]' % (epoch))

            print("END %s" % str(datetime.datetime.now()))
