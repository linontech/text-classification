# -*- coding: utf-8 -*-
# @Author  : lzw
# description: using gensim doc2vec to handle text classification( doc2vec as text representation )
# reference : tutorial on gensim official (https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb)
# trick: 1. using iterator to handle large file problem;
#        2. use an external metric to evaluate doc2vec embedding.(sentiment classification)

import datetime
import multiprocessing
import os
from collections import OrderedDict
from collections import defaultdict

import jieba
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn import linear_model
from sklearn.metrics import classification_report

from gensim_t.data_helper import getVecs, load_text_label, load_data_without_store_ixs_labeled
from gensim_t.train_utils import elapsed_timer

jieba.load_userdict('../dataset/digital forum comments/digital_selfdict.txt')
# logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
cores = multiprocessing.cpu_count()

corpus_path = '../dataset/digital forum comments/split_20180403/all.txt'
corpus = load_text_label(corpus_path)

simple_models = [  # 参数调整，模型比较
    # PV-DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=100, window=10, negative=20, hs=0, min_count=2, workers=cores),
    # PV-DBOW
    Doc2Vec(dm=0, size=200, negative=20, hs=0, min_count=2, workers=cores),
    # PV-DM w/ average
    Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=20, hs=0, min_count=2, workers=cores),
]
models_by_name = OrderedDict((str(model), model) for model in simple_models)

# Speed up setup by sharing results of the 1st model's vocabulary scan
simple_models[0].build_vocab(corpus)  # PV-DM w/ concat requires one special NULL word so it serves as template
print(simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print(model)

path = '../dataset/digital forum comments/split_20180403/classes/'
files = os.listdir(path)
classes_num = {'0': 84145, '1': 22057}  # get these nums first
split = [0.8, 0.2]
train_lens, dev_lens, train_len_total, dev_len_total = {}, {}, 0, 0
for _class, num in classes_num.items():
    train_lens[_class] = int(num * split[0])
    dev_lens[_class] = num - train_lens[_class]
    train_len_total += train_lens[_class]
    dev_len_total += dev_lens[_class]

best_error = defaultdict(lambda: 1.0)  # To selectively print only best errors achieved
alpha, min_alpha, epoches = 0.1, 0.001, 50
alpha_delta = (alpha - min_alpha) / epoches
print("START %s" % datetime.datetime.now())
infer_steps = 20
infer_alpha = 0.1
for epoch in range(1, epoches + 1):
    text_gens = {}
    for file in files:  # generate train, dev text sets' generators in shuffled order
        # print(file)  # check you know the generators' order; filename=labelname
        class_name=file.split('.')[0]
        text_gens[class_name] = load_data_without_store_ixs_labeled(path + file, classes_num[class_name])

    train_gen = [next(text_gens[c]) for c in classes_num.keys() for _ in range(train_lens[c])]
    dev_gen = [next(text_gens[c]) for c in classes_num.keys() for _ in range(dev_lens[c])]

    for name, train_model in models_by_name.items():
        # Train
        duration = 'na'
        train_model.alpha, train_model.min_alpha = alpha, alpha
        with elapsed_timer() as elapsed:
            train_model.train(train_gen, total_examples=train_len_total, epochs=1)
            duration = '%.1f' % elapsed()
        print('-%d epoch， duration=%s' % (epoch, duration))

        if (epoch % 5 == 0) or (epoch == epoches):
            clf = linear_model.SGDClassifier(max_iter=3000, verbose=1, tol=20, learning_rate='constant', eta0=0.01,
                                             class_weight={1: 1, 0: 1})

            clf.fit(getVecs(train_model, train_len_total, train_model.vector_size),
                    np.array([i.label for i in train_gen]))

            classifedLabels = clf.predict(np.array(
                [train_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in dev_gen]))
            print(
                '[clf  ' + name + ']' + '\n' + str(classification_report([i.label for i in dev_gen], classifedLabels)))

        alpha -= alpha_delta

print("END %s" % str(datetime.datetime.now()))
