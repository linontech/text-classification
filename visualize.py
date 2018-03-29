# -*- coding: utf-8 -*-
# @Author  : lzw

import random
import time

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import utils
from train_utils import get_embeddings

def plot_with_labels(vis_x, vis_y, labels, title=None):

    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)
    # colors = ["gold", "limegreen"]
    # cmap = matplotlib.colors.ListedColormap(colors)
    label_num = len(set(labels))
    cmap = plt.cm.get_cmap("jet", label_num)

    sc=plt.scatter(vis_x, vis_y, c=labels, cmap=cmap)
    plt.colorbar(sc, ticks=range(label_num))

    if title is not None:
        plt.title(title)

    plt.savefig('pca-tsne.png')

"""
    visualize doc_embeddings with a 2 dimension graph, try using PCA or tSNE.

"""

print('Get embeddings.')
trained_model_checkpoint_dir_pvdbow = 'model/trained_model_pvdbow_2018-03-22 05-22-01.830643/checkpoints'
trained_model_checkpoint_dir_pvdm = 'model/trained_model_pvdm_2018-03-22 06-21-33.759069/checkpoints'

word_embeddings_None, doc_embeddings_dbow = get_embeddings(checkpoint_dir=trained_model_checkpoint_dir_pvdbow,
                                                           model_type='pvdbow')

word_embeddings_dm, doc_embeddings_dm = get_embeddings(checkpoint_dir=trained_model_checkpoint_dir_pvdm,
                                                       model_type='pvdm')
embedding = np.concatenate((doc_embeddings_dbow, doc_embeddings_dm), axis=1)

train_filepath = 'dataset/20171211_train.txt'  # load dataset
train_texts, train_labels = utils.load_data_label(train_filepath)
train_labels = np.array(train_labels)
train_data_len = len(train_texts)
train_labels_1_ix = np.where(train_labels== 1)
print('there are {} positive samples out of {} samples. '.format(len(train_labels_1_ix[0]), train_data_len))

train_labels_0_ix = []
count=0
for i in range(40000):  # generate 10000 ix for negative samples
    tmp = random.randint(0, train_data_len)
    if tmp not in train_labels_1_ix[0]:
        train_labels_0_ix.append(tmp)
        count += 1
    if count==20000:
        break

train_labels_0_ix = np.array(train_labels_0_ix)
assert len(train_labels_0_ix) == 20000

# pca_50 = PCA(n_components=100)  # lower dim using pca first
# pca_result_50 = pca_50.fit_transform(embedding)
# print('Explained variation per principal component (PCA): {}'.format(np.sum(pca_50.explained_variance_ratio_)))

n_sne_len = len(train_labels_1_ix[0]) + len(train_labels_0_ix)
print('drawing {} points. '.format(n_sne_len))
n_sne_ix = np.concatenate((train_labels_0_ix, train_labels_1_ix[0]))

t0 = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
pca_tsne_results = tsne.fit_transform(embedding[n_sne_ix])

plot_with_labels(pca_tsne_results[:,0], pca_tsne_results[:,1], train_labels[n_sne_ix], "t-SNE embedding (time %.2fs)" %
                 (time.time() - t0))  # plot

#test_case
# labels = np.random.randint(0,2, size = 300)
# extraPoints = np.random.uniform(0,10, size = (300,2))
# plot_with_labels(extraPoints[:,0], extraPoints[:,1], labels, None)