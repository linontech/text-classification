# -*- coding: utf-8 -*-
# @Author  : lzw

import logging
import math
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

import utils

logging.getLogger().setLevel(logging.INFO)


class doc2vec_model:
    def __init__(self,
                 model_type,
                 document_size,
                 vocabulary_size,
                 num_of_steps_per_epoch,
                 embedding_size_w,
                 embedding_size_d,
                 num_neg_samples,
                 use_pretrained_word_embeddings=False,
                 batch_size=32,
                 window_size=5,
                 learning_rate=0.01,
                 n_epochs=20,
                 eval_every_epochs=10,
                 tolerance=10,
                 eplison=1e-5,
                 logdir='tmp/logs/'):

        self.logdir = logdir
        self.model_type = model_type
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.eval_every_epochs = eval_every_epochs
        self.tolerance = tolerance
        self.eplison = eplison
        self.num_of_steps_per_epoch = num_of_steps_per_epoch
        self.num_neg_samples = num_neg_samples

        self.window_size = window_size
        self.document_size = document_size
        self.vocabulary_size = vocabulary_size

        self.embedding_size_w = embedding_size_w
        self.embedding_size_d = embedding_size_d
        self.use_pretrained_word_embeddings = use_pretrained_word_embeddings  # 1/0

        if (self.model_type == 'pvdm'):
            self.init_pvdm()
        elif (self.model_type == 'pvdbow'):
            self.init_pvdbow()
        elif (self.model_type == None):
            # a infer model
            pass

    def init_pvdbow(self):

        self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)

        self.doc_embeddings = tf.Variable(  # 初始化embedding参数在 [-1,1] 之间的随机数
            tf.random_uniform([self.document_size, self.embedding_size_d], -1.0, 1.0), name='doc_embeddings',
            dtype=tf.float32)

        # 初始化weights参数为 mean=0, stddev=1/sqrt(embedding_size_d)  的正态分布（高斯分布）
        # the constituent word embeddings of doc embeddings; the output word vectors in pvdbow
        # in IBM research paper, 2016, using a pretrained word embeddings as initialising here helps building the
        #    doc embeddings, better than randomly initialising. the paper says "with a pretrained word embeddings, it
        #    become more easily for doc embeddings to be close to its more critical content words."
        # changing name "weights" to "output_wordvectors"; also need a "assign" method here
        self.weights = tf.Variable(
            tf.truncated_normal([self.vocabulary_size, self.embedding_size_d],
                                stddev=1.0 / math.sqrt(self.embedding_size_d)), name='output_wordvectors', dtype=tf.float32)

        self.word_embeddings = self.weights # add nickname here for not using same method "assign_pretrained_word_embedding"
        self.biases = tf.Variable(tf.zeros([self.vocabulary_size]), name='biases', dtype=tf.float32)

        embed_d = tf.nn.embedding_lookup(self.doc_embeddings, self.train_dataset[:, 0])

        self.lr = tf.train.exponential_decay(self.learning_rate,
                                             global_step=self.global_step,
                                             decay_steps=self.eval_every_epochs * self.num_of_steps_per_epoch,
                                             decay_rate=0.96,
                                             staircase=True)

        # 还可以使用 sampled_softmax_loss() 来计算loss，上面这个函数直接帮我们考虑了negative_sampling
        # reduce_mean() 计算batch中所有样本的loss的平均值
        # 在计算test_loss的时候不考虑negative sampling，因为负采样只能够增加训练速度，而会少估计真实的损失，
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=self.weights,
                           biases=self.biases,
                           labels=self.train_labels,
                           inputs=embed_d,
                           num_sampled=self.num_neg_samples,
                           num_classes=self.document_size), name='loss')
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

        # optimizer = tf.train.AdamOptimizer(self.lr)
        # grads_and_vars = optimizer.compute_gradients(self.loss)
        # self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        # normalized 归一化（将要处理的数据限制在一定的范围之内）
        # 归一化的目的是是没有可比性的数据，变得有可比性，同时又保持两个数据之间的相对关系，比如在作图的时候。
        # 在机器学习算法的预处理阶段，归一化是一个重要步骤。例如在训练 svm 等线性模型时。一般的，推荐将每个属性缩放
        # 到 [-1,1]; [0,1]。
        # 这里使得每个向量各自的加总合为1。
        self.normalized_doc_embeddings = tf.div(self.doc_embeddings,
                                                tf.sqrt(
                                                    tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True)),
                                                name='doc_embeddings_norm')

    def init_pvdm(self, average=0):

        self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)

        self.word_embeddings = tf.Variable(  # 初始化embedding参数
            tf.random_uniform([self.vocabulary_size, self.embedding_size_w], -1.0, 1.0), name='word_embeddings',
            dtype=tf.float32)
        self.doc_embeddings = tf.Variable(
            tf.random_uniform([self.document_size, self.embedding_size_d], -1.0, 1.0), name='doc_embeddings',
            dtype=tf.float32)

        combined_embed_vector_length = self.embedding_size_w * (self.window_size - 1) + self.embedding_size_d

        self.weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, combined_embed_vector_length],  # 初始化weights参数
                                                  stddev=1.0 / math.sqrt(combined_embed_vector_length)), name='weights',
                              dtype=tf.float32)

        self.biases = tf.Variable(tf.zeros([self.vocabulary_size]), name='biases', dtype=tf.float32)

        embed = []
        embed_d = tf.nn.embedding_lookup(self.doc_embeddings, self.train_dataset[:, 0])
        embed.append(embed_d)
        for j in range(1, self.window_size):
            embed_w = tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
            embed.append(embed_w)

        if not average:
            """
            # 论文说这里可以是将嵌入向量 average或者concatenate
            # 回顾一下，在word2vec cbow 的模型中对嵌入向量的处理可以是 average或者sum"""
            embed = tf.concat(embed, 1)

        self.lr = tf.train.exponential_decay(self.learning_rate,
                                             global_step=self.global_step,
                                             decay_steps=self.eval_every_epochs * self.num_of_steps_per_epoch,
                                             decay_rate=0.96,
                                             staircase=True)

        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=self.weights,
                           biases=self.biases,
                           labels=self.train_labels,
                           inputs=embed,
                           num_sampled=self.num_neg_samples,
                           num_classes=self.vocabulary_size), name='loss')
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
        # grads_and_vars = optimizer.compute_gradients(self.loss)
        # self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        self.normalized_word_embeddings = tf.div(self.word_embeddings,
                                                 tf.sqrt(
                                                     tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True)),
                                                 name='word_embeddings_norm')
        self.normalized_doc_embeddings = tf.div(self.doc_embeddings,
                                                tf.sqrt(
                                                    tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True)),
                                                name='doc_embeddings_norm')

    def infer_vector_pvdm(self,
                          checkpoint_dir,
                          new_train_batches,
                          index2word,
                          word2index,
                          infer_texts,
                          trained_texts,
                          model_info):
        """
        #
        :return:
        """
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        logging.info('[infer_vector_pvdm] Loaded the trained model: {}'.format(checkpoint_file))
        restore_graph = tf.Graph()
        with restore_graph.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            session_conf = tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))

            with tf.Session(config=session_conf, graph=restore_graph) as sess:  # restore parameters
                saver.restore(sess, checkpoint_file)
                get_word_embeddings, get_doc_embeddings = \
                    restore_graph.get_operation_by_name('word_embeddings').outputs[0], \
                    restore_graph.get_operation_by_name('doc_embeddings').outputs[0]
                get_weights = restore_graph.get_operation_by_name('weights').outputs[0]
                get_biases = restore_graph.get_operation_by_name('biases').outputs[0]

                old_word_embeddings, old_doc_embeddings, old_weights, old_biases = sess.run(
                    [get_word_embeddings, get_doc_embeddings, get_weights, get_biases])

                new_doc_start_index = old_doc_embeddings.shape[0]

        # if sess is not None:
        #     print('Close interactive session pvdm')
        #     sess.close()
        # del sess

        new_graph = tf.Graph()
        with new_graph.as_default():

            self.predict_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size])
            self.predict_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)

            new_doc_embeddings = np.random.uniform(-1, 1, [self.document_size, self.embedding_size_d])
            concat_doc_embeddings_nparray = np.concatenate((old_doc_embeddings, new_doc_embeddings), axis=0)
            self.concat_doc_embeddings = tf.Variable(
                tf.zeros([concat_doc_embeddings_nparray.shape[0], concat_doc_embeddings_nparray.shape[1]]),
                name='doc_embeddings', dtype=tf.float32)
            word_embeddings = tf.Variable(tf.zeros([old_word_embeddings.shape[0], old_word_embeddings.shape[1]]),
                                          trainable=False, name='word_embeddings', dtype=tf.float32)

            weights = tf.Variable(tf.zeros([old_weights.shape[0], old_weights.shape[1]]), trainable=False,
                                  name='weights', dtype=tf.float32)
            biases = tf.Variable(tf.zeros([old_biases.shape[0]]), trainable=False, name='biases', dtype=tf.float32)

            assign_doc_embeddings = tf.assign(self.concat_doc_embeddings, concat_doc_embeddings_nparray)
            assign_word_embeddings = tf.assign(word_embeddings, old_word_embeddings)
            assign_weights = tf.assign(weights, old_weights)
            assign_biases = tf.assign(biases, old_biases)

            embed = []
            embed_d = tf.nn.embedding_lookup(self.concat_doc_embeddings,
                                             new_doc_start_index + self.predict_dataset[:, 0])
            embed.append(embed_d)
            for j in range(1, self.window_size):
                embed_w = tf.nn.embedding_lookup(word_embeddings, self.predict_dataset[:, j])
                embed.append(embed_w)

            embed = tf.concat(embed, 1)

            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=weights,
                               biases=biases,
                               labels=self.predict_labels,
                               inputs=embed,
                               num_sampled=self.num_neg_samples,
                               num_classes=self.vocabulary_size), name='loss')

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                                 global_step=self.global_step,
                                                 decay_steps=self.eval_every_epochs * self.num_of_steps_per_epoch,
                                                 decay_rate=0.96,
                                                 staircase=True)

            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                                                 global_step=self.global_step)

            norm_w = tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = tf.div(word_embeddings, norm_w, name='word_embeddings_norm')

            norm_d = tf.sqrt(tf.reduce_sum(tf.square(self.concat_doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = tf.div(self.concat_doc_embeddings, norm_d, name='doc_embeddings_norm')

            init = tf.global_variables_initializer()
            saver = tf.train.Saver(tf.global_variables())

            #
            session_conf = tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
            session_conf.gpu_options.allow_growth = True

            out_dir = os.path.abspath(
                os.path.join(os.path.curdir,
                             "model/retrained_infer_model_pvdm" + '_' + model_info))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            def train_step(x_batch, y_batch):
                feed_dict = {
                    self.predict_dataset: x_batch,
                    self.predict_labels: y_batch
                }
                _, loss_train = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

                return loss_train

            best_train_avg_loss, best_at_step = float("inf"), 1
            current_epoch = 1
            total_train_loss = 0
            tolerance = self.tolerance
            logging.info('[infer_vector_pvdm] Following is kind of retrained things, since you are inferring. ')
            with tf.Session(config=session_conf, graph=new_graph) as sess:

                logging.info("[infer_vector_pvdm] assign old variables w, b, docvectors, wordvectors")
                sess.run(init)  # initialize variables
                sess.run(assign_biases)
                sess.run(assign_weights)
                sess.run(assign_doc_embeddings)
                sess.run(assign_word_embeddings)
                for train_batch in new_train_batches:
                    x_train_batch, y_train_batch = zip(*train_batch)
                    loss = train_step(x_train_batch, y_train_batch)
                    current_step = tf.train.global_step(sess, self.global_step)
                    total_train_loss += loss

                    if (current_step % self.num_of_steps_per_epoch == 0) and (tolerance > 0):
                        avg_train_loss = total_train_loss / self.num_of_steps_per_epoch
                        logging.info(
                            '[infer_vector_pvdm] Current epoch: {}, avg loss on train set: {}'.format(current_epoch,
                                                                                                      avg_train_loss))

                        if best_train_avg_loss - avg_train_loss > self.eplison:
                            best_train_avg_loss, best_at_step = avg_train_loss, current_step
                        else:
                            tolerance -= 1
                            logging.info(
                                '[infer_vector_pvdm] {} tolerance left, best avg train loss: {} at epoch {}.'.format(
                                    tolerance,
                                    best_train_avg_loss,
                                    current_epoch))

                        if (current_epoch % self.eval_every_epochs == 0) or (tolerance == 0) or (
                                    current_epoch == self.n_epochs):

                            logging.info('+++++++++++++++++++++++++++++++++++eval+++++++++++++++++++++++++++++++++')
                            sim_matric_argmax_ix = self.calSimilarity_infer(sess, new_doc_start_index,
                                                                            k=10)  # 计算相似度，LabeledSentences_train前十个句子
                            oneToTen_ix = range(10)

                            assert len(oneToTen_ix) == len(sim_matric_argmax_ix)
                            logging.info('-' * 80)
                            i = 1
                            for t, sim_t in zip(oneToTen_ix, sim_matric_argmax_ix):
                                logging.info('第{}句话: '.format(i))
                                logging.info(
                                    infer_texts[t] + '\n'
                                    + '   \t\t'
                                    + ' '.join(
                                        utils.ids2words(index2word, utils.words2ids(word2index, infer_texts[t].split(
                                            ' ')))) + '\n'
                                    + '   \t\t'
                                    + trained_texts[sim_t] + '\n'
                                    + '   \t\t'
                                    + ' '.join(utils.ids2words(index2word,
                                                               utils.words2ids(word2index,
                                                                               trained_texts[sim_t].split(' ')))))
                                i += 1
                            logging.info('-' * 80)

                            if (tolerance == 0) or (current_epoch == self.n_epochs):
                                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                                logging.info(
                                    '[infer_vector_pvdm] {} tolerance left. Saved model at {} at step {}'.format(
                                        tolerance, path,
                                        best_at_step))
                                logging.info(
                                    '[infer_vector_pvdm] Best avg loss on train is {} at step {}'.format(
                                        best_train_avg_loss, best_at_step))
                                break

                        total_train_loss = 0
                        current_epoch += 1

                retrained_doc_embeddings = sess.run(self.normalized_doc_embeddings)

        logging.info('[infer_vector_pvdm] ' +
                     str(datetime.now()).replace(':', '-') + '  retraining completed. {} tolerances used. '.format(
            self.tolerance - tolerance))

        return retrained_doc_embeddings, new_doc_start_index

    def infer_vector_pvdbow(self,
                            checkpoint_dir,
                            new_train_batches,
                            index2word,
                            infer_texts,
                            trained_texts,
                            word2index,
                            model_info):
        """
        while training in order to evaluate the similarity performance, every 20 epoches calulate the similarity
        between the first 10 sentences in new added sentnece between old use-to-train sentences.
        :return:
        """
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        logging.info('[infer_vector_pvdbow] Loaded the trained model: {}'.format(checkpoint_file))
        restore_graph = tf.Graph()
        with restore_graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))

            with tf.Session(config=session_conf, graph=restore_graph).as_default() as sess:
                saver.restore(sess, checkpoint_file)

                get_doc_embeddings = restore_graph.get_operation_by_name('doc_embeddings').outputs[0]
                get_weights = restore_graph.get_operation_by_name('weights').outputs[0]
                get_biases = restore_graph.get_operation_by_name('biases').outputs[0]

                old_doc_embeddings, old_weights, old_biases = sess.run([get_doc_embeddings, get_weights, get_biases])

        new_doc_start_index = old_doc_embeddings.shape[0]

        del restore_graph
        if sess is not None:
            print('[infer_vector_pvdbow] Close interactive session pvdbow')
            sess.close()

        new_graph = tf.Graph()
        with new_graph.as_default():

            self.predict_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.predict_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)

            weights = tf.Variable(tf.zeros([old_weights.shape[0], old_weights.shape[1]]), trainable=False,
                                  name='weights', dtype=tf.float32)
            biases = tf.Variable(tf.zeros([old_biases.shape[0]]), trainable=False, name='biases', dtype=tf.float32)

            new_doc_embeddings = np.random.uniform(-1, 1, [self.document_size, self.embedding_size_d])
            concat_doc_embeddings_nparray = np.concatenate((old_doc_embeddings, new_doc_embeddings), axis=0)
            self.concat_doc_embeddings = tf.Variable(
                tf.zeros([concat_doc_embeddings_nparray.shape[0], concat_doc_embeddings_nparray.shape[1]]),
                name='doc_embeddings', dtype=tf.float32)
            embed_d = tf.nn.embedding_lookup(self.concat_doc_embeddings, self.predict_dataset[:, 0])

            assign_doc_embeddings = tf.assign(self.concat_doc_embeddings, concat_doc_embeddings_nparray)
            assign_weights = tf.assign(weights, old_weights)
            assign_biases = tf.assign(biases, old_biases)

            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=weights,
                               biases=biases,
                               labels=self.predict_labels,
                               inputs=embed_d,
                               num_sampled=self.num_neg_samples,
                               num_classes=self.vocabulary_size), name='loss')

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                                 global_step=self.global_step,
                                                 decay_steps=self.eval_every_epochs * self.num_of_steps_per_epoch,
                                                 decay_rate=0.96,
                                                 staircase=True)

            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
                                                                                 global_step=self.global_step)

            norm_d = tf.sqrt(tf.reduce_sum(tf.square(self.concat_doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = tf.div(self.concat_doc_embeddings, norm_d, name='doc_embeddings_norm')

            init = tf.global_variables_initializer()
            saver = tf.train.Saver(tf.global_variables())

            session_conf = tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
            session_conf.gpu_options.allow_growth = True

            out_dir = os.path.abspath(
                os.path.join(os.path.curdir,
                             "model/retrained_infer_model_pvdbow" + '_' + model_info))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            def train_step(x_batch, y_batch):
                feed_dict = {
                    self.predict_dataset: x_batch,
                    self.predict_labels: y_batch
                }
                _, loss_train = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

                return loss_train

            best_train_avg_loss, best_at_step = float("inf"), 1
            current_epoch = 1
            total_train_loss = 0
            tolerance = self.tolerance
            logging.info('[infer_vector_pvdbow] Following is kind of retrained things, since you are inferring. ')
            with tf.Session(config=session_conf, graph=new_graph).as_default() as sess:

                sess.run(init)  # initialize variables
                logging.info("[infer_vector_pvdbow] assign old variables w, b, docvectors")
                sess.run(assign_biases)
                sess.run(assign_weights)
                sess.run(assign_doc_embeddings)

                for train_batch in new_train_batches:
                    x_train_batch, y_train_batch = zip(*train_batch)
                    loss = train_step(x_train_batch, y_train_batch)
                    current_step = tf.train.global_step(sess, self.global_step)
                    total_train_loss += loss

                    if (current_step % self.num_of_steps_per_epoch == 0) and (tolerance > 0):
                        avg_train_loss = total_train_loss / self.num_of_steps_per_epoch
                        logging.info(
                            '[infer_vector_pvdbow] Current epoch: {}, avg loss on train set: {}'.format(current_epoch,
                                                                                                        avg_train_loss))

                        if best_train_avg_loss - avg_train_loss > self.eplison:
                            best_train_avg_loss, best_at_step = avg_train_loss, current_step
                        else:
                            tolerance -= 1
                            logging.info(
                                '[infer_vector_pvdbow] {} tolerance left, best avg train loss: {} at epoch {}.'.format(
                                    tolerance,
                                    best_train_avg_loss,
                                    current_epoch))

                        if (current_epoch % self.eval_every_epochs == 0) or (tolerance == 0) or (
                                    current_epoch == self.n_epochs):

                            logging.info('+++++++++++++++++++++++++++++++++++eval+++++++++++++++++++++++++++++++++')
                            sim_matric_argmax_ix = self.calSimilarity_infer(sess, new_doc_start_index,
                                                                            k=10)  # 计算相似度，LabeledSentences_train前十个句子
                            oneToTen_ix = range(10)

                            assert len(oneToTen_ix) == len(sim_matric_argmax_ix)
                            logging.info('-' * 80)
                            i = 1
                            for t, sim_t in zip(oneToTen_ix, sim_matric_argmax_ix):
                                logging.info('第{}句话: '.format(i))
                                logging.info(
                                    infer_texts[t] + '\n'
                                    + '   \t\t'
                                    + ' '.join(utils.ids2words(index2word, utils.words2ids(word2index,
                                                                                           infer_texts[t].split(
                                                                                               ' ')))) + '\n'
                                    + '   \t\t'
                                    + trained_texts[sim_t] + '\n'
                                    + '   \t\t'
                                    + ' '.join(utils.ids2words(index2word, utils.words2ids(word2index,
                                                                                           trained_texts[sim_t].split(
                                                                                               ' ')))))
                                i += 1
                            logging.info('-' * 80)

                            if (tolerance == 0) or (current_epoch == self.n_epochs):
                                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                                logging.info(
                                    '[infer_vector_pvdbow] {} tolerance left. Saved model at {} at step {}'.format(
                                        tolerance, path,
                                        best_at_step))
                                logging.info(
                                    '[infer_vector_pvdbow] Best avg loss on train is {} at step {}'.format(
                                        best_train_avg_loss, best_at_step))
                                break

                        total_train_loss = 0
                        current_epoch += 1

                retrained_doc_embeddings = sess.run(self.normalized_doc_embeddings)
        logging.info(
            str(datetime.now()).replace(':', '-') + '  retraining completed. {} tolerances used. '.format(
                self.tolerance - tolerance))

        return retrained_doc_embeddings, new_doc_start_index

    def calSimilarity(self, sess, k=10):

        normalized_doc_embeddings = sess.run(self.normalized_doc_embeddings)
        doc_embeddings_top10 = normalized_doc_embeddings[:k, :]
        doc_embeddings_rest = normalized_doc_embeddings[k:, :]

        matrix = np.dot(doc_embeddings_rest, doc_embeddings_top10.T)
        norm_all = np.linalg.norm(doc_embeddings_rest, axis=1, keepdims=True)
        norm_top10 = np.linalg.norm(doc_embeddings_top10, axis=1, keepdims=True)
        norm_matrix_row = np.dot(norm_all, norm_top10.T)  # i列j行表示，第i(69138)个句子与第j(10)个句子之间各自norm相乘

        cos = matrix / norm_matrix_row
        cos = 0.5 + 0.5 * cos  # 归一化
        sim_matric_argmax_ix = np.argmax(cos, 0)

        return sim_matric_argmax_ix

    def calSimilarity_infer(self, sess, new_doc_start_index, k=10):

        normalized_doc_embeddings = sess.run(self.concat_doc_embeddings)
        doc_embeddings_top10 = normalized_doc_embeddings[new_doc_start_index:new_doc_start_index + k, :]
        doc_embeddings_rest = normalized_doc_embeddings[:new_doc_start_index:, :]

        matrix = np.dot(doc_embeddings_rest, doc_embeddings_top10.T)
        norm_all = np.linalg.norm(doc_embeddings_rest, axis=1, keepdims=True)
        norm_top10 = np.linalg.norm(doc_embeddings_top10, axis=1, keepdims=True)
        norm_matrix_row = np.dot(norm_all, norm_top10.T)  # i列j行表示，第i(69138)个句子与第j(10)个句子之间各自norm相乘

        cos = matrix / norm_matrix_row
        cos = 0.5 + 0.5 * cos  # 归一化
        sim_matric_argmax_ix = np.argmax(cos, 0)

        return sim_matric_argmax_ix
