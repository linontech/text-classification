# -*- coding: utf-8 -*-
# @Author  : lzw

import logging
import os
from datetime import datetime
from itertools import tee
import jieba
import numpy as np
import tensorflow as tf

import utils
from doc2vec_model import doc2vec_model
from lr_model import lr_model

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
batch_size = 64
n_epochs = 100 # big batches causes memory.
eval_every_epochs = 20
# cal_sim_every_epochs = 30
tolerance = 10
eplison = 1e-4
learning_rate = 0.1
window_size = 5  # 5-12
embedding_size_w = 100
embedding_size_d = 100


def train(model_type,
          word2index,
          index2word,
          train_texts,
          train_batches,
          vocabulary_size,
          document_size,
          embedding_size_w,
          embedding_size_d,
          batch_size,
          n_epochs,
          eval_every_epochs,
          eplison,
          tolerance=tolerance,
          logdir='logs/tmp'):
    timestamp = str(datetime.now()).replace(':', '-')
    logdir = logdir + '_' + timestamp
    graph = tf.Graph()  # tf.device('/gpu:0'),
    with tf.device('/gpu:0'), graph.as_default():

        num_of_batches_total = len(train_batches) # calculate a generator's len
        # train_batches_gen = tee(train_batches, 2)  # 复制生成器
        # num_of_batches_total = sum(1 for _ in train_batches_gen[0])
        n_steps = num_of_batches_total
        sample_len = num_of_batches_total * batch_size
        num_of_steps_per_epoch = num_of_batches_total // n_epochs + 1
        logging.critical('{} LabeledSentences_train sentences.'.format(document_size))
        logging.critical('{} samples.'.format(sample_len))
        logging.critical('{} steps to go. '.format(n_steps))
        logging.critical('{} epochs to go. '.format(n_epochs))
        logging.critical('{} steps per epoch.'.format(num_of_steps_per_epoch))
        logging.critical('saving variables to path: {}'.format(logdir))

        global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)

        model = doc2vec_model(model_type=model_type,
                              embedding_size_w=embedding_size_w,
                              embedding_size_d=embedding_size_d,
                              batch_size=batch_size,
                              vocabulary_size=vocabulary_size,
                              document_size=document_size,
                              eval_every_epochs=eval_every_epochs,
                              num_of_steps_per_epoch=num_of_steps_per_epoch,
                              tolerance=tolerance,
                              eplison=eplison,
                              n_epochs=n_epochs,
                              logdir=logdir)

        lr = tf.train.exponential_decay(model.learning_rate,
                                        global_step=global_step,
                                        decay_steps=model.eval_every_epochs * model.num_of_steps_per_epoch,
                                        decay_rate=0.96,
                                        staircase=True)

        optimizer = tf.train.AdamOptimizer(lr)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(
            os.path.join(os.path.curdir, "model/trained_model_" + model_type + '_' + timestamp))

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables())

        def train_step(x_batch, y_batch):
            feed_dict = {
                model.train_dataset: x_batch,
                model.train_labels: y_batch
            }
            _, loss_train = sess.run([train_op, model.loss], feed_dict=feed_dict)

            return loss_train

        best_train_avg_loss, best_at_step = float("inf"), 1
        current_epoch = 1
        total_train_loss = 0
        tolerance = model.tolerance

        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        logging.critical(str(datetime.now()).replace(':', '-') + '  Start training. ')
        with tf.Session(config=session_conf).as_default() as sess:

            sess.run(tf.global_variables_initializer())
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                loss = train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)
                total_train_loss += loss
                if current_step%1000==0:
                    print (current_step)
                if (current_step % num_of_steps_per_epoch == 0) and (tolerance > 0):
                    avg_train_loss = total_train_loss / num_of_steps_per_epoch
                    logging.critical(
                        'Current epoch: {}, avg loss on train set: {}'.format(current_epoch, avg_train_loss))

                    if best_train_avg_loss - avg_train_loss > eplison:
                        best_train_avg_loss, best_at_step = avg_train_loss, current_step
                    else:
                        tolerance -= 1
                        logging.critical(
                            '{} tolerance left, avg train loss: {} at epoch {}.'.format(tolerance,
                                                                                        best_train_avg_loss,
                                                                                        current_epoch))

                    if (current_epoch % eval_every_epochs == 0) or (tolerance == 0) or (current_epoch == n_epochs):

                        logging.critical('+++++++++++++++++++++++++++++++++++eval+++++++++++++++++++++++++++++++++')
                        sim_matric_argmax_ix = model.calSimilarity(sess, k=10)  # 计算相似度，LabeledSentences_train前十个句子
                        oneToTen_ix = range(10)

                        assert len(oneToTen_ix) == len(sim_matric_argmax_ix)

                        logging.critical('-' * 80)
                        i = 1
                        for t, sim_t in zip(oneToTen_ix, sim_matric_argmax_ix):
                            logging.critical('第{}句话: '.format(i))
                            logging.critical(
                                train_texts[t] + '\n'
                                + ' \t\t\t'
                                + ' '.join(utils.ids2words(index2word, utils.words2ids(word2index, train_texts[t].split(
                                    ' ')))) + '\n'
                                + ' \t\t\t'
                                + train_texts[sim_t] + '\n'
                                + ' \t\t\t'
                                + ' '.join(utils.ids2words(index2word,
                                                           utils.words2ids(word2index, train_texts[sim_t].split(' ')))))
                            i += 1
                        logging.critical('-' * 80)

                        if (tolerance == 0) or (current_epoch == n_epochs):
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            logging.critical(
                                '{} tolerance left. Saved model at {} at step {}'.format(tolerance, path,
                                                                                         best_at_step))
                            logging.critical(
                                'Best avg loss on train is {} at step {}'.format(best_train_avg_loss, best_at_step))
                            break

                    total_train_loss = 0
                    current_epoch += 1

    logging.critical(str(datetime.now()).replace(':', '-') + '  Training completed. {} tolerances used. '.format(
        model.tolerance - tolerance))


def get_embeddings(checkpoint_dir, model_type):
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    logging.critical('Loaded the trained model: {}'.format(checkpoint_file))
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            if (model_type == 'pvdm'):
                normalized_word_embeddings = graph.get_operation_by_name('word_embeddings').outputs[0]
                word_embeddings = sess.run(normalized_word_embeddings)
            else:
                word_embeddings = None

            normalized_doc_embeddings = graph.get_operation_by_name('doc_embeddings').outputs[0]
            doc_embeddings = sess.run(normalized_doc_embeddings)
    return word_embeddings, doc_embeddings


def infer(checkpoint_dir_pvdm,
          checkpoint_dir_pvdbow,
          new_LabeledSentences_train,
          train_texts,
          word2index,
          index2word,
          vocabulary_size,
          document_size,
          eval_every_epochs,
          batch_size=batch_size,
          tolerance=tolerance,
          n_epochs=n_epochs,
          logdir=None):
    new_train_batches_pvdbow = utils.generate_pvdbow_batches(batch_size=batch_size,
                                                             window_size=window_size,
                                                             LabeledSentences=new_LabeledSentences_train,
                                                             n_epochs=n_epochs)
    num_of_batches_total = len(new_train_batches_pvdbow)
    n_steps = num_of_batches_total
    sample_len = num_of_batches_total * batch_size
    num_of_steps_per_epoch = num_of_batches_total // n_epochs + 1
    logging.critical(str(datetime.now()).replace(':', '-') + '  Start infering pvdbow. ')
    logging.critical('{} new LabeledSentences_train_pvdbow sentences.'.format(document_size))
    logging.critical('{} samples.'.format(sample_len))
    logging.critical('{} steps to go. '.format(n_steps))
    logging.critical('{} epochs to go. '.format(n_epochs))

    model_pvdbow_infer = doc2vec_model(model_type=None,
                                       batch_size=batch_size,
                                       learning_rate=learning_rate,
                                       vocabulary_size=vocabulary_size,
                                       document_size=document_size,
                                       eval_every_epochs=eval_every_epochs,
                                       num_of_steps_per_epoch=num_of_steps_per_epoch,
                                       tolerance=tolerance,
                                       n_epochs=n_epochs,
                                       logdir=logdir)

    new_doc_embeddings_pvdbow, new_doc_start_index_pvdbow = \
        model_pvdbow_infer.infer_vector_pvdbow(checkpoint_dir=checkpoint_dir_pvdbow,
                                               index2word=index2word,
                                               word2index=word2index,
                                               train_texts=train_texts,
                                               new_train_batches=new_train_batches_pvdbow)

    new_train_batches_pvdm = utils.generate_pvdbow_batches(batch_size=batch_size,
                                                           window_size=window_size,
                                                           LabeledSentences=new_LabeledSentences_train,
                                                           n_epochs=n_epochs)

    num_of_batches_total = len(new_train_batches_pvdm)
    n_steps = num_of_batches_total
    sample_len = num_of_batches_total * batch_size
    num_of_steps_per_epoch = num_of_batches_total // n_epochs + 1
    logging.critical(str(datetime.now()).replace(':', '-') + '  Start infering pvdm. ')
    logging.critical('{} new LabeledSentences_train-pvdm sentences.'.format(document_size))
    logging.critical('{} samples.'.format(sample_len))
    logging.critical('{} steps to go. '.format(n_steps))
    logging.critical('{} epochs to go. '.format(n_epochs))
    model_pvdm_infer = doc2vec_model(model_type=None,  # fake infer model
                                     batch_size=batch_size,
                                     learning_rate=learning_rate,
                                     vocabulary_size=vocabulary_size,
                                     document_size=document_size,
                                     eval_every_epochs=eval_every_epochs,
                                     num_of_steps_per_epoch=num_of_steps_per_epoch,
                                     tolerance=tolerance,
                                     n_epochs=n_epochs,
                                     logdir=logdir)

    new_doc_embeddings_pvdm, new_doc_start_index_pvdm = \
        model_pvdm_infer.infer_vector_pvdm(checkpoint_dir=checkpoint_dir_pvdm,
                                           word2index=word2index,
                                           index2word=index2word,
                                           train_texts=train_texts,
                                           new_train_batches=new_train_batches_pvdm)

    assert new_doc_start_index_pvdbow == new_doc_start_index_pvdm
    assert new_doc_embeddings_pvdm.shape == new_doc_embeddings_pvdbow.shape

    return np.concatenate((new_doc_embeddings_pvdm[-new_doc_start_index_pvdm:],
                           new_doc_embeddings_pvdbow[-new_doc_start_index_pvdbow:]), axis=1)


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    train_already = False
    logging.critical(str(datetime.now()).replace(':', '-') + '  start load_data.')
    train_texts = utils.load_data(filepath='dataset/sentence_20180318_without_stopwords.txt',
                                  sample_file_path='dataset/sample_sentences.txt',
                                  data_size=100000) # 175000

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
                                              word2index=word2index,
                                              window_size=window_size)
    # tee_gen_train      = tee(train_data, 2)  # 复制生成器
    logging.critical(str(datetime.now()).replace(':', '-') + '  end generate_texts2indexes.')

    if not train_already:
        # logging.critical('Start training doc2vec. ')
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
        word2index_path = 'vocabulary/word2index' + '_v1'
        index2word_path = 'vocabulary/index2word' + '_v1'
        word2index = utils.load_obj(word2index_path)
        id2word = utils.load_obj(index2word_path)

        logging.critical('Get embeddings.')
        trained_model_checkpoint_dir_pvdbow = 'model/trained_model_pvdbow_2018-03-15 13-46-43.285413/checkpoints'
        trained_model_checkpoint_dir_pvdm = 'model/trained_model_pvdm_2018-03-15 14-25-00.769217/checkpoints'

        _, doc_embeddings_dbow = get_embeddings(checkpoint_dir=trained_model_checkpoint_dir_pvdbow,
                                                model_type=model_type_pvdbow)

        word_embeddings_dm, doc_embeddings_dm = get_embeddings(checkpoint_dir=trained_model_checkpoint_dir_pvdm,
                                                               model_type=model_type_pvdm)
        embedding = np.concatenate((doc_embeddings_dbow, doc_embeddings_dm), axis=1)

        """
        训练 logistic regression
        """
        train_texts, train_labels = utils.load_data_label(train_filepath, window_size=window_size)
        new_LabeledSentences_train = utils.generate_texts2indexes(input_data=train_texts, word2index=word2index,
                                                                 window_size=window_size)
        logging.critical('Num of test sentences: {}'.format(len(train_texts)))

        new_test_batches = utils.generate_pvdbow_batches(batch_size=batch_size,
                                                         window_size=window_size,
                                                         LabeledSentences=train_data,
                                                         n_epochs=n_epochs)

        assert embedding.shape[0] == len(train_labels)
        checkpoint_dir_pvdm = trained_model_checkpoint_dir_pvdm
        checkpoint_dir_pvdbow = trained_model_checkpoint_dir_pvdbow
        new_train_batches = utils.generate_pvdbow_batches(batch_size=batch_size,
                                                          window_size=window_size,
                                                          LabeledSentences=train_data,
                                                          n_epochs=n_epochs)

        test_embeddings = infer(checkpoint_dir_pvdm=checkpoint_dir_pvdm,
                                checkpoint_dir_pvdbow=checkpoint_dir_pvdbow,
                                new_LabeledSentences_train=new_LabeledSentences_train,
                                train_texts=train_texts,
                                word2index=word2index,
                                index2word=index2word,
                                batch_size=batch_size,
                                vocabulary_size=len(word2index),
                                document_size=len(new_LabeledSentences_train),
                                eval_every_epochs=eval_every_epochs,
                                tolerance=tolerance,
                                n_epochs=n_epochs,
                                logdir=None)
        # compare with sklearn logistic optimazation. looks like my logistic regression still not perfect...
        # from sklearn import linear_model
        # lr = linear_model.SGDClassifier(max_iter=500, verbose=1, tol=100, epsilon=0.01, learning_rate='constant', eta0=0.00001)
        # lr.fit(embedding, train_labels)
        #
        # classifedLabels = lr.predict(embedding)
        lr = lr_model(epsilon=1e-3,
                      llambda=0.01,
                      alpha=0.1,
                      num_iters=8000,
                      batch_size=embedding.shape[0],
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
        #
        print('Accuracy on training data: ', (np.sum(classifedLabels == train_labels) / len(train_labels)) * 100, '%')

        ################################################################################################################
        """
        评估 logistic regression 在测试集
        """

        test_texts, test_labels = utils.load_data_label(test_filepath, window_size=window_size)
        new_LabeledSentences_test = utils.generate_texts2indexes(input_data=test_texts, word2index=word2index,
                                                                 window_size=window_size)
        logging.critical('Num of test sentences: {}'.format(len(test_texts)))

        new_test_batches = utils.generate_pvdbow_batches(batch_size=batch_size,
                                                         window_size=window_size,
                                                         LabeledSentences=train_data,
                                                         n_epochs=n_epochs)
        checkpoint_dir_pvdm = trained_model_checkpoint_dir_pvdm
        checkpoint_dir_pvdbow = trained_model_checkpoint_dir_pvdbow
        test_embeddings = infer(checkpoint_dir_pvdm=checkpoint_dir_pvdm,
                                checkpoint_dir_pvdbow=checkpoint_dir_pvdbow,
                                new_LabeledSentences_train=new_LabeledSentences_test,
                                train_texts=test_texts,
                                word2index=word2index,
                                index2word=index2word,
                                batch_size=batch_size,
                                vocabulary_size=len(word2index),
                                document_size=len(new_LabeledSentences_test),
                                eval_every_epochs=eval_every_epochs,
                                tolerance=tolerance,
                                n_epochs=n_epochs,
                                logdir=None)

        assert test_embeddings.shape[0] == len(test_labels)
        classifedLabels = lr.classify(test_embeddings)
        classifedProbs, classifedLabels = zip(*classifedLabels)
        # classifedProbs = np.array(classifedProbs).flatten()
        classifedLabels = np.array(classifedLabels).flatten()

        print('Accuracy on test data: ', (np.sum(classifedLabels == test_labels) / len(test_labels)) * 100, '%')

