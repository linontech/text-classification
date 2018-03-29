# -*- coding: utf-8 -*-
# @Author  : lzw

import logging
import os
from datetime import datetime
import jieba
import numpy as np
import tensorflow as tf
# from itertools import tee

import utils
from doc2vec_model import doc2vec_model

logging.getLogger().setLevel(logging.INFO)
jieba.load_userdict('dataset/digital forum comments/digital_selfdict.txt')
# jieba.load_userdict('dataset/car forum comments/carSelfDict.txt')

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
          num_neg_samples,
          learning_rate,
          eplison,
          tolerance,
          model_info,
          logdir='logs/tmp'):

    graph = tf.Graph()  # tf.device('/gpu:0'),
    with tf.device('/gpu:0'), graph.as_default():

        num_of_batches_total = len(train_batches)  # calculate a generator's len
        # train_batches_gen = tee(train_batches, 2)  # 复制生成器
        # num_of_batches_total = sum(1 for _ in train_batches_gen[0])
        n_steps = num_of_batches_total
        sample_len = num_of_batches_total * batch_size
        num_of_steps_per_epoch = num_of_batches_total // n_epochs
        logging.info('{} LabeledSentences_train sentences.'.format(document_size))
        logging.info('{} samples.'.format(sample_len))
        logging.info('{} steps to go. '.format(n_steps))
        logging.info('{} epochs to go. '.format(n_epochs))
        logging.info('{} steps per epoch.'.format(num_of_steps_per_epoch))
        logging.info('saving variables to path: {}'.format(logdir))

        model = doc2vec_model(model_type=model_type,
                              embedding_size_w=embedding_size_w,
                              embedding_size_d=embedding_size_d,
                              batch_size=batch_size,
                              learning_rate=learning_rate,
                              vocabulary_size=vocabulary_size,
                              document_size=document_size,
                              eval_every_epochs=eval_every_epochs,
                              num_of_steps_per_epoch=num_of_steps_per_epoch,
                              tolerance=tolerance,
                              eplison=eplison,
                              num_neg_samples=num_neg_samples,
                              n_epochs=n_epochs,
                              logdir=logdir)
        #
        out_dir = os.path.abspath(
            os.path.join(os.path.curdir, "model/trained_model_" + model_type + '_' + model_info))

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
            _, loss_train = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)

            return loss_train

        best_train_avg_loss, best_at_step = float("inf"), 1
        current_epoch = 1
        total_train_loss = 0
        tolerance = model.tolerance

        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        logging.info(str(datetime.now()).replace(':', '-') + '  Start training. ')
        with tf.Session(config=session_conf).as_default() as sess:

            sess.run(tf.global_variables_initializer())
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                loss = train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, model.global_step)
                total_train_loss += loss

                if (current_step % num_of_steps_per_epoch == 0) and (tolerance > 0):
                    avg_train_loss = total_train_loss / num_of_steps_per_epoch
                    logging.info(
                        '[train ' + model.model_type + '] {}, avg loss on train set: {}'.format(current_epoch,
                                                                                                avg_train_loss))

                    if best_train_avg_loss - avg_train_loss > eplison:
                        best_train_avg_loss, best_at_step, best_at_epoch = avg_train_loss, current_step, current_epoch
                    else:
                        tolerance -= 1
                        logging.info(
                            '[train ' + model.model_type + '] {} tolerance left,best avg train loss: {} at step {}, epoch {}. '.format(
                                tolerance,
                                best_train_avg_loss,
                                best_at_step,
                                best_at_epoch))

                    if (current_epoch % eval_every_epochs == 0) or (tolerance == 0) or (current_epoch == n_epochs):

                        logging.info('+++++++++++++++++++++++++++++++++++eval+++++++++++++++++++++++++++++++++')
                        sim_matric_argmax_ix = model.calSimilarity(sess, k=10)  # 计算相似度，LabeledSentences_train前十个句子
                        oneToTen_ix = range(10)

                        assert len(oneToTen_ix) == len(sim_matric_argmax_ix)

                        logging.info('-' * 80)
                        i = 1
                        for t, sim_t in zip(oneToTen_ix, sim_matric_argmax_ix):
                            logging.info('第{}句话: '.format(i))
                            logging.info(
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
                        logging.info('-' * 80)

                        if (tolerance == 0) or (current_epoch == n_epochs):
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            logging.info(
                                '[train ' + model.model_type + '] {} tolerance left. Saved model at {} at step {}'.format(
                                    tolerance, path,
                                    best_at_step))
                            logging.info(
                                '[train ' + model.model_type + '] Best avg loss on train is {} at step {}'.format(
                                    best_train_avg_loss, best_at_step))
                            break

                    total_train_loss = 0
                    current_epoch += 1

    logging.info(str(datetime.now()).replace(':', '-') + '  Training completed. {} tolerances used. '.format(
        model.tolerance - tolerance))


def get_embeddings(checkpoint_dir, model_type):
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    logging.info( '[get embedding ' + model_type + '] Loaded the trained model: {}'.format(checkpoint_file))
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            if (model_type == 'pvdm'):
                normalized_word_embeddings = graph.get_operation_by_name('word_embeddings_norm').outputs[0]
                word_embeddings = sess.run(normalized_word_embeddings)
            else:
                word_embeddings = None

            normalized_doc_embeddings = graph.get_operation_by_name('doc_embeddings_norm').outputs[0]
            doc_embeddings = sess.run(normalized_doc_embeddings)
    return word_embeddings, doc_embeddings


def infer(checkpoint_dir_pvdm,
          checkpoint_dir_pvdbow,
          infer_LabeledSentences,
          trained_texts,
          infer_texts,
          word2index,
          index2word,
          vocabulary_size,
          document_size,
          eval_every_epochs,
          batch_size,
          window_size,
          embedding_size_w,
          embedding_size_d,
          learning_rate,
          num_neg_samples,
          eplison,
          tolerance,
          n_epochs,
          model_info,
          logdir=None):
    """
    create two fake model for infer. During inferring restore trained weights, biases, word_embeddings, and
    doc_embeddings. Add more columns to doc_embedding matrix. While training, only gradient descent updating the
    doc_embeddings.

    :param checkpoint_dir_pvdm:
    :param checkpoint_dir_pvdbow:
    :param new_LabeledSentences_train:
    :param train_texts:
    :param word2index:
    :param index2word:
    :param vocabulary_size:
    :param document_size:
    :param eval_every_epochs:
    :param batch_size:
    :param tolerance:
    :param n_epochs:
    :param logdir:
    :return:
    """
    # pvdbow infer
    new_train_batches_pvdbow = utils.generate_pvdbow_batches(batch_size=batch_size,
                                                             window_size=window_size,
                                                             LabeledSentences=infer_LabeledSentences,
                                                             n_epochs=n_epochs)
    num_of_batches_total = len(new_train_batches_pvdbow)
    n_steps = num_of_batches_total
    sample_len = num_of_batches_total * batch_size
    num_of_steps_per_epoch = num_of_batches_total // n_epochs
    logging.info(str(datetime.now()).replace(':', '-') + '  Start infering pvdbow. ')
    logging.info('[infer pvdbow] {} new LabeledSentences_train_pvdbow sentences.'.format(document_size))
    logging.info('[infer pvdbow] {} samples.'.format(sample_len))
    logging.info('[infer pvdbow] {} steps to go. '.format(n_steps))
    logging.info('[infer pvdbow] {} epochs to go. '.format(n_epochs))

    model_pvdbow_infer = doc2vec_model(model_type=None,
                                       embedding_size_w=embedding_size_w,
                                       embedding_size_d=embedding_size_d,
                                       batch_size=batch_size,
                                       learning_rate=learning_rate,
                                       vocabulary_size=vocabulary_size,
                                       document_size=document_size,
                                       eval_every_epochs=eval_every_epochs,
                                       num_of_steps_per_epoch=num_of_steps_per_epoch,
                                       tolerance=tolerance,
                                       num_neg_samples=num_neg_samples,
                                       eplison=eplison,
                                       n_epochs=n_epochs,
                                       logdir=logdir)

    new_doc_embeddings_pvdbow, new_doc_start_index_pvdbow = \
        model_pvdbow_infer.infer_vector_pvdbow(checkpoint_dir=checkpoint_dir_pvdbow,
                                               index2word=index2word,
                                               word2index=word2index,
                                               trained_texts=trained_texts,
                                               infer_texts=infer_texts,
                                               model_info=model_info,
                                               new_train_batches=new_train_batches_pvdbow)

    # pvdm infer
    new_train_batches_pvdm = utils.generate_pvdm_batches(batch_size=batch_size,
                                                         window_size=window_size,
                                                         LabeledSentences=infer_LabeledSentences,
                                                         n_epochs=n_epochs)

    num_of_batches_total = len(new_train_batches_pvdm)
    n_steps = num_of_batches_total
    sample_len = num_of_batches_total * batch_size
    num_of_steps_per_epoch = num_of_batches_total // n_epochs
    logging.info(str(datetime.now()).replace(':', '-') + '  Start infering pvdm. ')
    logging.info('[infer pvdm] {} new LabeledSentences_train-pvdm sentences.'.format(document_size))
    logging.info('[infer pvdm] {} samples.'.format(sample_len))
    logging.info('[infer pvdm] {} steps to go. '.format(n_steps))
    logging.info('[infer pvdm] {} epochs to go. '.format(n_epochs))
    model_pvdm_infer = doc2vec_model(model_type=None,  # fake infer model
                                     embedding_size_w=embedding_size_w,
                                     embedding_size_d=embedding_size_d,
                                     batch_size=batch_size,
                                     learning_rate=learning_rate,
                                     vocabulary_size=vocabulary_size,
                                     document_size=document_size,
                                     eval_every_epochs=eval_every_epochs,
                                     num_of_steps_per_epoch=num_of_steps_per_epoch,
                                     tolerance=tolerance,
                                     num_neg_samples=num_neg_samples,
                                     eplison=eplison,
                                     n_epochs=n_epochs,
                                     logdir=logdir)

    new_doc_embeddings_pvdm, new_doc_start_index_pvdm = \
        model_pvdm_infer.infer_vector_pvdm(checkpoint_dir=checkpoint_dir_pvdm,
                                           word2index=word2index,
                                           index2word=index2word,
                                           trained_texts=trained_texts,
                                           infer_texts=infer_texts,
                                           model_info=model_info,
                                           new_train_batches=new_train_batches_pvdm)

    assert new_doc_start_index_pvdbow == new_doc_start_index_pvdm
    assert new_doc_embeddings_pvdm.shape == new_doc_embeddings_pvdbow.shape
    # dm vec in the front
    return np.concatenate((new_doc_embeddings_pvdm[new_doc_start_index_pvdm:],
                           new_doc_embeddings_pvdbow[new_doc_start_index_pvdbow:]), axis=1)


def evaluate_analysis(x, y, y_pred, thresold, path):
    """
    :param x:
    :param y:
    :param y_pred:
    :param thresold:
    :param path: path of result out folder
    :return:
    """
    # assert y.shape == y_pred.shape
    logging.info('writing error analysis result to ' + path)
    from sklearn.metrics import classification_report
    y_predHat = [1 if i > thresold else 0 for i in y_pred]
    classification_report(y, y_predHat)
    f = open(path, 'w', encoding='utf-8')

    tmp = []
    f.write('ix\t' + 'real\t' + 'pred\t' + 'text\n')
    for ix in range(len(y_pred)):
        y_pred_single = 1 if y_pred[ix] > thresold  else 0
        if y_pred_single != y[ix]:  # write error first
            f.write(str(ix + 1) + '\t' + str(y[ix]) + '\t' +
                    str(y_pred[ix]) + '\t' + str(x[ix]) + '\n')
        tmp.append(str(ix + 1) + '\t' + str(y[ix]) + '\t' +
                   str(y_pred[ix]) + '\t' + str(x[ix]) + '\n')
    f.write('-----------------------ALL RESULT------------------------')
    for text in tmp:
        f.write(text)

    f.close()


