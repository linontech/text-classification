# -*- coding: utf-8 -*-
# training the model.
# modified: lzw
# reference: brightmart/text_classification (
#            https://github.com/brightmart/text_classification)
# description: textrnn handle sentiment classification
#               handle vocabulary size, drop too rare word and too frequent word( if in stopwords )

import os
import sys

sys.path.insert(0, "../")
import numpy as np
import tensorflow as tf
# from data_util_zhihu import load_data_multilabel_new,create_voabulary,create_voabulary_label
from tflearn.data_utils import pad_sequences, shuffle  # to_categorical
from _textcnn.textcnn_model import TextCNN
from data_helper import load_text_label, split_file_classes, load_text_label_test, load_obj, generate_texts2indexes
import jieba
import logging
import pickle

# configuration
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes", 2, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir", "text_cnn_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length", 30, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 60, "embedding size")
tf.app.flags.DEFINE_integer("validate_every", 10, "Validate every validate_every epochs.")  # 每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
tf.app.flags.DEFINE_integer("num_filters", 64, "cnn filter size")
tf.app.flags.DEFINE_string("traning_data_path", "", "path of traning data.")
tf.app.flags.DEFINE_string("pretrained_word_embeddings_path",
                           "../_gensim_support/vocabulary/new_word2vec_12.3w.pkl",
                           "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("cache_path", "cache/tmp_cache.save", "for tmp data reload.")
tf.app.flags.DEFINE_boolean("multi_label_flag",False,"use multi label or single label.")
tf.app.flags.DEFINE_integer("tolerance",5,"early stop of training.")
tf.app.flags.DEFINE_float("eplison",1e-4,"how much to set as an improvement")
filter_sizes=[1,2,3,4,5,6,7]

def main(_):
    jieba.load_userdict('../dataset/digital_forum/digital_selfdict.txt')

    if os.path.exists(FLAGS.cache_path):
        with open(FLAGS.cache_path, 'rb') as data_f:
            trainX, trainY, devX, devY, testX, testY, word2vec_dict, index2word, label_index = pickle.load(data_f)
            vocab_size = len(index2word)
    else:

        """ load texts """
        classes_path = '../dataset/digital_forum/classes/'
        label_index, class_name_count = split_file_classes('../dataset/digital_forum/all.txt',
                                                           '../dataset/digital_forum/classes/')
        split = [0.7, 0.3]
        corpus, train_len_total, dev_len_total = load_text_label(classes_path, class_name_count, label_index, split,
                                                                 shuffle=True)
        corpus_train = [d for d in corpus if d.split == 'train']
        corpus_dev = [d for d in corpus if d.split == 'dev']
        corpus_unk = [d for d in corpus if d.split == 'UNK']  # sentences with unknown label # corpus_len = len(corpus)
        print('train texts num: %d' % (train_len_total))
        print('dev texts num: %d' % (dev_len_total))
        print('unk texts num %d' % (len(corpus_unk)))

        """ load test texts """
        test_filepath = '../dataset/digital_forum/测试集3_联想_杨欣标注15000_20171116 (2).txt'
        corpus_test = load_text_label_test(test_filepath, label_index)
        # test_labels = [d.label for d in corpus_test]
        # test_negative_len = len([d for d in corpus_test if d.label == 0])

        """
            using a corpus or a built vocabulary.
        """
        use_pretrained_word_embeddings = True
        if use_pretrained_word_embeddings:
            pretrained_word_embeddings_path = FLAGS.pretrained_word_embeddings_path
            logging.info(
                "using pre-trained word embeddings, word2vec_model_path: {}".format(pretrained_word_embeddings_path))
            word2vec_dict = load_obj(pretrained_word_embeddings_path)
            word2index = {'_UNK_': 0, '_NULL_': 1}
            for i, k_v in enumerate(word2vec_dict.items()):
                word2index[k_v[0]] = i + 2
            index2word = dict(zip(word2index.values(), word2index.keys()))
            vocab_size = len(index2word)

        trainX = generate_texts2indexes(input_data=[d.words for d in corpus_train],
                                        word2index=word2index)
        trainY = [d.label for d in corpus_train]
        devX = generate_texts2indexes(input_data=[d.words for d in corpus_dev],
                                      word2index=word2index)
        devY = [d.label for d in corpus_dev]
        testX = generate_texts2indexes(input_data=[d.words for d in corpus_test],
                                       word2index=word2index)
        testY = [d.label for d in corpus_test]

        # 2.Data preprocessing.Sequence padding; why pad behind?
        print("start padding & transform to one hot...")
        trainX = pad_sequences([t.word_indexes for t in trainX], maxlen=FLAGS.sequence_length,
                               value=0.)  # padding to max length
        devX = pad_sequences([t.word_indexes for t in devX], maxlen=FLAGS.sequence_length,
                             value=0.)  # padding to max length
        testX = pad_sequences([t.word_indexes for t in testX], maxlen=FLAGS.sequence_length,
                              value=0.)  # padding to max length
        # print(testY)
        ###############################################################################################
        with open(FLAGS.cache_path, 'wb') as data_f:  # save data to cache file, so we can use it next time quickly.
            pickle.dump((trainX, trainY, devX, devY, testX, testY, word2vec_dict, index2word, label_index), data_f)
        ###############################################################################################
        print("trainX[0]:", trainX[0])  # ;print("trainY[0]:", trainY[0])
        print("end padding & transform to one hot...")
        exit()

    class_weights = tf.constant([0.75, 0.25])
    train(trainX, trainY, devX, devY, testX, testY, word2vec_dict, index2word, vocab_size, FLAGS.embed_size,
          label_index,class_weights)


def train(trainX, trainY, devX, devY, testX, testY, word2vec_dict, index2word, vocab_size, embed_size, label_index,class_weights):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        textcnn = TextCNN(filter_sizes,
                    FLAGS.num_filters,
                    FLAGS.num_classes,
                    FLAGS.learning_rate, 
                    FLAGS.batch_size, 
                    FLAGS.decay_steps,
                    FLAGS.decay_rate,
                    FLAGS.sequence_length,
                    vocab_size,
                    FLAGS.embed_size,
                    FLAGS.is_training,
                    class_weights,
                    multi_label_flag=FLAGS.multi_label_flag)

        saver = tf.train.Saver()  # Initialize Save
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):  # for continue training
            print("Restoring Variables from Checkpoint for rnn model.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:  # assign pre-trained word embedding
                assign_pretrained_word_embedding(sess, word2vec_dict, index2word, embed_size, textcnn)

        curr_epoch = sess.run(textcnn.epoch_step)
        number_of_training_data = len(trainX)
        num_batches_per_epoch = int(number_of_training_data / FLAGS.batch_size) + 1
        tol = FLAGS.tolerance
        best_train_loss, best_at_step = float("inf"), 1

        for epoch in range(curr_epoch, FLAGS.num_epochs + 1):  # 3.feed data & training
            shuffle(trainX)  # shuffle data
            loss, acc, counter = 0.0, 0.0, 0
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * FLAGS.batch_size
                end_index = min((batch_num + 1) * FLAGS.batch_size, number_of_training_data)
                # for start, end in zip(range(0, number_of_training_data, FLAGS.batch_size),
                #                       range(FLAGS.batch_size, number_of_training_data, FLAGS.batch_size)):
                if epoch == 1 and counter == 0:
                    print("trainX[start:end]:",
                          trainX[start_index:end_index])  # ;print("trainY[start:end]:",trainY[start:end])
                curr_loss, curr_acc, _ = sess.run([textcnn.loss_val, textcnn.accuracy, textcnn.train_op],
                                                  feed_dict={textcnn.input_x: trainX[start_index:end_index],
                                                             textcnn.input_y: trainY[start_index:end_index],
                                                             textcnn.dropout_keep_prob: 0.5})
                loss, counter, acc = loss + curr_loss, counter + 1, acc + curr_acc
                if counter % 250 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.5f\tTrain Accuracy:%.5f" % (
                        epoch, counter, loss / float(counter),acc / float(counter)))

            loss_avg = loss/counter
            if best_train_loss - loss_avg > FLAGS.eplison:
                best_train_loss, best_at_step = loss_avg, epoch
            else:
                tol -= 1
                print("%d tolerance left" % tol)

            # print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every == 0))
            if epoch % FLAGS.validate_every == 0 or epoch == FLAGS.num_epochs or tol==0:  # 4.validation
                # eval_loss, eval_acc = do_eval(sess, textRNN, testX, testY, FLAGS.batch_size, index2word)
                # print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))
                do_eval_confusion_matrix(sess, textcnn, devX, devY, 2, label_index)

                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)

            if tol==0:
                print ("0 tolerance. training stops.")
                break

            # print("going to increment epoch counter....")  # epoch increment
            sess.run(textcnn.epoch_increment)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        do_eval_confusion_matrix(sess, textcnn, testX, testY, 2, label_index)



def assign_pretrained_word_embedding(sess, word2vec_dict, index2word, embed_size, target_model):
    """
    """
    vocab_size = len(index2word)
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0
    for i in range(0, vocab_size):  # loop each word
        word = index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, embed_size)
            count_not_exist = count_not_exist + 1  # init a random value for the word.

    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    # print (word_embedding_final)
    t_assign_embedding = tf.assign(target_model.word_embeddings,
                                   word_embedding)
    sess.run(t_assign_embedding)
    logging.info("word exists embedding: {}, word not exist embedding: {}".format(count_exist, count_not_exist))
    logging.info("using pre-trained word emebedding ended.")


def do_eval_confusion_matrix(sess, textRNN, evalX, evalY, num_classes, label_index):
    """
    ref:https://stats.stackexchange.com/questions/179835/how-to-build-a-confusion-matrix-for-a-multiclass-classifier
    """
    number_examples = len(evalX)
    num_batches_per_epoch = int(number_examples / FLAGS.batch_size) + 1
    all_predictions = []
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * FLAGS.batch_size
        end_index = min((batch_num + 1) * FLAGS.batch_size, number_examples)
        # for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        batch_predictions = sess.run(textRNN.predictions, feed_dict={textRNN.input_x: evalX[start_index:end_index],
                                                                     textRNN.input_y: evalY[start_index:end_index],
                                                                     textRNN.dropout_keep_prob: 1})

        all_predictions = np.concatenate((all_predictions, batch_predictions))

    confusion_matrix = tf.contrib.metrics.confusion_matrix(all_predictions, evalY)
    matrix = sess.run(confusion_matrix)
    # print(matrix)
    true_pos = np.diag(matrix)
    false_pos = np.sum(matrix, axis=0) - true_pos
    false_neg = np.sum(matrix, axis=1) - true_pos
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    print('\t\tprecision\trecall\t\tf1-score')
    index_label = dict(zip(label_index.values(), label_index.keys()))
    for _class_index in range(num_classes):
        f1_score = (2 * recall[_class_index] * precision[_class_index]) / (
            recall[_class_index] + precision[_class_index])
        print('' + index_label[_class_index] + ' \t%.3f\t\t%.3f\t\t%.3f' % (
            recall[_class_index], precision[_class_index], f1_score))


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, textRNN, evalX, evalY, batch_size, vocabulary_index2word_label):
    number_examples = len(evalX)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        curr_eval_loss, logits, curr_eval_acc = sess.run([textRNN.loss_val, textRNN.logits, textRNN.accuracy],
                                                         # curr_eval_acc--->textCNN.accuracy
                                                         feed_dict={textRNN.input_x: evalX[start:end],
                                                                    textRNN.input_y: evalY[start:end]
                                                             , textRNN.dropout_keep_prob: 1})
        # label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        # curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
        eval_loss, eval_acc, eval_counter = eval_loss + curr_eval_loss, eval_acc + curr_eval_acc, eval_counter + 1
    return eval_loss / float(eval_counter), eval_acc / float(eval_counter)

if __name__ == "__main__":
    tf.app.run()
