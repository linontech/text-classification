# -*- coding: utf-8 -*-
# beginners' textrnn with my comments. mine is with """""" type while the author's comments are with "#".
#            all comments provided with resource/reference, for further reading and handle copyright problem. 
# description: textrnn for text classification.
# reference: brightmart/text_classification (
#            https://github.com/brightmart/text_classification)import tensorflow as tf

from tensorflow.contrib import rnn
import numpy as np
import tensorflow as tf


class TextRNN:
    def __init__(self,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,
                 vocab_size,embed_size,hidden_size,is_training,class_weights,initializer=tf.random_normal_initializer(stddev=0.1)):

        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.is_training=is_training
        self.learning_rate=learning_rate
        self.initializer=initializer
        self.num_sampled=20
        self.class_weights=class_weights

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32,[None], name="input_y")  # y [None,num_classes]
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(1,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference() #[None, self.label_size]. main computation graph is here.
        if not is_training:
            return
        # self.loss_val = self.loss() # self.loss_nce() specific which loss to use.
        self.loss_val=self.loss_weight()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()

    def instantiate_weights(self):
        # define all weights here
        self.word_embeddings = tf.get_variable("word_embeddings",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) #[vocab_size,embed_size]
                                                                                                            # tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
        self.W_projection = tf.get_variable("W_projection",shape=[self.hidden_size*2, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
        self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])#[label_size]

    def inference(self):
        # main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax 

        #1.get emebeddings of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.word_embeddings,self.input_x) #shape:[None,sentence_length,embed_size]

        #2. Bi-lstm layer (define lstm cess:get lstm cell output)
        lstm_fw_cell=rnn.BasicLSTMCell(self.hidden_size) #forward direction cell
        lstm_bw_cell=rnn.BasicLSTMCell(self.hidden_size) #backward direction cell
        if self.dropout_keep_prob is not None:
            lstm_fw_cell=rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell=rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, dtype=tf.float32)
        """ 
         creates a dynamic bidirectional recurrent neural network
          1) idirectional_dynamic_rnn: input: [batch_size, max_time, input_size] "max_time" is the shape of "self.embedded_words" here. 
                                     outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor` 
                                                    # each for one input word, and shape = [batch_size,sequence_length,hidden_size], 
                                     output_states: only the final states of bidirectional rnn. (fw,bw)
          2) static_bidirectional_rnn : fixed number of RNN cells, while "bidirectional_dynamic_rnn" generate RNN cells during training time using while function.while
          difference: i) the dynamic one can have multiple number of RNN cells, while the static one can not change RNN cells, if you have a sentence whose length longer
                        than length of static RNN cells, it just cut it. 
                      ii) though dynamic one can handle multiple length of sentences, you still need to pad it wo the same length, and then input a parameter "sequence_length"
                        into it, so that it knows how long each sentence is in a batch.
                        ref:https://danijar.com/variable-sequence-lengths-in-tensorflow/
                            https://stackoverflow.com/questions/43396431/how-to-convert-static-rnn-inputs-to-dynamic-rnn-inputs-in-tensorflow
        """

        print("outputs:===>",outputs) 
        
        #3. concat output
        output_rnn=tf.concat(outputs,axis=2) #[batch_size,sequence_length,hidden_size*2]
        self.output_rnn_last=tf.reduce_mean(output_rnn, axis=1) 
        """
            why? reduce_mean here ? 
            as in the graph and the documentation, the upper "bidirectional_dynamic_rnn" function generates 
                "outputs" are fw, bw hidden states output of each element in sequence of a sentence.
            so its shape become [batch_size,hidden_size*2]

            print("output_rnn_last:", self.output_rnn_last) # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>
        """

        #4. logits(use linear layer) # no activation layer here ? 
        with tf.name_scope("output"): #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection  # [batch_size,num_classes]

        return logits

    def loss_weight(self, l2_lambda=0.01):

        with tf.name_scope("weight_loss"):

            weights = tf.gather(self.class_weights, self.input_y)
            losses = tf.losses.sparse_softmax_cross_entropy(labels=self.input_y, logits=self.logits, weights=weights)
            loss=tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses

        return loss

    def loss(self,l2_lambda=0.01):
        """ Compute the "sparse_softmax_cross_entropy_with_logits" loss for a uni-label. """
        #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
        #        logits 指的是神经网络最后一层的输出，
        #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss=tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)

            """ tensorflow中四种主要的损失函数
                1) losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
                    使用场景，单标签多分类，多标签多分类（每类互相排斥）
                    得到一个交叉熵向量，若是求损失函数则求平均，求交叉熵则加总，因为有多个样本。
                    输入的labels是某样本可能属于每个类的概率，加总和为1
                2) losses=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
                    使用场景，单标签多分类，每类互相排斥
                    输入的labels必须有一个指定的类别，注意必须是一个，不能是概率分布
                3) losses=tf.nn.sigmoid_cross_entropy_with_logits()
                    使用场景，二分类或者多标签多分类
                    类别不互相独立。举例来说，"一个图像可能由多个类别，里面同时有一只狗和一只大象"
                4) losses=tf.nn.weighted_cross_entropy_with_logits()
                    使用场景，多标签多分类
                    带有权重的sigmoid_cross_entropy_with_logits()，调整正/负样本的在损失函数中的权值，使得你可以在recall和precision之间权衡
            """ #reference:https://blog.csdn.net/u013250416/article/details/78230464

        return loss

    def loss_nce(self,l2_lambda=0.01): #0.0001-->0.001
        # Compute the average NCE loss for the batch.
        #input: W_projection, b_projection, output_rnn_last, different from upper function.

        #labels=tf.reshape(self.input_y,[-1])               #[batch_size,1]------>[batch_size,]
        labels=tf.expand_dims(self.input_y,1)                   #[batch_size,]----->[batch_size,1]
        loss = tf.reduce_mean( #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            tf.nn.nce_loss(weights=tf.transpose(self.W_projection),#[hidden_size*2, num_classes]--->[num_classes,hidden_size*2]. nce_weights:A `Tensor` of shape `[num_classes, dim].O.K.
                           biases=self.b_projection,                 #[label_size]. nce_biases:A `Tensor` of shape `[num_classes]`.
                           labels=labels,                 #[batch_size,1]. train_labels, # A `Tensor` of type `int64` and shape `[batch_size,num_true]`. The target classes.
                           inputs=self.output_rnn_last,# [batch_size,hidden_size*2] #A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
                           num_sampled=self.num_sampled,  #scalar. 100
                           num_classes=self.num_classes,
                           partition_strategy="div"))  #scalar. 1999
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
        """ 
        background: training a model with lots of classes costs a lot of time, for example a language model with large corpus.
                    训练一个类别非常多的分类器，训练过程会非常缓慢。为了解决这个问题，有了候选采样的技巧，每次只用类别中的一小部分作梯度下降回传误差。
        tensorflow中有在样本中选出候选样本子集的函数，uniform_candidate_sampler/log_uniform_candidate_sampler/...
        在生成候选类别子集后送给采样损失函数计算损失，最小化候选采样损失便能训练模型。两种:(两者的区别主要在损失函数不同)
            1) tf.nn.sampled_softmax_loss / softmax
            2) tf.nn.nce_loss            / simoid (logistic) / automatically draws a new sample of the negative labels each time we evaluate the loss
                                        / 优先采样词频高的词作为负样本 
               Here the basic Idea is to train logistic regression classifier which can separate the samples obtained 
                from true distribution and sampled obtained from noise distribution.
        """ # reference:http://www.algorithmdog.com/tf-candidate-sampling, https://blog.csdn.net/u010223750/article/details/69948463
             # reference:https://www.linkedin.com/pulse/heavy-softmax-use-nce-loss-shamane-siriwardhana/
        return loss

    def train(self):
        #based on the loss, use SGD to update parameter
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op

# def test(): #test started
#     #below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
#     num_classes=10
#     learning_rate=0.01
#     batch_size=8
#     decay_steps=1000
#     decay_rate=0.9
#     sequence_length=5
#     vocab_size=10000
#     embed_size=100
#     is_training=True
#     dropout_keep_prob=1#0.5
#     textRNN=TextRNN(num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,is_training)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for i in range(100):
#             input_x=np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
#             input_y=input_y=np.array([1,0,1,1,1,2,1,1]) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
#             loss,acc,predict,_=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.train_op],feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})
#             print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
# test()