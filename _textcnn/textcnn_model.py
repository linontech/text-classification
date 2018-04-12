# -*- coding: utf-8 -*-
# Time    : 2018/1/22 22:39
# Author  : lzw
# description: textcnn model referenced from brightmart with my comments.
# reference: brightmart/text_classification (
# https://github.com/brightmart/text_classification)

import tensorflow as tf
import numpy as np

class TextCNN:
    def __init__(self, filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,
                 is_training,class_weights,initializer=tf.random_normal_initializer(stddev=0.1),multi_label_flag=False,clip_gradients=5.0,decay_rate_big=0.50):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.is_training=is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")#ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes=filter_sizes # it is a list of int. e.g. [3,4,5]
        self.num_filters=num_filters
        self.initializer=initializer
        self.num_filters_total=self.num_filters * len(filter_sizes) #how many filters totally.
        self.multi_label_flag=multi_label_flag
        self.clip_gradients = clip_gradients
        self.class_weights=class_weights

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32, [None,],name="input_y")  # y:[None,num_classes]
        self.input_y_multilabel = tf.placeholder(tf.float32,[None,self.num_classes], name="input_y_multilabel")  # y:[None,num_classes]. this is for multi-label classification only.
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(1,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference() #[None, self.label_size]. main computation graph is here.
        if not is_training:
            return
        if multi_label_flag:
            print("going to use multi label loss.")
            print ("not implement yet")
            exit()
            # self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            # self.loss_val = self.loss()
            self.loss_val=self.loss_weight()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]

        if not self.multi_label_flag:
            correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()
        else:
            print (" Not implement error. ")
            exit()

    def instantiate_weights(self):
        """define all weights here"""
        self.word_embeddings = tf.get_variable("word_embeddings",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) #[vocab_size,embed_size] 
                                #tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
        self.W_projection = tf.get_variable("W_projection",shape=[self.num_filters_total, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
        self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       #[label_size] #ADD 2017.06.09

    def inference(self):
        # textcnn主要的图结构: 1.embedding-->2.average-->3.linear classifier
        
        # 1. get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.word_embeddings,self.input_x)#[None,sentence_length,embed_size] 第一个None都是未指定的batch_size
        self.sentence_embeddings_expanded=tf.expand_dims(self.embedded_words,-1) #[None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
                                                                                 #[batch_size, height, width, channels] 这里处理的是数据所以channels=1
        # 2. loop each filter size. 
        # for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" %filter_size):

                filter=tf.get_variable("filter-%s"%filter_size,[filter_size,self.embed_size,1,self.num_filters],initializer=self.initializer) # create filter
                """
                  过滤器的维度为[height, width, channels, out_channels] 这里的filter_size是kim论文中的window_size
                  这里的channels对应上面self.sentence_embeddings_expanded的channels，
                  out_channels表示有几个这样的过滤器（卷积核）
                """

                conv=tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1,1,1,1], padding="VALID",name="conv")# convolution 卷积
                """
                  stride 表示图像在每一维的步长，一共有4维。batch，height，width，deepth=1（由句子embedding组成的矩阵）
                  padding='valid', 代表不padding； 相对的'same'则是padding使得输出的特征矩阵weight，height还和输入的一样
                  p.s textcnn在做卷积的时候把window_size内的wordembeddings链接在一起
                  卷积之后的维度shape:[batch_size, sequence_length - filter_size + 1, 1, num_filters]
                """

                # 2. apply non-linearity
                b=tf.get_variable("b-%s"%filter_size,[self.num_filters])
                h=tf.nn.relu(tf.nn.bias_add(conv,b),"relu") 
                """
                 卷积得到的特征放入非线性激活函数，得到feature map，将feature map输入到池化层
                 shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. 
                """
                pooled=tf.nn.max_pool(h, ksize=[1,self.sequence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")# pool 池化
                """
                 ksize: 池化窗口的大小，
                        一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
                        这里 “self.sequence_length-filter_size+1”是feature map的维度
                 strides: 窗口在每一个维度上滑动的步长
                          一般是[1, stride, stride, 1]
                 池化后这里的输出是scalar因为是max，输出的shape: [batch_size, 1, 1, num_filters] 
                """
                pooled_outputs.append(pooled)

        # 3. combine all pooled features, and flatten the feature.output' shape is a [1,None]
        self.h_pool=tf.concat(pooled_outputs,3) # 所有池化后的features打平
        # tf.concat=>concatenates tensors along one dimension.
        # shape:[batch_size, 1, 1, num_filters_total].   num_filters_total=num_filters_1+num_filters_2+num_filters_3
        #         # e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #         x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #         x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]

        self.h_pool_flat=tf.reshape(self.h_pool,[-1,self.num_filters_total])
        # here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
        # shape should be:[None,num_filters_total].

        #4. add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            self.h_drop=tf.nn.dropout(self.h_pool_flat,keep_prob=self.dropout_keep_prob) #[None,num_filters_total]

        #5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop,self.W_projection) + self.b_projection  
            #shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        return logits

    def loss_weight(self, l2_lambda=0.01):

        with tf.name_scope("weight_loss"):

            weights = tf.gather(self.class_weights, self.input_y)
            losses = tf.losses.sparse_softmax_cross_entropy(labels=self.input_y, logits=self.logits, weights=weights)
            loss=tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses

        return loss

    def loss(self,l2_lambda=0.01):#0.001
        with tf.name_scope("loss"):
            #input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            #losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss=tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def loss_multilabel(self,l2_lambda=0.01): #0.0001 #this loss function is for multi-label classification
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            #input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))

            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits);
                    #losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
                    #losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:",losses) #shape=(?, 1999).
            losses=tf.reduce_sum(losses,axis=1) #shape=(?,). loss for all data in the batch
            loss=tf.reduce_mean(losses)         #shape=().   average loss in the batch
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

#test started
#def test():
    #below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    #num_classes=3
    #learning_rate=0.01
    #batch_size=8
    #decay_steps=1000
    #decay_rate=0.9
    #sequence_length=5
    #vocab_size=10000
    #embed_size=100
    #is_training=True
    #dropout_keep_prob=1 #0.5
    #filter_sizes=[3,4,5]
    #num_filters=128
    #textRNN=TextCNN(filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,is_training)
    #with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   for i in range(100):
    #        input_x=np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
    #        input_x[input_x>0.5]=1
    #       input_x[input_x <= 0.5] = 0
    #       input_y=np.array([1,0,1,1,1,2,1,1])#np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
    #       loss,acc,predict,W_projection_value,_=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.W_projection,textRNN.train_op],feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})
    #       print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
            #print("W_projection_value_:",W_projection_value)
#test()