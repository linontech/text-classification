### Text Classifications
------
##### # Contains:

1. implementation of PV-DM, PV-DBoW models using tensorflow  structure (including inferencing new paragraph/sentence). My goal is to learn how to generate good vectors for sentences, paragraphs. Probably a good start point beginners who want to represent texts using neural model.
2. text_cnn, text_rnn model implementations referenced from [brightmart/text_classification](https://github.com/brightmart/text_classification) with my personal comments.
3. fasttext tutorial #todo

##### # todo

- evaluations of text representation

  - task0: the second experiment as benchmark in Mikolov. etal:2014, using the IMDB dataset.
    - IMDB (Large Movie Review Dataset) from "Learning Word Vectors for Sentiment Analysis", Maas-EtAl:2011:ACL-HLT2011 
      - textrnn and textcnn both perform well on this dataset.
  - task1: a positive/negative sentiment classification of titles in digital forum.
      - in my experiment, textrnn with lstm cells(300) is better than textcnn with 128 filters with sizes range [2,5]. This result maybe my texts from these forums are short, and textcnn performs better in a long paragraph classification most of time. 
  - task3: 300,000 documents and 100 categories, from Minmin, 2017, "Efficient Vector Representation for Documents through Corruption"(Doc2vecC)
      - the author implements this in c++ based on Mikolov's word2vec. 

------

##### # My insights ( for beginners : ;)

- read in the front

  - word2vec/doc2vec uses a unsupervised method generate text representation. More precisely, you average the word vectors in a sentences while using word2vec as representation of words; and doc2vec generates a vector for each sentence/paragraph right after training with a shallow network in word2vec structures and a doc embedding.

  - fasttext is a method that you can get representation of words with supervised information, such as label of sentence. Usually,  you get word vectors which generate for a given task, and  averaging these vectors in the sentence gives you representation/features of the sentence. Finally, you use these features as input to train a classifier/regressor for your task.

  - textcnn/textrnn train a slightly deeper model than word2vec/do2vec. 

    - Kim, etal 2014 is a classical text classification convolution network. you should read this first.

    - textrnn/textrnn generate representations of sentence too.(during trianing)

      - for example, in Kim's paper, after one convolution layer ( 1D-Conv ) and Max pooling layer, you get the representation of a sentence.( a vector in shape [filter_num, 1])
      - in textrnn, representation of sentence can be the last hidden layer of lstm, or a average of all the hidden layers' output of lstm cells.

      ​

- Control the size of your vocabulary

  - for chinese, the common size should be 15w. 
    - usually the initial size of vocabulary size of your corpus might be more than 50w, and you choose a optimal min_count of these word and remove certain stopwords and meaningless words. Then, you get your vocabulary.
  - if you don't control vocabulary size, the training might have problem(OOM). And also, these low-occurrance, meaningless words are a interfere during training your nlp tasks.
  - fixed error phrases/words in your corpus as more as possible. = ;

  ​

- unbalanced class of your nlp tasks

  - using class weights [# ref](https://stackoverflow.com/questions/40198364/how-can-i-implement-a-weighted-cross-entropy-loss-in-tensorflow-using-sparse-sof)

    - there several ways using class_weights in tensorflow

      - use a sample_weight for each batch. ( pass sample weight into tf.losses.sparse_softmax_cross_entropy. ) // I use this way.
      - use class_weights  directly with a weighted logits. ( somebody think it's wrong, [# stackoverflow](https://stackoverflow.com/questions/35155655/loss-function-for-class-imbalanced-binary-classifier-in-tensor-flow) )
      - use [`tf.nn.weighted_cross_entropy_with_logits()`](https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits) to handle unbalanced binary classes.

      ​

- most GPU version tutorial on these shallow neural networks(including mine) are slower than a CPU version (eg. gensim) implementation except for an [adjustment](https://github.com/phunterlau/word2vec_cbow) on CUDA 

  - use the original word2vec(Mikolov) is a better choice. [# word2vec/dav](https://github.com/dav/word2vec)   [# google 官方word2vec 中文注释版/tankle](https://github.com/tankle/word2vec)
  - use gensim with correct parameters.
  - doc2vec is the same.

  ​

------



##### # Reference

- text representations
  - [distributed-representations-of-words-and-phrases-and-their-compositionality_Mikolov_2013](https://arxiv.org/abs/1310.4546.pdf)
  - [Distributed Representations of Sentences and Documents_Thomas_Mikolov_2014](https://arxiv.org/pdf/1405.4053.pdf)
  - [Bag of Tricks for Efﬁcient Text Classiﬁcation_facebook_Mikolov_2015](https://arxiv.org/pdf/1607.01759.pdf)
  - [Skip-Thought Vectors_2015](https://arxiv.org/pdf/1506.06726.pdf)
  - [An Empirical Evaluation of doc2vec with practical insights into document embedding generation_ibm_research_2016](https://arxiv.org/pdf/1607.05368.pdf)
  - [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf)
  - [Efficient Vector Representation for Documents through Corruption_Minmin_2017](https://arxiv.org/pdf/1707.02377.pdf)
- textcnn

  - [Convolutional Neural Networks for Sentence Classification_kim_2014_Emnlp](https://arxiv.org/pdf/1408.5882.pdf)
  - [A Convolutional Neural Network for Modelling Sentences_university of oxford](http://www.aclweb.org/anthology/P14-1062)  # being referenced 1009 times
- textrnn

  - [Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling](https://arxiv.org/pdf/1611.06639.pdf)
  - [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)
- rcnn
  - [Recurrent Convolutional Neural Networks for Text Classification_AAAI_2015](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745)  [#ref](https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier) #Institute of Automation, Chinese Academy of Sciences
  - [Learning text representation using recurrent convolutional neural network with highway layers_2016](https://arxiv.org/abs/1606.06905)



---

##### # Learning resources

- [Word2Vec源码解析](http://www.cnblogs.com/neopenx/p/4571996.html)   @[Physcalの大魔導書/cnblog](http://www.cnblogs.com/neopenx/)


- [google 官方word2vec 中文注释版/tankle](https://github.com/tankle/word2vec)

- [word2vec/dav](https://github.com/dav/word2vec)

- [C++ implement of Tomas Mikolov's word/document embedding](https://github.com/hiyijian/doc2vec)

- [Doc2VecC from the paper "Efficient Vector Representation for Documents through Corruption"/mchen24](https://github.com/mchen24/iclr2017)

- [Tutorial for Sentiment Analysis using Doc2Vec in gensim/linanqiu](https://github.com/linanqiu/word2vec-sentiments)

- ["Distributed Representations of Sentences and Documents" Code?/google forum](https://groups.google.com/forum/#!msg/word2vec-toolkit/Q49FIrNOQRo/J6KG8mUj45sJ)

  ​

------



##### **# Requirements**

( recommend [Google Colab](https://colab.research.google.com) with free 12-hour GPU, Tesla K80, [tutorial](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d) here )

( my .ipynb online: https://drive.google.com/file/d/1WPWM103comn-1kyGv5FXGO-ZC2Lt7ChH/view?usp=sharing )

- 8GB memory, with Nvidia GeForce MX150, compute capability: 6.1
- python3.5
- tensorflow-gpu 1.4.0
- numpy 1.13.1
- jieba 0.39
- t-SNE