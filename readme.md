### Text Classifications
------
##### # Contains:

1. implementation of PV-DM, PV-DBoW models using tensorflow  structure (including inferencing new paragraph/sentence). My goal is to learn how to generate good vectors for sentences, paragraphs. Probably a good start point beginners who want to represent texts using neural model.
2. text_cnn, text_rnn model implementations referenced from [brightmart/text_classification](brightmart/text_classification) with my personal comments.
3. fasttext tutorial

##### # todo

- evaluations of text representation

  - task0: the second experiment as benchmark in Mikolov. etal:2014, using the IMDB dataset.
  - IMDB (Large Movie Review Dataset) from "Learning Word Vectors for Sentiment Analysis", Maas-EtAl:2011:ACL-HLT2011 


  - task1: a positive/negative sentiment classification of titles in digital forum.
  - task2: similarity calculations between paragraph vector and a label vector.
  - task3: 300,000 documents and 100 categories, from Minmin, 2017, "Efficient Vector Representation for Documents through Corruption"(Doc2vecC)

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

  - [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

- textrnn

  - [Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling](https://arxiv.org/pdf/1611.06639.pdf)
  - [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)

  ​

  ​

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

( most GPU version tutorial on these shallow neural networks(including mine) are slower than a CPU version (eg. gensim) implementation except for an [adjustment](https://github.com/phunterlau/word2vec_cbow) on CUDA )

- 8GB memory, with Nvidia GeForce MX150, compute capability: 6.1
- python3.5
- tensorflow-gpu 1.4.0
- numpy 1.13.1
- jieba 0.39
- t-SNE