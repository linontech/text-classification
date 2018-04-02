#### Text Representations
------
this repository contains implementation of PV-DM, PV-DBoW models, including inferencing new paragraph/sentence, using tensorflow framwork. My goal is to learn how to generate good vectors for sentences, paragraphs. Probably a good start point beginners who want to represent texts using neural model.

1. **evaluation**

- task0: the second experiment as benchmark in Mikolov. etal:2014, using the IMDB dataset.
  - IMDB (Large Movie Review Dataset) from "Learning Word Vectors for Sentiment Analysis", Maas-EtAl:2011:ACL-HLT2011 
- task1: a positive/negative sentiment classification of titles in digital forum.
- task2: similarity calculations between paragraph vector and a label vector.
- task3: 300,000 documents and 100 categories, from Minmin, 2017, "Efficient Vector Representation for Documents through Corruption"(Doc2vecC)

2. **requirements**:

   ( recommend [Google Colab](https://colab.research.google.com) with free 12-hour GPU, Tesla K80, [tutorial](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d) here )

   ( my .ipynb online: https://drive.google.com/file/d/1WPWM103comn-1kyGv5FXGO-ZC2Lt7ChH/view?usp=sharing )

   ( most GPU version tutorial on these shallow neural networks(including mine) are slower than a CPU version (eg. gensim) implementation except for an [adjustment](https://github.com/phunterlau/word2vec_cbow) on CUDA )

- 8GB memory, with Nvidia GeForce MX150, compute capability: 6.1
- python3.5
- tensorflow-gpu 1.4.0
- numpy 1.13.1
- jieba 0.39
- t-SNE

------

(1) [distributed-representations-of-words-and-phrases-and-their-compositionality_Mikolov_2013](https://arxiv.org/abs/1310.4546.pdf)

(2) [Distributed Representations of Sentences and Documents_Thomas_Mikolov_2014](https://arxiv.org/pdf/1405.4053.pdf)

(3) [Bag of Tricks for Efﬁcient Text Classiﬁcation_facebook_Mikolov_2015](https://arxiv.org/pdf/1607.01759.pdf)

(4) [Skip-Thought Vectors_2015](https://arxiv.org/pdf/1506.06726.pdf)

(5) [An Empirical Evaluation of doc2vec with practical insights into document embedding generation_ibm_research_2016](https://arxiv.org/pdf/1607.05368.pdf)

(6) [Efficient Vector Representation for Documents through Corruption_Minmin_2017](https://arxiv.org/pdf/1707.02377.pdf)

>  