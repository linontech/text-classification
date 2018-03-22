#### implementation of "2014_Distributed Representations of Sentences and Documents_Thomas_Mikolov" using tensorflow

------

this repository contains implementation of PV-DM, PV-DBoW models, including inferencing new paragraph/sentence, using tensorflow framwork.

evaluation:

- task0: the second experiment as benchmark in Mikolov. etal:2014, using the IMDB dataset.
  - IMDB (Large Movie Review Dataset) from "Learning Word Vectors for Sentiment Analysis", Maas-EtAl:2011:ACL-HLT2011 
- task1: a positive/negative sentiment classification of titles in digital forum .
- task2: similarity calculations between paragraph vector and a label vector.
- task3: 300,000 documents and 100 categories, from Minmin, 2017, "Efficient Vector Representation for Documents through Corruption"(Doc2vecC)

requirements: ( recommend [Google Colab](https://colab.research.google.com) with free 12-hour GPU, Tesla K80, [tutorial](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d) here )

- 8GB memory	

- python3.5

- tensorflow-gpu 1.4.0

- numpy 1.13.1

- jieba 0.39

- t-SNE

  ​

