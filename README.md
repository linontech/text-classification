#### implementation of "2014_Distributed Representations of Sentences and Documents_Thomas_Mikolov" using tensorflow

------

this repository contains implementation of PV-DM, PV-DBoW models, including inferencing new paragraph/sentence, using tensorflow framwork.

evaluation:

- task0: the second experiment as benchmark in Mikolov. etal:2014, using the IMDB dataset.
  - IMDB (Large Movie Review Dataset) from "Learning Word Vectors for Sentiment Analysis", Maas-EtAl:2011:ACL-HLT2011 
- task1: a positive/negative sentiment classification of titles in the digital forum .
- task2: similarity calculations between paragraph vector and a label vector.
- task3: 300,000 documents and 100 categories, from "Efficient Vector Representation for Documents through Corruption"(Doc2vecC)

requirements:

- 8GB memory

- python3.5

- tensorflow-gpu 1.4.0

- numpy 1.13.1

- jieba 0.39

- t-SNE

  ​
