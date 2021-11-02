# COVID_19_Word_Vectors
This repository provides the word embeddings learned from COVID-19 related Twitter datasets.

The learned word embeddings have dimensions of 300.

The word embeddings can be used for training neural network models. 
There is the keyed-vector word embeddings (COVID19_300d_vectors.kv) that have the COVID-19 tweets vocabs and their respective learned vectors
The keyed vectors can be used using the gensim python library as follows:

from gensim.models import KeyedVectors
wv = KeyedVectors.load('COVID19_300d_vectors.kv')
vector = wv["england']

To continue training the COVID-19 tweets word embeddings with more COVID-19 related tweets, use the COVID-19 model (COVID19_300d_word2vec.model) dataset.
Use the following gensim python code to load the model and continue training the word embeddings;

from gensim.models import WordeVec
model = Word2Vec.load("COVID19_300d_word2vec.model") # Load the COVID-19 tweets word embeddings model
more_sentences = [
    ['Advanced', 'users', 'can', 'load', 'a', 'model',
     'and', 'continue', 'training', 'it', 'with', 'more', 'sentences'],
] # sample text corpus
model.build_vocab(more_sentences, update=True)
model.train(more_sentences, total_examples=model.corpus_count, epochs=model.epochs)
