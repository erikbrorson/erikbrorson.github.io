---
title: "Tweet2vec 1: The Word2Vec model"
layout: post
---

## Motivation

In the earlier post we quantified the tweets as bag of word-vectors. We are now going to look at a more sophisticated way of vectorising tweets based on word2vec vectors. These vectors are described in a [paper](https://arxiv.org/abs/1301.3781) by Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Each word is represented by an n-dimensional vector in such a way that the distance between two words that are semantically similar is shorter that two words that are semantically less similar. For example, if we had two word pairs (King, Man) and (King, Woman) we want the distance between the words in the first pair to be smaller than the distance between the words in the second pair.

These word embeddings are learned using a neural network that predicts the probability of one word being close to another word in a text. We will cover these models and how to implement them in Tensorflow in a future blog post.

The rest of this post is structured as follows. We start off by looking at the word embeddings we will use in the later tweet2vec processing and through a couple of examples get an intuition of how they work. 

## The pretrained Google word2vec embeddings

In order to learn word2vec embeddings you need to have a very big corpus of texts and substantial computational resources. Lucky for us, there are easier ways to get the embeddings without paying hundreds of dollars in EC2 costs. We are going to use a pretrained model released by Google that can be [downloaded here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit). How to use this model is adequately described in [this blogpost](http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/).

To get to the word embedding vectors we need to put the whole model in memory. We are going to use the gensim package. The code belows imports the model into memory.


```python
import gensim

# import the word2vec model to memory
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    '/Volumes/PICS/pyProject/GoogleNews-vectors-negative300.bin.gz', 
    binary = True)
```




The file containing the word2vec model is 1.5 gb, so make sure you have enough memory available. It takes a few minutes to load.

### Let's take it for a spin

Let's run some code. We want to find whether or not the similarities behave in such a way as described earlier. We want to see if the similarity between the words *man* and *king* is bigger than the similarity between *woman* and *king*. We also want to see the dimensions of the word vectors.


```python
# first, grab the vector
king = word2vec_model.wv['king']

# print its dimensions
print('The dimension of one word vector is: ' + str(king.shape))

# check the similarity between king and man and compare it 
#   to king and woman
man_king_sim   = word2vec_model.wv.similarity('king', 'man')
woman_king_sim = word2vec_model.wv.similarity('king', 'woman')

print('The similarity between the words \'king\' and \'man\' is ' + str(man_king_sim))
print('The similarity between the words \'king\' and \'woman\' is ' + str(woman_king_sim))
```

```
The dimension of one word vector is: (300,)
The similarity between the words 'king' and 'man' is 0.229426704576
The similarity between the words 'king' and 'woman' is 0.12847973557
```



As we see, it is as we predicted. And we also see that our are of length 300, neat! Next: let's see how we can use these word vectors to go from word2vec to tweet2vec! 

## Tweets2vec

Let's say that we want to use these vectors for something more useful, for example as inputs to a machine learning model? If our data would be tweets, we would need to find a way to represent full sentences as vectors an not only individual words. There are a bunch of different ways to do this, for example we could just take the element wise mean of the vectors in a sentence. We find a naive implementation of this idea below in the function text2vec.


```python

# some dependencies
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

punctuation_traslator = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))


def text2vec(text):
    """
    takes a string as input, returns a vector representation 
        calculated by taking the mean of the individual word
        embedding vectors.

    ARGS:
    - text 
    
    Returns:
    - A vector representation of the 
    """
    text = text.lower()
    text = text.translate(punctuation_traslator)
    text = nltk.word_tokenize(text)
    filtered_sentence = [w for w in text if not w in stop_words]
    i = 1
    vector_representation = np.zeros((1,300))

    for word in filtered_sentence:
        try: 
            vector_representation = vector_representation + word2vec_model.wv[word]
            i = i + 1
        except KeyError:
            i = i
    vector_representation = np.divide(vector_representation, i)
    return(vector_representation)
```



Let's see what the vector for the sentence *I really want to have some cheese* looks like.


```python
# first, we store our sentence in a variable
sentence = 'I really want to have some cheese'

# second, we process our vector representation of the sentence
sentence_in_vector_form = text2vec(sentence)

# now, let's print the first 10 elements of the vector
print(sentence_in_vector_form[0:10])
```

```
[[-0.01306152  0.04898071  0.01190186  0.18371582 -0.03271484
0.05787659
   0.10913086 -0.07702637  0.01344299  0.03594971 -0.04655457
-0.11230469
  -0.03070068 -0.07617188 -0.16723633  0.1484375  ...]]
```


