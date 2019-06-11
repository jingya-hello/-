# Contest 1 DL_comp1_50_report.ipynb

#Project 1: Predicting Appropriate Response

# Student ID, name of each team member
- 104062101 劉芸瑄
- 104000033 邱靖雅
- 104062226 王科鈞
- 104062315 李辰康

# Private score: 4th

# Preprocess data
## cleaning
We remove all space and Unicode punctuation.
```python
import re
def _delete_(string):
    string = string.replace('「','').replace('」','')
    return re.sub(r"!|:|\.|\+|\*|\-|\/|~|#|～|，|,", "", string)
```
## Cleaning the question and answer
```python
stop_words =  set([' ', 
                   '你', '我', '自己', '人',
                   '的', '啦', '了', '嗎', '嘛', '耶', '啊',
                   '不', '是', '是不是', '有沒有',
                   '反而', '這', '又', '這麼', '都', '才', '應該', '其實', '因為', '你還', '該'
                  ])
```
We only clean out some stop words in the question: we find out that because the answers in the test case are usually short sentence therefore cleaning out stopword may make the sentence even shorter and missing information.

# How did you build the classifier (model, training algorithm, special techniques, etc.)?
## Word Embedding
Since vector space models can be used to represent words in a continuous vector space, we then need to make each word into word embeddings after preprocessing all the training data. In the competition, we use word2vec model as our word embedding model.

### word2vec

Word2vec is a method that learns how words are used in a particular text corpus. It outputs each word as a numeric vector by mapping all words in the given corpus into the vector space.

#### How to use it?
Words with similar semantics are put close in the vector space since they usually have a similar context in the corpus.

### Implementation
There exists a model in the gensim package that can help us to build word2vec model
```python
from gensim.models import Word2Vec
```
We need to feed some parameters into the word2vec model.

```python
# document: the training data
# size: dimensionality of the word vectors.
# window: maximum distance between the current and predicted word within a sentence.
# min_count: ignores all words with the total frequency lower than this.
model = Word2Vec(document, size=100, window=5, min_count=0,workers=4)

# train model
model.train(document, total_examples=len(document), epochs=100)

# save model
model.save("word2vec.model")
```
### Experiments that we done
#### parameters:
* size (int, optional) – Dimensionality of the word vectors.
* window (int, optional) – Maximum distance between the current and predicted word within a sentence.
* min_count (int, optional) – Ignores all words with total frequency lower than this.
* epoch 
```python 
model.train(document, total_examples=len(document), epochs=150) 
```

#### we have tried 
* size: $[100,125,250,512,1024]$
* window: $[5,10,15]$
* min_count: $[0]$
* epoch: $[0,100,150]$

#### the best combination of parameter
* $size=100$
* $window=5$
* $min\_count=0$
* $epoch=150$

When we increase the size values, we get worse performance.
> **The possible reason for this result:**
> bigger size values require more training data, but we don't have so much data.

When we increase the window values, we get worse performance.
> **The possible reason for this result:**
> sentences in our data are short, so bigger window values may make some irrelevant words become too close.

When we increase the epoch values, we get BETTER performance.
> **The possible reason for this result:**
> As the number of epochs increases, the number of times the weight is changed in the neural network and the curve goes from underfitting to optimal to overfitting curve.


#### Find similarity between two sentence using word2vec
we have try 2 solutions to evaluate the similarity between 2 sentence:
1. use the build in function n_similarity
the gensim word2vec model exist a function n_similarity that can culculate the similarity of two sentence
```python
def n_similarity(self, ws1, ws2):
    v1 = [self[word] for word in ws1]
    v2 = [self[word] for word in ws2]
    return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))
```
2. handcraft similarity calculation
use the build in method model.wv.simlarity to caluculate similarity between each two pairs of words in question and answer(pseudo code)
```python
score = -1000
for word1 in sentence1:
    for word2 in sentence2:
        score += (np.power(model.wv.similarity(word1, word2), 3)
```
We find out that the second solution performs better than than build-in solution. Therefore, we create a scoring system to calculate scores bwtween the question and each answers.


## TFIDF

### What is TFIDF

Tf-idf is an abbreviation for frequency-inverse document frequency, and the tf-idf weight is composed by word is to a document in a collection or corpus

tf-idf weight is composed of two terms : 
1. the first computes the normalized Term Frequency (TF) which is the number of times a word appears in a document then divided by the total number of words in that document.
2. the second term is the Inverse Document Frequency (IDF) which computers the log of the number of documents in the corpus divided by the number of documents where the specific term appears
![](https://i.imgur.com/MpNDn6E.png)

### Implementation

There exists model in the gensim package that can help us to build Tfidf model
```python
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
```

We first use the training data which is the 8 programs to build up a TFIDF model
```python
# create dictionary from the document
dictionary = corpora.Dictionary(document.split() for document in open("tfidf_dict.txt", encoding='utf-8'))
# get all texts from the document
texts = [[word for word in document.split()] for document in open("tfidf_dict.txt", encoding='utf-8')]
# build corpus by converting document into the bag-of-words (BoW) format 
corpus = [dictionary.doc2bow(text) for text in texts]
# build Tfidf model
tfidf_model = TfidfModel(corpus)
```
Then, we construct a function to get a list of words as input then return each word and its tfidf value in the TFIDF model. In our model, we only get the top three important words based on the result of experiment.

```python
'''
getTfidfWeight
@param testlist : a list of words (a sentence) that you want to compute tfidf value
@return weight_word : a list of the top three important words in the format tuple (word, tfidf_value) 
'''
def getTfidfWeight(testList):
    weight_word = []
    # translate tested sentence into bag of word
    test_corpus = dictionary.doc2bow(testList)
    
    #find out all the importance of word in a sentence by feeding the corpus into the Tfidf model
    test_corpus_tfidf = tfidf_model[test_corpus]
    # sort words by the tfidf value
    test_corpus_tfidf = sorted(test_corpus_tfidf, key=lambda item: item[1], reverse=True)
    
    # return the highest 3 important words
    rangeOfLoop = min(3, len(test_corpus_tfidf_1))
    for i in range(rangeOfLoop):
        index, value = test_corpus_tfidf_1[i]
        weight_word.append((dictionary.get(index), value))
    return weight_word
```
### Experiment
#### Tfidf-Weighted Word2Vec
Since we need to control the weight of words in a sentence, we test 2 solutions to fugure out how to weight each word:
1. Calculate the ratio of each word in a sentence : 
    For example : given a list of words with tfidf value (0.9,0.2,0.1,0.01), multiply 0.9/(0.9+0.3+0.1+0.01) to the first word's word2vec vector to higher its importance.
3. Given the top 3 important words a fixed weight : 
    For example : mutiply (1.5, 1.3, 1.1) to the top 3 word's word2vec vector

After some experiments, we find out that the second solution performs better than the first one by giving the hyperparameter (1.5, 1.3, 1.1) to the first, second, third highest words, respectively.

# Conclusions (interesting findings, pitfalls, takeaway lessons, etc.)?
From this competition, We learn how to use word2vec model, doc2vec model, TFIDF method, SIF method.