from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import sgd
from sklearn.cross_validation import StratifiedKFold
from decimal import Decimal
import operator
from fractions import Fraction
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.preprocessing import sequence
import numpy as np
import pdb
import csv
from PrepareInputANN import process
def makeWordDict(wordarray, nMostPopularWords):
    wordcount = dict()
    for sentence in wordarray:
        for word in sentence:
            if word not in wordcount:
                wordcount[word]=1
            else:
                wordcount[word]+=1
    counts = list(wordcount.values())
    keys = list(wordcount.keys())

    sorted_idx = numpy.argsort(counts)
    wordcount = sorted(wordcount.items(), key=operator.itemgetter(1))
    print(wordcount)

    #print(sorted_idx)

    worddict = dict()
    count = len(wordcount)
    if(nMostPopularWords<len(wordcount)):
        count = nMostPopularWords
    for i in range(0, count-2):
        worddict[wordcount[len(wordcount)-i-1][0]] = i+1
    #print("worddict:",worddict)
    #pdb.set_trace()
    #print (numpy.sum(counts), ' total words ', len(keys), ' unique words')

    return worddict
def words_2_ints(wordarray, worddict):
    intarray = []
    for sentence in wordarray:
        list = []
        for word in sentence:
            list.append(worddict.get(word, -1)+1)
        intarray.append(list)
    print("intarray:",intarray)
    return intarray
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
    permutation = numpy.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def getanswers(corpus):
    answers = []
    for i in corpus:
        if i[2] == "Addition":
            answers.append(0)
        if i[2] == "Subtraction":
            answers.append(1)
        if i[2] == "Multiplication":
            answers.append(2)
        if i[2] == "Division":
            answers.append(3)
    return answers

def load_data(top_words):
    corpus = []
    answers = []
    f = list(csv.reader(open("Textcorpus.csv")))
    for i in f:
        if (i[1] == "Arithmetic" and (
                        i[2] == "Addition" or i[2] == "Subtraction" or i[2] == "Division" or i[2] == "Multiplication")):
            processed = process(i[0])
            corpus.append([processed[0], i[1], i[2], processed[1]])
            if i[2] == "Addition":
                answers.append(0)
            if i[2] == "Subtraction":
                answers.append(1)
            if i[2] == "Multiplication":
                answers.append(2)
            if i[2] == "Division":
                answers.append(3)
    corpus, answers = shuffle_in_unison(numpy.array(corpus), numpy.array(answers))

    sentences = [s[0].strip().split() for s in corpus]
    print("sentences",sentences)
    wordDict = makeWordDict(sentences, top_words)
    X = words_2_ints(sentences, wordDict)


    Y = getanswers(corpus)
    print("X:", X)
    print("Y:",Y)

    return X,np.array(Y), wordDict
#if(__name__=="__main__"):
#    load_data()
