from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import sgd
from sklearn.cross_validation import StratifiedKFold
from decimal import Decimal
import operator
from fractions import Fraction
import nltk
from nltk.tokenize.moses import MosesDetokenizer
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

def process(s):

    #print("processing: " + s)
    nums = []
    words = nltk.word_tokenize(s)
    numcounter = 1
    tagged = nltk.pos_tag(words)
    for i in range(0, len(words)):
        if(i<len(words) and (words[i]=="Mrs." or words[i]=="Mr.")):
            del words[i]
    for i in range(0, len(words)):
        if(tagged[i][1]=="NNP"):
            words[i] = "NAME"
    # processing parantehtical numbers eg: twenty-two (22)
    for i in range(0, len(words)): # for each word in the string
        if (i<len(words) and words[i][0] == "("): # if the word is a parantehtical
            nums.append(int(words[i+1])) #add whats inside the parentheses (a number) to the numbers
            del words[i+2]
            del words[i] # delete the instance of the paranthetical
            del words[i-1]
            words[i-1] = "N"+str(numcounter) #replace the word before the parenthetical (the written-out number) with the numbertext
            numcounter+=1

    #processing fractions eg: 3/2
    #words = str.split(s)
    for i in range(0, len(words)):
        if (i<len(words) and "/" in words[i]):
            if (words[i-1][0].isdigit()):
                nums.append(Fraction(int(words[i-1])*int(words[i].split("/")[1])+int(words[i].split("/")[0]),int(words[i].split("/")[1])))
                del words[i]
                words[i-1] = "N" + str(numcounter)
            else:
                nums.append(Fraction(int(words[i].split("/")[0]),int(words[i].split("/")[1])))
                words[i] = "N" + str(numcounter)
            numcounter+=1

    #processing normal numbers eg: 253, 12.2
   # print(words)
    for i in range(0, len(words)):
       # print(words[i])
        # if (words[i][len(words[i])-1].isalnum()==False):
        #     words[i] = words[i][:-1]
        if (words[i][0].isdigit() and words[i][len(words[i])-1].isdigit()):
            #print(words[i])
            nums.append(Decimal(words[i].replace(',', '')))
            words[i] = "N"+str(numcounter)
            numcounter+=1

    #processing money eg: $3.50
    # for i in range(0, len(words)):
    #     if (words[i][0]=="$"):
    #         s = s.replace(words[i],"N"+str(numcounter) + " dollars",1)
    #         nums.append(Decimal(words[i].replace('$','').replace(',','')))
    #         numcounter+=1
    units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    for i in range(0, len(words)):
        for j in range(0, len(units)):
            if (words[i].lower()==units[j] and len(nums)<2):
                words[i] = "N" + str(numcounter)
                nums.append(j)
                numcounter += 1
    #words = nltk.word_tokenize(s)
    # for i in range(0, len(words)):
    #     if (words[i].lower()=="older"):
    #         s = s.replace(words[i],"more")
    #     if (words[i].lower() == "younger"):
    #         s = s.replace(words[i], "fewer")
    #print(s)
    ##print(nums)
    return [words,nums]
def popularPhrases(strings, nWords, minRepeats):
    words = dict()
    phrasesinsamesentence = set()
    #print("thastrings into account:",strings)
    #pdb.set_trace()
    for problem in strings:
        phrasesinsamesentence.clear()
        for j in range(0, len(problem)-nWords+1):
            phraseList = [problem[j]]
            #print(phrase)
            for k in range(1, nWords):
                phraseList.append(problem[j+k])
            detoken = MosesDetokenizer()
            phrase = detoken.detokenize(phraseList,return_str=True)
            #print(phrase)
            if (phrase not in phrasesinsamesentence) and (phrase[0].isdigit() is False) and phrase[0] != "(":
                phrasesinsamesentence.add(phrase)
                if (phrase not in words):
                    words[phrase]=1
                else:
                    words[phrase] = words[phrase]+1
    sortedList = (sorted(words.items(), key=operator.itemgetter(1)))
    #print(sortedList)
    ct = len(sortedList)-1
    ans = []
    while (sortedList[ct][1]>=minRepeats and ct>=0):
        #print(ct)
        ans.append(sortedList[ct][0])
        ct-=1
    return ans

def initializeWordsList(strings):
    #for arithmetic, 9,5,5,4,3
    wdlst = (popularPhrases(strings, 1, 20) + popularPhrases(strings, 2, 10) + popularPhrases(strings, 3, 9) + popularPhrases(strings, 4, 8) + popularPhrases(strings,5,7))
    #pdb.set_trace()
    print("wordlist: ", wdlst)
    return wdlst

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

def makemap(strs, wordslist):
    #pdb.set_trace()
    ans = []
    for i in strs:
        detoken = MosesDetokenizer()
        str = detoken.detokenize(i, return_str=True)
        #print(str)
        onehotmap = []
        for j in wordslist:
            if j in str:
                onehotmap.append(1)
            else:
                onehotmap.append(0)
        ans.append(onehotmap)
    return ans

def load_ANN():
    corpus = []
    f = list(csv.reader(open("Textcorpus.csv")))
    for i in f:
        if (i[1] == "Arithmetic" and (
                        i[2] == "Addition" or i[2] == "Subtraction" or i[2] == "Division" or i[2] == "Multiplication")):
            processed = process(i[0])
            corpus.append([processed[0], i[1], i[2], processed[1]])
    strings = [row[0] for row in corpus]
    print("stirngs: ", strings)
    # pdb.set_trace()
    wordslist = initializeWordsList(strings)
    # pdb.set_trace()
    print(wordslist)
    X = makemap(strings, wordslist)
    X = np.array(X)
    Y = np.array(getanswers(corpus))
    print(X.tolist()[0].count(1))
    print("answers", getanswers(corpus))
    print("wordslist", wordslist)
    return X,Y,wordslist
if __name__ == '__main__':
    print(process("There are twelve (12) birds on the fence. Eight (8) more birds land on the fence. How many birds are on the fence?"))
    load_ANN()