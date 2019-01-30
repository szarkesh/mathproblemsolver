from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.cross_validation import StratifiedKFold
from decimal import Decimal
import operator
from fractions import Fraction
import numpy
from keras.datasets import imdb
import sklearn
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from keras.layers import Dropout
from keras.preprocessing import sequence
import PrepareInputANN
import PrepareInputLSTM
import numpy as np
import pdb
import csv
def defineModelandOptimizer(networkType, top_words=0, max_length=0, ninputs=785):
    if (networkType=='LSTM'):
        embedding_vector_length = 32
        max_review_length = max_length
        model = Sequential()
        model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
        model.add(LSTM(100,return_sequences=True))
        model.add(LSTM(100, input_shape=(max_review_length, top_words)))

        model.add(Dropout(0.3))
        #model.add(Dense(12, activation='sigmoid'))
        model.add(Dense(4, activation='softmax'))
        opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    if (networkType=='ANN'):
        model = Sequential()
        model.add(Dense(12,input_dim=ninputs,activation='sigmoid'))
        model.add(Dropout(0))
        model.add(Dense(4, activation='softmax'))
        opt = SGD(lr=0.1)
        model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
def solve(nums, type):
    case = type.index(max(type))
    indexes = [0, 1, 2, 3]
    #print('case: ', case)
    if (len(nums)==2):
        if (case == 0):
            return nums[0]+nums[1]
        if (case==1):
            return max(nums)-min(nums)
        if(case ==2):
            return nums[0]*nums[1]
        if (case==3):
            return max(nums)/min(nums)

def main(demoMode, networkType):
    X = []
    Y = []
    topWords = 150
    maxProblemLength = 53

    #getting the data in the correct format, depending on which model we're using
    if (networkType=='LSTM'):
        X,Y,words = PrepareInputLSTM.load_data(top_words=topWords)
        X = sequence.pad_sequences(X, maxlen=maxProblemLength)
    else:
        X,Y,words = PrepareInputANN.load_ANN()


    #in demo mode, we train teh model on everything and then wait for the user to input
    if (demoMode):
        model = defineModelandOptimizer(networkType,top_words=topWords,max_length=maxProblemLength, ninputs=len(X[0]))
        print("X is:", X)
        model.fit(X, to_categorical(Y, 4), epochs=300, batch_size=62, verbose=0)
        while (True):
            problem = input("Poly> What's your problem? ")
            print("Poly> Hmm...I'm thinking...")
            if (networkType == 'LSTM'):
                processed = PrepareInputLSTM.process(problem)
                X_test = np.array(PrepareInputLSTM.words_2_ints([processed[0].strip().split()],words))
                X_test =  sequence.pad_sequences(X_test, maxlen=maxProblemLength)
            if (networkType == 'ANN'):
                processed = PrepareInputANN.process(problem)
                X_test = np.array(PrepareInputANN.makemap([processed[0]],words))
                print(processed)
                print(X_test)
            nums = processed[1]
            problemType = list(model.predict(X_test))[0]
            print(problemType)
            problemTypes = ["Addition", "Subtraction", "Multiplication", "Division"]
            #pdb.set_trace()
            ans = problemTypes[problemType.tolist().index(max(problemType))]
            print("Poly> Looks like this is a",ans, "problem")
            print("Poly> The numbers in the problem are ", nums[0], "and", nums[1])
            print("Poly> So the answer is...", solve(nums,problemType.tolist()))

    #in testing mode, we use kfold cross validation
    else:
        print("Number of features", len(X[0]))
        true_classes = []
        predicted_classes = []
        if (networkType == "NaiveBayes"):
            #model = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
            model = svm.SVC()
            scores = sklearn.model_selection.cross_val_score(model, X, Y, cv=5)
            print(scores)
        else:
            kfold = StratifiedKFold(Y, n_folds=10, shuffle=True)
            for train, test in kfold:
                if (networkType=="LSTM"):
                    #print(train,test)
                    #pdb.set_trace()
                    model = defineModelandOptimizer(networkType, top_words=topWords, max_length=maxProblemLength)
                    model.fit(X[train], to_categorical(Y[train], 4), epochs=100, batch_size=100, verbose=1)
                if (networkType=="ANN"):
                    model = defineModelandOptimizer(networkType, ninputs=len(X[0]))
                    model.fit(X[train], to_categorical(Y[train], 4), epochs=500, batch_size=100, verbose=0)
                #pdb.set_trace()
                scores = model.evaluate(X[test], to_categorical(Y[test],4), verbose=0)
                finalloss = model.evaluate(X[train], to_categorical(Y[train],4), verbose=0)
                #print(X[test], model.predict_classes(X[test], len(X[test])))
                true_classes += Y[test].tolist()
                predicted_classes += model.predict_classes(X[test], len(X[test])).tolist()
                print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
                print("Validation Accuracy: " + str(finalloss[1]*100))
            ct= 0
            for i in range(0, len(true_classes)):
                if (true_classes[i]==predicted_classes[i]):
                    ct+=1
            print(ct, "out of", len(true_classes), "guesses are correct. That's an accuracy of", float(ct)/float(len(true_classes)))
            print("Number of features", len(X[0]))
if (__name__ == "__main__"):
    main(True, 'ANN')