import nltk
import PrepareInputANN
import csv
import numpy as np
import operator
import math
from PrepareInputANN import process
def prepareAlgebraInput():
    ans = list(csv.reader(open("Algebra.csv")))
    print(ans[0])
    problems = []
    answers = []
    classes = []
    counter=0
    problemtypes = dict()
    problemtypelist = []
    for i in range(2,len(ans)):
        problem = ans[i]
        if(len(problem[0])>0):
            #print(problem)
            problems.append(process(problem[0]))
            if(len(problem[2])!=0):
                answers.append([float(problem[1]),float(problem[2])])
            else:
                answers.append([float(problem[1])])
            problemtype = problem[4]+" "+problem[5]
            problemtypelist.append(problemtype)
            if(problemtype not in problemtypes):
                problemtypes[problemtype]=1
            else:
                problemtypes[problemtype]+=1
            if (problem[4]=="m + n = a" and problem[5]=="m - n = b"):
                classes.append(1)
            else:
                classes.append(0)
    print("Problems List", sorted(list(problemtypes.values()),reverse=True))
    sorted_dict = sorted(problemtypes.items(), key=operator.itemgetter(1), reverse=True)
    realclass = []
    for i in range(0, len(answers)):
        for j in range(0, len(answers[i])):
            if(answers[i][j]<0 or math.floor(answers[i][j])!=answers[i][j]):
                print("RUH ROH")
    for i in range(0, len(problemtypelist)):
        #print(len(sorted_dict))
        boo = False
        for j in range(0, len(sorted_dict)):
            if(problemtypelist[i] == sorted_dict[j][0]):
                realclass.append(j+1)
                boo=True
        if boo==False:
            realclass.append(0)
    print(len(realclass),realclass)
    print(sorted_dict)
    print(counter)
    print(problemtypes)
    questions = [row[0] for row in problems]
    print(problems)
    print(classes)
    print(answers)
    words = PrepareInputANN.initializeWordsList(questions)
    map = PrepareInputANN.makemap(questions,words)
    return np.array(map), np.array(realclass), words

if(__name__=="__main__"):
    prepareAlgebraInput()