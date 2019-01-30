import nltk
import csv
from PrepareInputANN import process
f = open("Textcorpus.csv")
f = list(csv.reader(open("Textcorpus.csv")))
corpus = []
for i in f:
    if (i[1] == "Arithmetic" and (
                            i[2] == "Addition" or i[2] == "Subtraction" or i[2] == "Division" or i[
                2] == "Multiplication")):
        processed = process(i[0])
        corpus.append([processed[0], i[1], i[2], processed[1]])
print(corpus)
strings = [row[0] for row in corpus]
print("stirngs: ", strings)
grammar = r"""
  NP:
    {<.*>+}          # Chunk everything
    }<VBD|IN>+{      # Chink sequences of VBD and IN
  """
cp = nltk.RegexpParser(grammar)
for i in range(0,1):
    ans = (cp.parse(nltk.pos_tag(strings[i])))
ans.draw()
print(cp)