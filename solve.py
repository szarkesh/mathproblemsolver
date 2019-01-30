import nltk
def main():
    #nltk.download_shell()
    nltk.help.upenn_tagset("MD")
    a = [2,3]
    a[1] = 5
    print(a)
    f = open("Textcorpus.csv")

    ans = nltk.word_tokenize("I have five pancakes.")
    tagged = nltk.pos_tag(ans)
    grammar = r"""
      NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
          {<NNP>+}                # chunk sequences of proper nouns
    """
    cp = nltk.RegexpParser(tagged)
    result = cp.parse(ans)
    print(result)
    parser = nltk.parse.api.ParserI()
    print(parser.parse(ans))
    print("2", ans)
    questions = f.readlines()
    for i in range(0, 10):
        tok = nltk.word_tokenize(questions[i])
        ans = nltk.pos_tag(tok)
        print(ans)
    print(2)

if(__name__ == "__main__"):
    main()