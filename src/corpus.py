'''
Created on Jun 17, 2016

@author: pedrom
'''
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

class Corpus(object):

    def __init__(self, sentiment_corpus_file):
        self.vocab = None
        self.sent_labels = []
        self.W_D_matrix = None
        self.docs_L0 = []
        self.docs_L1 = []
        i = 0
        all_txt = []
        with open(sentiment_corpus_file, encoding="utf8") as f:
            lines = f.readlines()
            for lin in lines:
                label, txt = self.process_lin(lin)
                all_txt.append(txt)
                self.sent_labels.append(label)
                if label == "0":
                    self.docs_L0.append(i)
                else:
                    self.docs_L1.append(i)
                i += 1
        vectorizer = CountVectorizer(analyzer = "word", strip_accents = "unicode", stop_words = stopwords.words("english"), max_features = 10000)
        self.W_D_matrix = vectorizer.fit_transform(all_txt)
        self.vocab = vectorizer.vocabulary_
        self.V = len(self.vocab)
        
    def process_lin(self, lin):
        split_lin = lin.split("\t")
        return split_lin[1], split_lin[2] 