'''
Created on Jun 17, 2016

@author: pedrom
'''
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from dist_sampler import sample_bernouli, sample_multinomial
import numpy as np
from scipy import sparse


class Corpus(object):

    def __init__(self, sentiment_corpus_file, max_features, maxDocs=None, n_training = 0):
        self.vocab = None
        self.sent_labels = []
        self.W_D_matrix = None
        i = 0
        all_txt = []
        with open(sentiment_corpus_file, encoding="utf8") as f:
            lines = f.readlines()[1:]
            if maxDocs != None:
                lines = lines[:maxDocs]
                 
            for lin in lines:
                label, txt = self.process_lin(lin)
                all_txt.append(txt)
                self.sent_labels.append(int(label))
                i += 1
        vectorizer = CountVectorizer(analyzer = "word", strip_accents = "unicode", stop_words = stopwords.words("english"), max_features = max_features)
        self.W_D_matrix = vectorizer.fit_transform(all_txt)
        self.N = self.W_D_matrix.shape[0]
        self.W_D_matrix_training = self.W_D_matrix[:n_training]
        self.W_D_matrix = self.W_D_matrix[n_training:]
        self.vocab = vectorizer.vocabulary_
        self.V = len(self.vocab)
        self.sent_labels = np.array(self.sent_labels)
        self.sent_labels_training = self.sent_labels[:n_training]
        self.sent_labels = self.sent_labels[n_training:]
        
    def process_lin(self, lin):
        split_lin = lin.split("\t")
        return split_lin[1], split_lin[2]
    
class Corpus_synthetic(object):
    def __init__(self, pi, theta0, theta1, nDocs, n_word_draws, n_training = 0):
        #fliping coins to determine the labels 
        self.sent_labels = sample_bernouli(pi, nDocs)
        self.V = len(theta0)
        docs = []
        for label in self.sent_labels:
            if label == 0:
                word_vec = sample_multinomial(n_word_draws, theta0)
            else:
                word_vec = sample_multinomial(n_word_draws, theta1)
            docs.append(word_vec)
            
        self.W_D_matrix = sparse.csr_matrix(docs)
        self.N = self.W_D_matrix.shape[0]
        self.W_D_matrix_training = self.W_D_matrix[:n_training]
        self.W_D_matrix = self.W_D_matrix[n_training:]
        self.sent_labels_training = self.sent_labels[:n_training]
        self.sent_labels = self.sent_labels[n_training:]
        