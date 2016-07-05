'''
Created on Jun 17, 2016

@author: pedrom
'''
from corpus import Corpus
from model import ModelState
import numpy as np


if __name__ == '__main__':
    gamma_pi0 = 0.1
    gamma_pi1 = 0.1
    gamma_theta_val = 0.2
    
    corpus = Corpus("labeledTrainData.tsv")
    gamma_theta = np.full(corpus.V, gamma_theta_val)
    model_state = ModelState(gamma_pi0, gamma_pi1, gamma_theta, corpus.W_D_matrix.shape[0], corpus)
    print(model_state.sample_L(44))
    print("finish running")