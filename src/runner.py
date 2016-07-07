'''
Created on Jun 17, 2016

@author: pedrom
'''
from corpus import Corpus
from model import ModelState
import numpy as np


if __name__ == '__main__':
    gamma_pi0 = 1
    gamma_pi1 = 1
    gamma_theta_val = 1
    
    corpus = Corpus("labeledTrainData.tsv")
    gamma_theta = np.full(corpus.V, gamma_theta_val)
    model_state = ModelState(gamma_pi0, gamma_pi1, gamma_theta, corpus.W_D_matrix.shape[0], corpus)
    model_state.sample_L()
    #model_state.sample_Theta0()
    print("finish running")