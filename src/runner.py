'''
Created on Jun 17, 2016

@author: pedrom
'''
from corpus import Corpus
from model import ModelState
import numpy as np
import matplotlib.pyplot as plt

def plot_results(results_file):
    with open(results_file) as r_file:
        lins = r_file.readlines()[:-1]
        x = range(len(lins))
        y = []
        for lin in lins:
            y.append(float(lin))
            
        plt.plot(x, y)
        plt.ylabel('f1 score')
        plt.show()
        
if __name__ == '__main__':
    results_file = "results.txt"
    gamma_pi0 = 1
    gamma_pi1 = 1
    gamma_theta_val = 1
    corpus = Corpus("labeledTrainData.tsv")
    gamma_theta = np.full(corpus.V, gamma_theta_val)
    model_state = ModelState(gamma_pi0, gamma_pi1, gamma_theta, corpus.W_D_matrix.shape[0], corpus, results_file)
    
    n_iter = 50000
    burn_in = 5000
    lag = 25
    model_state.gibbs_sampler(n_iter, burn_in, lag)
    
    plot_results(results_file)
    print("finish running")
            