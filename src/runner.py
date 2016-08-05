'''
Created on Jun 17, 2016

@author: pedrom
'''
import json

from corpus import Corpus, Corpus_synthetic
from dist_sampler import sample_dirichlet
import matplotlib.pyplot as plt
from model import ModelState
import numpy as np


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
    with open('config.txt') as data_file:    
        config = json.load(data_file)
    
    results_file = config["results_file"]
    gamma_pi0 = config["gamma_pi0"]
    gamma_pi1 = config["gamma_pi1"]
    gamma_theta_val = config["gamma_theta_val"]
    maxDocs = config["maxDocs"]
    run_corpus_synthetic = config["run_corpus_synthetic"]
    n_training = config["n_training"]
    
    if maxDocs == "None":
        maxDocs = None
        
    if run_corpus_synthetic == "True":
        pi = 0.6
        theta0 =  sample_dirichlet([0.5]*100)
        theta1 = sample_dirichlet([0.5]*100)
        nDocs = 1000
        n_word_draws = 10
        n_training = 100
        corpus = Corpus_synthetic(pi, theta0, theta1, nDocs, n_word_draws, n_training)
    else:
        corpus = Corpus(config["corpus"], config["max_features"], maxDocs, n_training)
        
    gamma_theta = np.full(corpus.V, gamma_theta_val)
    model_state = ModelState(gamma_pi0, gamma_pi1, gamma_theta, corpus, results_file)
    
    n_iter = config["n_iter"]
    burn_in = config["burn_in"]
    lag = config["lag"]
    model_state.gibbs_sampler(n_iter, burn_in, lag)
    plot_results(results_file)
    print("finish running")
            