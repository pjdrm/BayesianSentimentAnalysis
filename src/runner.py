'''
Copyright Pedro Mota 2016

@author: pedrom
'''
import json
from corpus import Corpus, Corpus_synthetic
from dist_sampler import sample_dirichlet
import matplotlib.pyplot as plt
plt.rcdefaults()
import operator
from model import ModelState
import numpy as np
import sys

def plot_results(results_file):
    with open(results_file) as r_file:
        lins = r_file.readlines()[:-1]
        x = range(len(lins))
        y = []
        for lin in lins:
            y.append(float(lin))
            
        plt.plot(x, y)
        plt.ylabel('F1 score')
        plt.xlabel('Iteration')
        plt.show()
        
def plotPosterior(posteriors, labelDic, n, colors):
    f, axarr = plt.subplots(len(posteriors), squeeze=False)
    xAxis = [x/10. for x in range(0, 11)]
    for posterior, ax_i, color in zip(posteriors, range(len(posteriors)), colors):
        labelsVals = {}
        for i, val in enumerate(posterior):
            labelsVals[labelDic[i]] = val
        sorted_dic = sorted(labelsVals.items(), key=operator.itemgetter(1))
        yLabels = [label for label, val in sorted_dic[:n]]
        y_pos = np.arange(len(yLabels))/5.
        axarr[ax_i, 0].barh(y_pos, [labelsVals[y] for y in yLabels], align='center', color=color, height=0.2)
        axarr[ax_i, 0].set_yticks(y_pos)
        axarr[ax_i, 0].set_yticklabels(yLabels)

    plt.xlabel('Probability')
    plt.show()
        
if __name__ == '__main__':
    if len(sys.argv) == 1:
        config_file = '../config.txt'
    else:
        config_file = sys.argv[1]
    with open(config_file) as data_file:    
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
        pi = config["synthetic_corpus"]["pi"]
        
        n_features = config["synthetic_corpus"]["n_features"]
        gamma_theta = config["synthetic_corpus"]["gamma_theta"]
        theta0 =  sample_dirichlet([gamma_theta]*n_features)
        theta1 =  sample_dirichlet([gamma_theta]*n_features)
        
        nDocs = config["synthetic_corpus"]["nDocs"]
        n_word_draws = config["synthetic_corpus"]["n_word_draws"]
        corpus = Corpus_synthetic(pi, theta0, theta1, nDocs, n_word_draws, n_training)
    else:
        corpusPath = config["sentiment_corpus"]["corpus"]
        maxFeatures = config["sentiment_corpus"]["max_features"]
        corpus = Corpus(corpusPath, maxFeatures, maxDocs, n_training)
        
    gamma_theta = np.full(corpus.V, gamma_theta_val)
    model_state = ModelState(gamma_pi0, gamma_pi1, gamma_theta, corpus, results_file)
    
    n_iter = config["n_iter"]
    burn_in = config["burn_in"]
    lag = config["lag"]
    model_state.gibbs_sampler(n_iter, burn_in, lag)
    plot_results(results_file)
    
    if run_corpus_synthetic == "False":
        inv_vocab =  {v: k for k, v in corpus.vocab.items()}
        plotPosterior([model_state.estimated_theta0, model_state.estimated_theta1], inv_vocab, 10, ['g', 'r'])
