'''
Created on Jun 17, 2016

@author: pedrom
'''
import numpy as np

def sample_beta(pi0, pi1):
    return np.random.beta(pi0, pi1)

def sample_bernouli(pi, n_samples):
    return np.random.binomial(1, pi, size=n_samples)

def sample_dirichlet(alpha):
    return np.random.dirichlet(alpha)

def sample_multinomial(n, pvals):
    return np.random.multinomial(n, pvals)