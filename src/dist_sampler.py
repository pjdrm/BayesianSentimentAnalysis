'''
Created on Jun 17, 2016

@author: pedrom
'''
import numpy

def sample_beta(pi0, pi1):
    return numpy.random.beta(pi0, pi1)

def sample_bernouli(pi, n_samples):
    return numpy.random.binomial(1, pi, size=n_samples)

def sample_dirichlet(alpha):
    return numpy.random.dirichlet(alpha)