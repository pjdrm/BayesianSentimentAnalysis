'''
Created on Jun 17, 2016

@author: pedrom
'''
from dist_sampler import sample_bernouli, sample_beta, sample_dirichlet
import numpy as np

class ModelState(object):
    '''
    classdocs
    '''

    def __init__(self, gamma_pi0, gamma_pi1, gamma_theta, N):
        self.N = N
        self.L = np.zeros(N)
        self.C0 = 0
        self.C1 = 0 
        self.theta0 = None
        self.theta1 = None
        self.gamma_pi0 = gamma_pi0
        self.gamma_pi1 = gamma_pi1
        self.gamma_theta = gamma_theta
        
        pi = sample_beta(self.gamma_pi0, self.gamma_pi1)
        self.L = sample_bernouli(pi, self.N)
        self.theta0 = sample_dirichlet(self.gamma_theta)
        self.C1 = np.count_nonzero(self.L)
        self.C0 = self.N - self.C1