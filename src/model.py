'''
Created on Jun 17, 2016

@author: pedrom
'''
from dist_sampler import sample_bernouli, sample_beta, sample_dirichlet
import numpy as np
import math

class ModelState(object):
    def __init__(self, gamma_pi0, gamma_pi1, gamma_theta, N, corpus):
        self.corpus = corpus
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
        self.theta1 = sample_dirichlet(self.gamma_theta)
        self.C1 = np.count_nonzero(self.L)
        self.C0 = self.N - self.C1
        
    def sample_L(self, j):
        factor2_log_L0 = 1.0
        factor2_log_L1 = 1.0
        for i in range(self.corpus.V):  
            #print(factor2_log_L0)
            factor2_log_L0 += np.log(self.theta0[i]**self.corpus.W_D_matrix[j, i])
            factor2_log_L1 += np.log(self.theta1[i]**self.corpus.W_D_matrix[j, i])
        #Computing for x = 0 (Lj = 0)
            
        C0 = self.C0 - 1
        if C0 == -1:
            C0 = 1
        factor1_log_L0 = np.log((C0 + self.gamma_pi0 - 1.0) / (C0 + self.C1 + self.gamma_pi0 + self.gamma_pi1 - 1.0))
        log_val0 = factor1_log_L0 + factor2_log_L0
        
        #Computing for x = 1 (Lj = 1)
        C1 = self.C1 - 1
        if C1 == -1:
            C1 = 1
        factor1_log_L1 = np.log((C1 + self.gamma_pi1 - 1.0) / (C1 + self.C0 + self.gamma_pi0 + self.gamma_pi1 - 1.0))
        log_val1 = factor1_log_L1 + factor2_log_L1
        
        post_L0 = np.exp(log_val0 - np.logaddexp(log_val0, log_val1))
        post_L1 = np.exp(log_val1 - np.logaddexp(log_val0, log_val1))
        coin_weight = sample_beta(post_L0, post_L1)
        print("post_L0: %f post_L1: %f post_pi: %f" % (post_L0, post_L1, coin_weight))
        L_new = sample_bernouli(coin_weight, 1)
        return L_new
        