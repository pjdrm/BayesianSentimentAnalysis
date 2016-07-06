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
        
    def sample_L(self):
        print("current L\n %s" % (str(self.L)))
        post_L = [self.sample_Lj(j) for j in range(len(self.L))]
        self.L = post_L
        print("postL\n%s" % (str(self.L)))
        
    def sample_Lj(self, j):
        if self.L[j] == 0:
            self.C0 = self.C0 - 1.0
        else:
            self.C1 = self.C1 - 1.0
        print("C0 %f C1 %f"  % (self.C0, self.C1))
        #Computing for x = 0 (Lj = 0)
        factor1_log_L0 = np.log((self.C0 + self.gamma_pi0 - 1.0) / (self.C0 + self.C1 + self.gamma_pi0 + self.gamma_pi1 - 1.0))
        
        #Computing for x = 1 (Lj = 1)
        factor1_log_L1 = np.log((self.C1 + self.gamma_pi1 - 1.0) / (self.C1 + self.C0 + self.gamma_pi0 + self.gamma_pi1 - 1.0))
        
        factor2_log_L0 = 0.0
        factor2_log_L1 = 0.0
        for i in range(self.corpus.V):  
            #print(factor2_log_L0)
            factor2_log_L0 += np.log(self.theta0[i]**self.corpus.W_D_matrix[j, i])
            factor2_log_L1 += np.log(self.theta1[i]**self.corpus.W_D_matrix[j, i])
            
        log_val0 = factor1_log_L0 + factor2_log_L0
        log_val1 = factor1_log_L1 + factor2_log_L1
        
        post_L0 = np.exp(log_val0 - np.logaddexp(log_val0, log_val1))
        post_L1 = np.exp(log_val1 - np.logaddexp(log_val0, log_val1))
        coin_weight = sample_beta(post_L0, post_L1)
        print("post_L0: %f post_L1: %f post_pi: %f" % (post_L0, post_L1, coin_weight))
        Lj_new = sample_bernouli(coin_weight, 1)
        self.L[j] = Lj_new
        if Lj_new == 0:
            self.C0 += 1
        else:
            self.C1 += 1
            
    def sample_Theta0(self):
        post_theta0 = self.sample_Theta(self.gamma_theta, 0)
        self.theta0 = post_theta0
        print("theta0 %s" % (str(self.theta0)))
        
    def sample_Theta1(self):
        post_theta1 = self.sample_Theta(self.gamma_theta, 1)
        self.theta1 = post_theta1
        print("theta1 %s" % (str(self.theta1)))
        
    def sample_Theta(self, gamma_theta_prior, Cx):
        if Cx == 0:
            docs = self.corpus.docs_L0
        else:
            docs = self.corpus.docs_L1
        
        t = np.zeros(self.corpus.V)
        for i in range(len(t)):
            NCxi = self.count_wi(docs, i)
            gamma_theta_i = gamma_theta_prior[i]
            t[i] = NCxi + gamma_theta_i
        return sample_dirichlet(t)
            
    def count_wi(self, docs, wi):
        count = 0
        for doc_i in docs:
            count += self.corpus.W_D_matrix[doc_i, wi]
        return count