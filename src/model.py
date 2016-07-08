'''
Created on Jun 17, 2016

@author: pedrom
'''
from dist_sampler import sample_bernouli, sample_beta, sample_dirichlet
import numpy as np
from sklearn.metrics import classification_report, f1_score
import os.path

class ModelState(object):
    def __init__(self, gamma_pi0, gamma_pi1, gamma_theta, N, corpus, results_file):
        self.results_file = results_file
        if os.path.isfile(results_file):
            os.remove(results_file)   
            
        self.corpus = corpus
        self.N = N
        self.L = np.zeros(N)
        self.estimated_L = np.zeros(N)
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
        
        docs_L0 = []
        docs_L1 = []
        for doc_id, Lj in enumerate(self.L):
            if Lj == 0:
                docs_L0.append(doc_id)
            else:
                docs_L1.append(doc_id)
        self.word_counts_L0 = corpus.W_D_matrix[docs_L0].sum(0)
        self.word_counts_L1 = corpus.W_D_matrix[docs_L1].sum(0)
            
    def sample_L(self):
        #prevL = np.copy(self.L)
        for j in range(len(self.L)):
            #print("Sampling L%i" % (j))
            self.sample_Lj(j)
        #print("L diff %f" % ((prevL - self.L).sum(1)))
        
    def sample_Lj(self, j):
        if self.L[j] == 0:
            self.C0 = self.C0 - 1.0
            self.word_counts_L0 -= self.corpus.W_D_matrix[j]
        else:
            self.C1 = self.C1 - 1.0
            self.word_counts_L1 -= self.corpus.W_D_matrix[j]
            
        #print("C0 %f C1 %f"  % (self.C0, self.C1))
        #Computing for x = 0 (Lj = 0)
        factor1_log_L0 = np.log((self.C0 + self.gamma_pi0 - 1.0) / (self.C0 + self.C1 + self.gamma_pi0 + self.gamma_pi1 - 1.0))
        
        #Computing for x = 1 (Lj = 1)
        factor1_log_L1 = np.log((self.C1 + self.gamma_pi1 - 1.0) / (self.C1 + self.C0 + self.gamma_pi0 + self.gamma_pi1 - 1.0))
        
        '''
        factor2_log_L0 = 0.0
        factor2_log_L1 = 0.0
        for i in range(self.corpus.V):  
            #print(factor2_log_L0)
            factor2_log_L0 += np.log(self.theta0[i])*self.corpus.W_D_matrix[j, i]
            factor2_log_L1 += np.log(self.theta1[i])*self.corpus.W_D_matrix[j, i]
        '''
        
        factor2_log_L0 = (np.log(self.theta0)*self.corpus.W_D_matrix[j].toarray()).sum(1)[0]
        factor2_log_L1 = (np.log(self.theta1)*self.corpus.W_D_matrix[j].toarray()).sum(1)[0]
        
        log_val0 = factor1_log_L0 + factor2_log_L0
        log_val1 = factor1_log_L1 + factor2_log_L1
        #print("log_val0 %s logaddexp(log_val0, log_val1) %s" % (str(log_val0), str(np.logaddexp(log_val0, log_val1))))
        #print("log_val1 %s logaddexp(log_val0, log_val1) %s" % (str(log_val1), str(np.logaddexp(log_val0, log_val1))))
        
        post_L0 = np.exp(log_val0 - np.logaddexp(log_val0, log_val1))
        post_L1 = np.exp(log_val1 - np.logaddexp(log_val0, log_val1))
        #print("factor1_log_L0: %s factor2_log_L0 %s post_L0: %s" % (str(factor1_log_L0), str(factor2_log_L0), str(post_L0)))
        #print("factor1_log_L1: %s factor2_log_L1 %s post_L1: %s" % (str(factor1_log_L1), str(factor2_log_L1), str(post_L1)))
        if post_L0 == 0.0 or post_L1 == 0.0:
            print("doc %i" % (j))
        coin_weight = sample_beta(post_L1, post_L0)
        #print("post_pi: %s" % (str(coin_weight)))
        Lj_new = sample_bernouli(coin_weight, 1)
        self.L[j] = Lj_new
        if Lj_new == 0:
            self.C0 += 1
            self.word_counts_L0 += self.corpus.W_D_matrix[j]
        else:
            self.C1 += 1
            self.word_counts_L1 += self.corpus.W_D_matrix[j]
            
    def sample_Theta0(self):
        post_theta0 = self.sample_Theta(self.gamma_theta, 0)
        self.theta0 = post_theta0
        #print("theta0 %s" % (str(self.theta0)))
        
    def sample_Theta1(self):
        post_theta1 = self.sample_Theta(self.gamma_theta, 1)
        self.theta1 = post_theta1
        #print("theta1 %s" % (str(self.theta1)))
        
    def sample_Theta(self, gamma_theta_prior, Cx):
        if Cx == 0:
            word_counts = self.word_counts_L0
        else:
            word_counts = self.word_counts_L1
        
        #NOTE: on numpy sum operations we get a matrix. With .A1 we get back an array
        t = word_counts.A1 + gamma_theta_prior
        return sample_dirichlet(t)
    
    def gibbs_sampler(self, n_iter, burn_in, lag):
        lag_counter = lag
        iteration = 1
        while iteration <= n_iter:
            print("Gibbs sampler iter %d" % (iteration))
            self.sample_L()
            self.sample_Theta0()
            self.sample_Theta1()
            print("C0 %f C1 %f"  % (self.C0, self.C1))
            
            if burn_in > 0:
                print("Burn in iteration")
                burn_in -= 1
            else:
                if lag_counter > 0:
                    print("Lag iteration")
                    lag_counter -= 1
                else:
                    print("Estimate iteration")
                    lag_counter = lag
                    self.estimated_L += self.L
                    
                    with open(self.results_file, "a") as f:
                        current_estimated_L = np.rint(self.estimated_L / iteration)
                        f.write(str(f1_score(self.corpus.sent_labels, current_estimated_L))+"\n")
                        
                    iteration += 1
                
        self.estimated_L /= iteration
        self.estimated_L = np.rint(self.estimated_L)
        target_names = ['L0', 'L1']
        print(classification_report(self.corpus.sent_labels, self.estimated_L, target_names=target_names))
        '''
        self.sample_L()
        self.sample_Theta0()
        self.sample_Theta1()
        '''
        
            
    def count_wi(self, docs, wi):
        count = 0
        for doc_i in docs:
            count += self.corpus.W_D_matrix[doc_i, wi]
        return count
    
    def debug_array(self, arr):
        str0 = ""
        for i in range(arr.shape[0]):
            str0 += str(arr[i]) + "\n"
        with open("debug.txt", "w+") as f:
            f.write(str0)