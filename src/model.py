'''
Created on Jun 17, 2016

@author: pedrom
'''
from dist_sampler import sample_bernouli, sample_beta, sample_dirichlet
import numpy as np
from sklearn.metrics import classification_report, f1_score
import os.path
from tqdm import trange

class ModelState(object):
    def __init__(self, gamma_pi0, gamma_pi1, gamma_theta, corpus, results_file):
    
        self.results_file = results_file
        if os.path.isfile(results_file):
            os.remove(results_file)   
        
        n_training = corpus.sent_labels_training.shape[0]
        n_to_estimate =  corpus.W_D_matrix.shape[0]
        
        self.corpus = corpus
        self.N = corpus.N
        self.estimated_L = np.zeros(n_to_estimate)
        #adding label counts of training instances
        self.C1 = np.count_nonzero(corpus.sent_labels_training)
        self.C0 =  n_training - self.C1
        self.theta0 = None
        self.theta1 = None
        self.gamma_pi0 = gamma_pi0
        self.gamma_pi1 = gamma_pi1
        self.gamma_theta = gamma_theta
        
        pi = sample_beta(self.gamma_pi0, self.gamma_pi1)
        self.L = sample_bernouli(pi, n_to_estimate)
        self.theta0 = sample_dirichlet(self.gamma_theta)
        self.theta1 = sample_dirichlet(self.gamma_theta)
        self.estimated_theta0 = np.zeros(self.theta0.shape[0])
        self.estimated_theta1 = np.zeros(self.theta1.shape[0])
        zero_count_L = np.count_nonzero(self.L)
        self.C1 += zero_count_L
        self.C0 += n_to_estimate - zero_count_L
        
        docs_L0, docs_L1 = self.group_by_L(self.L)
        self.word_counts_L0 = corpus.W_D_matrix[docs_L0].sum(0)
        self.word_counts_L1 = corpus.W_D_matrix[docs_L1].sum(0)
        
        #adding word counts of training instances
        docs_L0, docs_L1 = self.group_by_L(corpus.sent_labels_training)
        self.word_counts_L0 += corpus.W_D_matrix_training[docs_L0].sum(0)
        self.word_counts_L1 += corpus.W_D_matrix_training[docs_L1].sum(0)
            
    def group_by_L(self, labels):
        docs_L0 = []
        docs_L1 = []
        for doc_id, Lj in enumerate(labels):
            if Lj == 0:
                docs_L0.append(doc_id)
            else:
                docs_L1.append(doc_id)
        return docs_L0, docs_L1
        
    def sample_L(self):
        for j in range(len(self.L)):
            self.sample_Lj(j)
        
    def sample_Lj(self, j):
        if self.L[j] == 0:
            self.C0 = self.C0 - 1.0
            self.word_counts_L0 -= self.corpus.W_D_matrix[j]
        else:
            self.C1 = self.C1 - 1.0
            self.word_counts_L1 -= self.corpus.W_D_matrix[j]
            
        denom = self.C0 + self.C1 + self.gamma_pi0 + self.gamma_pi1 - 1.0
        #Computing for x = 0 (Lj = 0)
        factor1_log_L0 = np.log((self.C0 + self.gamma_pi0 - 1.0) / denom)
        
        #Computing for x = 1 (Lj = 1)
        factor1_log_L1 = np.log((self.C1 + self.gamma_pi1 - 1.0) / denom)
        
        factor2_log_L0 = self.corpus.W_D_matrix[j].dot(np.log(self.theta0))
        factor2_log_L1 = self.corpus.W_D_matrix[j].dot(np.log(self.theta1))
        
        log_val0 = factor1_log_L0 + factor2_log_L0
        log_val1 = factor1_log_L1 + factor2_log_L1
        
        post_L1 = np.exp(log_val1 - np.logaddexp(log_val0, log_val1))
        
        coin_weight = post_L1
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
        
    def sample_Theta1(self):
        post_theta1 = self.sample_Theta(self.gamma_theta, 1)
        self.theta1 = post_theta1
        
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
        iteration = 1.0
        total_iterations = burn_in + n_iter*lag + n_iter
        t = trange(total_iterations, desc='', leave=True)
        for i in t:
            self.sample_L()
            self.sample_Theta0()
            self.sample_Theta1()
            
            if burn_in > 0:
                t.set_description("Burn-in iter %i C0 %d C1 %d" % (burn_in, self.C0, self.C1))
                burn_in -= 1
            else:
                if lag_counter > 0:
                    t.set_description("Lag iter %i\tC0 %d C1 %d" % (iteration, self.C0, self.C1))
                    lag_counter -= 1
                else:
                    t.set_description("Estimate iter %i\tC0 %d C1 %d" % (iteration, self.C0, self.C1))
                    lag_counter = lag
                    self.estimated_L += self.L
                    self.estimated_theta0 += self.theta0
                    self.estimated_theta1 += self.theta1
                    
                    with open(self.results_file, "a") as f:
                        current_estimated_L = np.rint(self.estimated_L / iteration)
                        f.write(str(f1_score(self.corpus.sent_labels, current_estimated_L))+"\n")
                        
                    iteration += 1.0
                
        self.estimated_L /= iteration
        self.estimated_L = np.rint(self.estimated_L)
        self.estimated_theta0 /= iteration
        self.estimated_theta1 /= iteration
        target_names = ['L0', 'L1']
        print(classification_report(self.corpus.sent_labels, self.estimated_L, target_names=target_names))