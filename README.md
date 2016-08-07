# BayesianSentimentAnalysis
This project is a python implementation of the paper "Gibbs Sampling for the Uninitiated" by Philip Resnik and Eric Hardisty.
The main motivation for the project is to bridge the gap between understanding the paper and implementing it. In this context, a ipython notebook was writen to describe the code and the overall development process.

#Requirements
- matplotlib==1.3.1
- nltk==3.2.1
- numpy==1.11.1
- tqdm==4.8.3
- scipy==0.18.0
- scikit_learn==0.17.1

#Running
To run the code use the following command from the BayesianSentimentAnalysis directory:

  python src/runner.py $CONFIG_PATH

The $CONFIG_PATH argument corresponds to the path for a configuration file. The file "config.txt" is an example of a configuration file.

#Configuration
Configuration file parameters:

  - run_corpus_synthetic: flag that indicates if a synthetic corpus should be generated for the model to learn.
  - results_file: path to a file where the F1 scores in each iteration of the Gibbs Sampler are written. 
  - n_training: number of training instances
  - gamma_pi0: prior of the pi variable
  - gamma_pi1: prior of the pi variable
  - gamma_theta_val: value of a symmetric prior used so sample theta
  - n_iter: numer of iterations of the Gibbs Sampler to be performed
  - burn_in: number of burn-in iterations of the Gibbs Sampler
  - lag:  number of lag iterations of the Gibbs Sampler
  - maxDocs: maximum number of documents from which a model is learned
	
  - sentiment_corpus
      - max_features: maximum number of features in the multinomial language model (top most frequent words)
      - corpus: path to the corpus file
		
  - synthetic_corpus
  	- pi: probability of generating a document with L = 1
  	- gamma_theta: value of a symmetric prior used so sample theta0 and theta1 from a Direchelet
  	- n_features: number of features in the multinomial language model
  	- nDocs: number of documents to be generated
  	- n_word_draws: number of words in each document
