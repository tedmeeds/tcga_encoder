import numpy as np
import pylab as pp
import scipy as sp

class BayesianLinearRegression( object ):
  def __init__(self, dim, mu_prior = 0.0, alpha_prior = 0.0, beta_prior = 1.0 ):
    self.dim = dim
    self.mu_prior = mu_prior
    self.alpha_prior = alpha_prior
    self.beta_prior  = beta_prior
    self.data_prior_inv_cov = beta_prior*np.eye( dim+1 )
    
    # remove penalty for bias
    self.data_prior_inv_cov[-1,:] *= 0
    
    self.lambda_prior = alpha_prior/beta_prior # equivalent ridge parameter 
    
    self.var_prior = 1.0 / self.alpha_prior 
    
    self.w_prior_mu = self.mu_prior*np.ones( dim + 1 )
    self.w_prior_cov= self.var_prior*np.eye( dim + 1 )
    self.w_prior_inv_cov = self.alpha_prior*np.eye( dim + 1 )
    
    # remove penlty for bias
    self.w_prior_inv_cov[-1,:] *= 0
    
    self.bias = 0.0
    
  def fit( self, X_orig, y ):
    X = np.hstack( (X_orig, np.ones((len(X_orig),1))))
    
    self.w_post_inv_cov = self.w_prior_inv_cov + np.dot( self.data_prior_inv_cov, np.dot( X.T, X ) )
    self.w_post_cov = np.linalg.pinv( self.w_post_inv_cov )
     
    self.w_post_mu = np.dot( self.w_post_cov, np.dot(self.w_prior_inv_cov,self.w_prior_mu) + np.dot( self.data_prior_inv_cov, np.dot( X.T, y ) ))
    
    self.bias      = self.w_post_mu[-1]
    self.w_post_mu = self.w_post_mu[:self.dim]
    self.w_post_cov = self.w_post_cov[:self.dim,:][:,self.dim]
    self.w_post_inv_cov = self.w_post_inv_cov[:self.dim,:][:,self.dim]
    
  def predict( self, X ):
    return np.dot( X, self.w_post_mu ) + self.bias
    