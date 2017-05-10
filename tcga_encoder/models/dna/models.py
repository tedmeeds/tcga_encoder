from tcga_encoder.utils.helpers import *
from scipy import special
#import numpy as np
import pdb
from sklearn.neighbors import KernelDensity
from sklearn import linear_model

def logistic_sigmoid( activations ):
  return 1.0 / (1.0 + np.exp(-activations) )
  
class NaiveBayesClassifier(object):
  def fit( self, X, y ):
    raise NotImplementedError
  def predict( self, X ):
    raise NotImplementedError

class LogisticRegressionModel( object ):
    
  def get_weights(self):
    return self.w
    
  def fit( self, X, y, one_hot_groups = None ):
    if one_hot_groups is None:
      return self.fit_normal( X, y )
    else:
      return self.fit_grouped( X, y, one_hot_groups )
  
  def fit_normal( self, X, y ):
    self.class_1 = pp.find( y==1 )
    self.class_0 = pp.find( y==0 )
    
    self.mu_1_0 = X[self.class_1,:].mean(0)
    self.mu_0_0 = X[self.class_0,:].mean(0)
    self.var_1_0 = X[self.class_0,:].var(0) 
    self.var_0_0 = X[self.class_0,:].var(0) 
    
    X_normed = X - self.mu_0_0
    X_normed /= self.var_0_0
    
    self.model = linear_model.LogisticRegression( penalty='l1', C=0.1 )
    self.model.fit( X_normed, y )
    
    self.pi_1 = len( self.class_1 ) / float(len(y))
    self.pi_0 = 1.0 - self.pi_1

    self.w = self.model.coef_
    self.common_b = self.model.intercept_
 
  def fit_grouped( self, X, y, groups ):
    assert False, "not implemented"
    self.class_1 = pp.find( y==1 )
    self.class_0 = pp.find( y==0 )
    
    self.pi_1 = len( self.class_1 ) / float(len(y))
    self.pi_0 = 1.0 - self.pi_1

    self.mu_1_0 = X[self.class_1,:].mean(0)
    self.mu_0_0 = X[self.class_0,:].mean(0)
    self.var_1_0 = X[self.class_0,:].var(0) 
    self.var_0_0 = X[self.class_0,:].var(0) 
    
    D = len(self.mu_1_0)
    N,K = groups.shape
      
    pi_1 = self.pi_1
    self.pi_1 = pi_1*np.ones(K)
    self.mu_1 = self.mu_1_0*np.ones((K,D))
    self.mu_0 = self.mu_0_0*np.ones((K,D))
    self.var_1 = self.var_1_0*np.ones((K,D))
    self.var_0 = self.var_0_0*np.ones((K,D))
    
    for k in range(K):
      ik = pp.find(groups[:,k]==1)
      class_1 = pp.find( y[ik]==1 )
      class_0 = pp.find( y[ik]==0 )
      if len(class_1) > 0:
        self.pi_1[ k ] = len( class_1 ) / float(len(ik))
        self.mu_1[ k,: ] = X[class_1,:].mean(0)
        self.mu_0[ k,: ] = X[class_0,:].mean(0)
        self.var_1[ k,: ] = 0.9*X[class_0,:].var(0) + 0.1*self.var_1_0.reshape((1,D))
        self.var_0[ k,: ] = 0.9*X[class_0,:].var(0)  + 0.1*self.var_0_0.reshape((1,D))
        
    
    
    self.w = self.mu_1/self.var_1 - self.mu_0/self.var_0
    self.b_vec = self.mu_0*self.mu_0/(2*self.var_0) - self.mu_1*self.mu_1/(2*self.var_1) + 0.5*np.log(self.var_0)- 0.5*np.log(self.var_1)          

    self.b_x_factor = 1.0/(2*self.var_0) - 1.0/(2*self.var_1) 
    
    self.common_b = np.log(self.pi_1) - np.log(self.pi_0)
    self.b = np.sum( self.b_vec )
        
  def predict( self, X, elementwise = False, one_hot_groups = None ):
    if one_hot_groups is None:
      return self.predict_normal( X, elementwise=elementwise)
    else:
      return self.predict_grouped( X, one_hot_groups, elementwise=elementwise )
  
  def predict_normal( self, X, elementwise = False ):
    N,D = X.shape

    common_b = self.common_b
    
    X_normed = X - self.mu_0_0
    X_normed /= self.var_0_0
    
    if elementwise is True:
  
      activations = X_normed*self.w + self.common_b
      predictions = logistic_sigmoid( activations )
    else:
      predictions = self.model.predict_proba( X_normed )[:,1]
    return predictions

  def predict_grouped( self, X, groups, elementwise = False ):
    N,D = X.shape
    
    common_b = np.dot( groups, self.common_b ) #.reshape( (N,1))
    w = np.dot( groups, self.w )
    b = np.dot( groups, self.b_vec )
    b_x_factor = np.dot( groups, self.b_x_factor )
        
    if elementwise is True:
      #pdb.set_trace()
      activations = X*w + np.square(X)*b_x_factor + b + common_b.reshape((N,1))

    else:
      #z = np.hstack( (Z_alpha,Z_beta) )
      #pdb.set_trace()
      activations = np.sum(X*w,1)  + np.sum(np.square(X)*b_x_factor,1) + b.sum(1) + common_b
      
    predictions = logistic_sigmoid( activations )
    
    if np.any( np.isnan( predictions) ) or np.any( np.isinf( predictions) ):
      pdb.set_trace()
    return predictions
        
class BetaNaiveBayesModel( object ):
    
  def get_weights(self):
    return np.vstack( (self.w_alpha,self.w_beta)).T
    
  def fit( self, X, y ):
    self.class_1 = pp.find( y==1 )
    self.class_0 = pp.find( y==0 )
    
    self.pi_1 = len( self.class_1 ) / float(len(y))
    self.pi_0 = 1.0 - self.pi_1
    
    self.mu_1 = X[self.class_1,:].mean(0)
    self.mu_0 = X[self.class_0,:].mean(0)
    self.var_1 = X[self.class_1,:].var(0)
    self.var_0 = X[self.class_0,:].var(0)
    
    self.alpha_1 = self.mu_1*( self.mu_1*(1-self.mu_1)/self.var_1 - 1.0 )
    self.alpha_0 = self.mu_0*( self.mu_0*(1-self.mu_0)/self.var_0 - 1.0 )
    
    self.beta_1 = (1.0-self.mu_1)*( self.mu_1*(1-self.mu_1)/self.var_1 - 1.0 )
    self.beta_0 = (1.0-self.mu_0)*( self.mu_0*(1-self.mu_0)/self.var_0 - 1.0 )
    
    # w is D by 2
    self.w_alpha = self.alpha_1 - self.alpha_0
    self.w_beta  = self.beta_1 - self.beta_0
    self.b_vec = - special.gammaln( self.alpha_0+self.beta_0 ) \
        + special.gammaln( self.alpha_1+self.beta_1 ) \
        + special.gammaln( self.alpha_0 ) + special.gammaln( self.beta_0 ) \
        - special.gammaln( self.alpha_1 ) - special.gammaln( self.beta_1 )
    
    
    self.common_b = np.log(self.pi_1) - np.log(self.pi_0)
    self.w = np.hstack( (self.w_alpha, self.w_beta )).T    
    self.b = np.sum( self.b_vec )
    
  def predict( self, X, elementwise = False ):
    logX     = np.log( np.maximum( X, 1e-12 ) )
    logXcomp = np.log( np.maximum( 1.0-X, 1e-12 ) )
    N,D = X.shape

    Z_alpha = logX
    Z_beta  = logXcomp
    
    if elementwise is True:
      activations_alpha = Z_alpha*self.w_alpha
      activations_beta = Z_beta*self.w_beta
      activations = activations_alpha+activations_beta + self.b_vec + self.common_b
      
    else:
      z = np.hstack( (Z_alpha,Z_beta) )
      activations = np.dot( z, self.w ) + self.b + self.common_b
      
    predictions = logistic_sigmoid( activations )
    return predictions

class PoissonNaiveBayesModel( object ):
    
  def get_weights(self):
    return self.w
    
  def fit( self, X, y ):
    self.class_1 = pp.find( y==1 )
    self.class_0 = pp.find( y==0 )
    
    self.pi_1 = len( self.class_1 ) / float(len(y))
    self.pi_0 = 1.0 - self.pi_1
    
    self.mu_1 = X[self.class_1,:].mean(0)
    self.mu_0 = X[self.class_0,:].mean(0)
    self.var_1 = X[self.class_1,:].var(0)
    self.var_0 = X[self.class_0,:].var(0)
    
    self.rate_1 = self.mu_1
    self.rate_0 = self.mu_0
    
    self.w = np.log(self.rate_1) - np.log(self.rate_0)
    
    self.b_vec = self.rate_0 - self.rate_1               

    self.common_b = np.log(self.pi_1) - np.log(self.pi_0)
    self.b = np.sum( self.b_vec )
    
  def predict( self, X, elementwise = False ):
    N,D = X.shape

    if elementwise is True:
      activations = X*self.w + self.b_vec + self.common_b
    else:
      activations = np.dot( X, self.w ) + self.b + self.common_b
      
    predictions = logistic_sigmoid( activations )
    return predictions
    
class GaussianNaiveBayesModel( object ):
    
  def get_weights(self):
    return self.w
    
  def fit( self, X, y, one_hot_groups = None ):
    if one_hot_groups is None:
      return self.fit_normal( X, y )
    else:
      return self.fit_grouped( X, y, one_hot_groups )
  
  def fit_normal( self, X, y ):
    self.class_1 = pp.find( y==1 )
    self.class_0 = pp.find( y==0 )
    
    self.pi_1 = len( self.class_1 ) / float(len(y))
    self.pi_0 = 1.0 - self.pi_1
 
    self.mu_1 = X[self.class_1,:].mean(0)
    self.mu_0 = X[self.class_0,:].mean(0)
    self.var_1 = X[self.class_1,:].var(0) 
    self.var_0 = X[self.class_0,:].var(0) 
    self.var_0 = 0.5*X.var(0) 
    
    self.var_1 = self.var_0
    self.w = self.mu_1/self.var_1 - self.mu_0/self.var_0
    self.b_vec = self.mu_0*self.mu_0/(2*self.var_0) - self.mu_1*self.mu_1/(2*self.var_1) + 0.5*np.log(self.var_0)- 0.5*np.log(self.var_1)          

    self.b_x_factor = 1.0/(2*self.var_0) - 1.0/(2*self.var_1) 
    
    self.common_b = np.log(self.pi_1) - np.log(self.pi_0)
    self.b = np.sum( self.b_vec )

  def fit_grouped( self, X, y, groups ):
    self.class_1 = pp.find( y==1 )
    self.class_0 = pp.find( y==0 )
    
    self.pi_1 = len( self.class_1 ) / float(len(y))
    self.pi_0 = 1.0 - self.pi_1

    self.mu_1_0 = X[self.class_1,:].mean(0)
    self.mu_0_0 = X[self.class_0,:].mean(0)
    self.var_1_0 = X[self.class_0,:].var(0) 
    self.var_0_0 = X[self.class_0,:].var(0) 
    
    D = len(self.mu_1_0)
    N,K = groups.shape
      
    pi_1 = self.pi_1
    self.pi_1 = pi_1*np.ones(K)
    self.mu_1 = self.mu_1_0*np.ones((K,D))
    self.mu_0 = self.mu_0_0*np.ones((K,D))
    self.var_1 = self.var_1_0*np.ones((K,D))
    self.var_0 = self.var_0_0*np.ones((K,D))
    
    for k in range(K):
      ik = pp.find(groups[:,k]==1)
      class_1 = pp.find( y[ik]==1 )
      class_0 = pp.find( y[ik]==0 )
      if len(class_1) > 0:
        self.pi_1[ k ] = len( class_1 ) / float(len(ik))
        self.mu_1[ k,: ] = X[class_1,:].mean(0)
        self.mu_0[ k,: ] = X[class_0,:].mean(0)
        self.var_1[ k,: ] = 0.9*X[class_0,:].var(0) + 0.1*self.var_1_0.reshape((1,D))
        self.var_0[ k,: ] = 0.9*X[class_0,:].var(0)  + 0.1*self.var_0_0.reshape((1,D))
        
    
    
    self.w = self.mu_1/self.var_1 - self.mu_0/self.var_0
    self.b_vec = self.mu_0*self.mu_0/(2*self.var_0) - self.mu_1*self.mu_1/(2*self.var_1) + 0.5*np.log(self.var_0)- 0.5*np.log(self.var_1)          

    self.b_x_factor = 1.0/(2*self.var_0) - 1.0/(2*self.var_1) 
    
    self.common_b = np.log(self.pi_1) - np.log(self.pi_0)
    self.b = np.sum( self.b_vec )
        
  def predict( self, X, elementwise = False, one_hot_groups = None ):
    if one_hot_groups is None:
      return self.predict_normal( X, elementwise=elementwise)
    else:
      return self.predict_grouped( X, one_hot_groups, elementwise=elementwise )
  
  def predict_normal( self, X, elementwise = False ):
    N,D = X.shape

    common_b = self.common_b
        
    if elementwise is True:
  
      activations = X*self.w + np.square(X)*self.b_x_factor + self.b_vec + common_b

    else:
      #z = np.hstack( (Z_alpha,Z_beta) )
      #pdb.set_trace()
      activations = np.dot( X, self.w.T ) + np.dot(np.square(X),self.b_x_factor.T) + self.b + common_b
      
    predictions = logistic_sigmoid( activations )
    return predictions

  def predict_grouped( self, X, groups, elementwise = False ):
    N,D = X.shape
    
    common_b = np.dot( groups, self.common_b ) #.reshape( (N,1))
    w = np.dot( groups, self.w )
    b = np.dot( groups, self.b_vec )
    b_x_factor = np.dot( groups, self.b_x_factor )
        
    if elementwise is True:
      #pdb.set_trace()
      activations = X*w + np.square(X)*b_x_factor + b + common_b.reshape((N,1))

    else:
      #z = np.hstack( (Z_alpha,Z_beta) )
      #pdb.set_trace()
      activations = np.sum(X*w,1)  + np.sum(np.square(X)*b_x_factor,1) + b.sum(1) + common_b
      
    predictions = logistic_sigmoid( activations )
    
    if np.any( np.isnan( predictions) ) or np.any( np.isinf( predictions) ):
      pdb.set_trace()
    return predictions

class KernelDensityNaiveBayesModel( object ):
  
  def fit( self, X, y, one_hot_groups = None ):
    if one_hot_groups is None:
      return self.fit_normal( X, y )
    else:
      return self.fit_grouped( X, y, one_hot_groups )
  
  def fit_normal( self, X, y ):
    self.class_1 = pp.find( y==1 )
    self.class_0 = pp.find( y==0 )
    
    self.pi_1 = len( self.class_1 ) / float(len(y))
    self.pi_0 = 1.0 - self.pi_1
 
    N,D = X.shape
    
    self.bandwidths = 5*np.std(X,0)*pow(4.0/3.0/N, 1.0/5.0)
    
    self.models0 = []
    self.models1 = []
    for d in xrange(D):
      self.models0.append( KernelDensity( kernel='gaussian', bandwidth = self.bandwidths[d] ) )
      self.models0[-1].fit( X[self.class_0,:][:,d][:, np.newaxis] )

      self.models1.append( KernelDensity( kernel='gaussian', bandwidth = self.bandwidths[d] ) )
      self.models1[-1].fit( X[self.class_1,:][:,d][:, np.newaxis] )
      
      #pdb.set_trace()
    
    # self.w = self.mu_1/self.var_1 - self.mu_0/self.var_0
    # self.b_vec = self.mu_0*self.mu_0/(2*self.var_0) - self.mu_1*self.mu_1/(2*self.var_1) + 0.5*np.log(self.var_0)- 0.5*np.log(self.var_1)
    #
    # self.b_x_factor = 1.0/(2*self.var_0) - 1.0/(2*self.var_1)
    #
    self.common_b = np.log(self.pi_1) - np.log(self.pi_0)
    # self.b = np.sum( self.b_vec )

  def predict( self, X, elementwise = False, one_hot_groups = None ):
    if one_hot_groups is None:
      return self.predict_normal( X, elementwise=elementwise)
    else:
      return self.predict_grouped( X, one_hot_groups, elementwise=elementwise )
  
  def predict_normal( self, X, elementwise = False ):
    N,D = X.shape

    common_b = self.common_b

    log_prob_0 = self.log_prob( self.models0, X )
    log_prob_1 = self.log_prob( self.models1, X )
    
    if elementwise is True:
      activations = log_prob_1 - log_prob_0 + common_b
    else:
      activations = np.sum(log_prob_1 - log_prob_0, 1 ) + common_b
      
    predictions = logistic_sigmoid( activations )
    return predictions

  def log_prob( self, models, X ):
    N,D = X.shape
    
    logprob = np.zeros( (N,D) )
    for model, d in zip( models, xrange(D) ):
      logprob[:,d] = model.score_samples( X[:,d][:, np.newaxis] )
      
    return logprob
      
    
class NegBinNaiveBayesModel( object ):
    
  def get_weights(self):
    return self.w
    
  def fit( self, X, y ):
    self.class_1 = pp.find( y==1 )
    self.class_0 = pp.find( y==0 )
    
    self.pi_1 = len( self.class_1 ) / float(len(y))
    self.pi_0 = 1.0 - self.pi_1
    
    self.mu_1 = X[self.class_1,:].mean(0)
    self.mu_0 = X[self.class_0,:].mean(0)
    
    self.var_1 = X[self.class_1,:].var(0)
    self.var_0 = X[self.class_0,:].var(0)
    
    self.alpha_1 = np.square(self.mu_1)/np.maximum(0.1,self.var_1 - self.mu_1)
    self.alpha_0 = np.square(self.mu_0)/np.maximum(0.1,self.var_0 - self.mu_0)
    self.beta_1 = self.mu_1/np.maximum(0.1,self.var_1 - self.mu_1)
    self.beta_0 = self.mu_0/np.maximum(0.1,self.var_0 - self.mu_0)
    
    self.w = np.log(self.beta_1) - np.log(self.beta_0)
                   
                 
    self.b_vec = - special.gammaln( self.alpha_1 ) \
                 + special.gammaln( self.alpha_0 ) \
                 - self.alpha_0*np.log(1.0+self.beta_0) + self.alpha_1*np.log(1.0+self.beta_1)
    
    self.common_b = np.log(self.pi_1) - np.log(self.pi_0)
    self.b = np.sum( self.b_vec )
    
  def predict( self, X, elementwise = False ):
    N,D = X.shape
    b_x = special.gammaln( self.alpha_0 + X ) + special.gammaln( self.alpha_1 + X )
    
    if elementwise is True:
      activations = X*self.w + b_x + self.b_vec + self.common_b
      
    else:
      activations = np.dot( X, self.w ) + b_x.sum() + self.b + self.common_b
      
    predictions = logistic_sigmoid( activations )
    return predictions
        
    