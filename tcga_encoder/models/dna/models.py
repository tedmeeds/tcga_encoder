from tcga_encoder.utils.helpers import *
from scipy import special
#import numpy as np
import pdb

def logistic_sigmoid( activations ):
  return 1.0 / (1.0 + np.exp(-activations) )
  
class NaiveBayesClassifier(object):
  def fit( self, X, y ):
    raise NotImplementedError
  def predict( self, X ):
    raise NotImplementedError
    
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
    
  def fit( self, X, y ):
    self.class_1 = pp.find( y==1 )
    self.class_0 = pp.find( y==0 )
    
    self.pi_1 = len( self.class_1 ) / float(len(y))
    self.pi_0 = 1.0 - self.pi_1
    
    self.mu_1 = X[self.class_1,:].mean(0)
    self.mu_0 = X[self.class_0,:].mean(0)
    self.var_1 = X[self.class_1,:].var(0)
    self.var_0 = X[self.class_0,:].var(0)
    
    self.w = self.mu_1/self.var_1 - self.mu_0/self.var_0
    self.b_vec = self.mu_0*self.mu_0/(2*self.var_0) - self.mu_1*self.mu_1/(2*self.var_1) + 0.5*np.log(self.var_0)- 0.5*np.log(self.var_1)          

    self.b_x_factor = 1.0/(2*self.var_0) - 1.0/(2*self.var_1) 
    
    self.common_b = np.log(self.pi_1) - np.log(self.pi_0)
    self.b = np.sum( self.b_vec )
    
  def predict( self, X, elementwise = False ):
    N,D = X.shape

    if elementwise is True:
      activations = X*self.w + np.square(X)*self.b_x_factor + self.b_vec + self.common_b
      
    else:
      #z = np.hstack( (Z_alpha,Z_beta) )
      activations = np.dot( X, self.w ) + np.dot(np.square(X),self.b_x_factor) + self.b + self.common_b
      
    predictions = logistic_sigmoid( activations )
    return predictions

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
        
    