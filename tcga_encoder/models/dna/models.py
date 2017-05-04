from tcga_encoder.utils.helpers import *
from scipy import special
#import numpy as np
import pdb

def logistic_sigmoid( activations ):
  return 1.0 / (1.0 + np.exp(-activations) )
  
class NaiveBayesClassifier(object):
  # def z_from_x( self, X ):
  #   raise NotImplementedError
  def fit( self, X, y ):
    raise NotImplementedError
  def predict( self, X ):
    raise NotImplementedError
    
class BetaNaiveBayesModel( object ):
  def __init__(self):
    pass
    
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
     
       
    #pdb.set_trace()
    
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
    
    #
    # pass
    # for rna_gene in rna2use:
    #   a_0 = alpha_0.loc[rna_gene]; a_1 = alpha_1.loc[rna_gene]
    #   b_0 = beta_0.loc[rna_gene];  b_1 = beta_1.loc[rna_gene]
    #
    #   w = np.array([ a_1 - a_0,
    #                  b_1 - b_0 ])
    #
    #   b = - special.gammaln( a_0+b_0 ) \
    #       + special.gammaln( a_1+b_1 ) \
    #       + special.gammaln( a_0 ) + special.gammaln( b_0 ) \
    #       - special.gammaln( a_1 ) - special.gammaln( b_1 )
    #
    #   z_train = np.vstack( ( np.log( R_train[rna_gene].values + 1e-12 ), np.log( 1.0-R_train[rna_gene].values + 1e-12 ) ) ).T
    #   z_val   = np.vstack( ( np.log( R_val[rna_gene].values + 1e-12 ), np.log( 1.0-R_val[rna_gene].values + 1e-12 ) ) ).T
    #
    #   activations_train = np.dot( z_train, w ) + b
    #   activations_val   = np.dot( z_val, w ) + b
    #
    #
    #   predictions_train = 1.0 / (1.0 + np.exp(-activations_train-common_b) )
    #   predictions_val = 1.0 / (1.0 + np.exp(-activations_val-common_b) )
    
    
  # def z_from_x( self, X ):
  #   logX = np.log( X + 1e-12 )
  #   logXcomp = np.log( 1.0-X + 1e-12 )
  #   z = np.vstack( (logX, logXcomp ))
  #   return z
     
  # def predict( self, X ):
  #   #z_train = np.vstack( ( , ) ).T
  #   pdb.set_trace()
  #   return np.random.rand( len(X) )


# class PoissonRsem(object):
#   def __init__( self, arch_dict, data_dict ):
#     self.arch_dict = arch_dict
#     self.data_dict = data_dict
#
#   def train(self, algo_dict, logging_dict ):
#     self.algo_dict = algo_dict
#     self.logging_dict = logging_dict
    