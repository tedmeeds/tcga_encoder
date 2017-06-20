import numpy as np
import scipy as sp
import pylab as pp
import pandas as pd

class GenerativeBinaryClassifier(object):
  def __init__(self):
    pass
    
  def fit( self, X, y ):
    self.y_true = y
    
    self.ids_1 = pp.find(y==1)
    self.ids_0 = pp.find(y==0)
    
    self.n = len(y)
    self.n_1 = y.sum()
    self.n_0 = self.n-self.n_1
    
    self.pi_1 = float(self.n_1) / float(self.n)
    self.pi_0 = float(self.n_0) / float(self.n)
    
    self.log_pi_1 = np.log(self.pi_1)
    self.log_pi_0 = np.log(self.pi_0)
    
    self.mu_1 = X[self.ids_1,:].mean(0)
    self.mu_0 = X[self.ids_0,:].mean(0)
    
    self.std_1 = X[self.ids_1,:].std(0) + 1e-12
    self.std_0 = X[self.ids_0,:].std(0) + 1e-12
    
    self.var_1 = np.square(self.std_1)
    self.var_0 = np.square(self.std_0)
    
    self.log_std_1 = np.log( self.std_1 )
    self.log_std_0 = np.log( self.std_0 )
    
    self.common_b = self.log_pi_1 - self.log_pi_0 \
                    + np.sum(self.log_std_0) - np.sum(self.log_std_1) \
                    + 0.5*np.sum( np.square(self.mu_0)/self.var_0) \
                    - 0.5*np.sum( np.square(self.mu_1)/self.var_1)
                    
    self.b_func_x_weights = 0.5*(1.0/self.var_0 - 1.0/self.var_1)
    
    self.w = self.mu_1 / self.var_1 - self.mu_0 / self.var_0
    
  def predict( self, X ):
    
    a = self.common_b
    
    a += np.dot( np.square(X), self.b_func_x_weights)
    
    a += np.dot( X, self.w )
    
    return 1.0 / (1.0 + np.exp( -a ))
                    