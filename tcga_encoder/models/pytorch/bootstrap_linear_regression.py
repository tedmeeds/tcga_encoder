import numpy as np
import pylab as pp
import scipy as sp
import sklearn
def bootstraps( x, m ):
  # samples from arange(n) with replacement, m times.
  #x = np.arange(n, dtype=int)
  n = len(x)
  N = np.zeros( (m,n), dtype=int)
  for i in range(m):
    N[i,:] = sklearn.utils.resample( x, replace = True )
    
  return N
  
class BootstrapLinearRegression( object ):
  def __init__(self, dim, ridge = 0.0 ):
    self.dim = dim
    self.ridge = ridge
    
  def fit( self, X_orig, y, n_bootstraps = 1 ):
    n = len(X_orig)
    X = X_orig #np.hstack( (X_orig, np.ones((len(X_orig),1))))
    indices = np.arange(n, dtype=int)
    
    self.W = np.zeros( (self.dim, n_bootstraps ), dtype=float )
    
    I = bootstraps( indices, n_bootstraps )
    for i in range( n_bootstraps):
      
      
      X_boot = X[I[i],:]
      y_boot = y[I[i]]
      
      w = np.dot( np.linalg.pinv( np.dot( X_boot.T, X_boot ) + self.ridge*np.eye(self.dim) ), np.dot( X_boot.T, y_boot ) )
      
      self.W[:,i] = w
    self.w = self.W.mean(1)
      
  def predict( self, X ):
    #predictions = np.dot( X, self.W )
    return np.dot( X, self.w )

class BootstrapLassoRegression( object ):
  def __init__(self, dim, ridge = 0.0 ):
    self.dim = dim
    self.ridge = ridge
    
  def fit( self, X_orig, y, n_bootstraps = 1 ):
    n = len(X_orig)
    X = X_orig #np.hstack( (X_orig, np.ones((len(X_orig),1))))
    indices = np.arange(n, dtype=int)
    
    self.W = np.zeros( (self.dim, n_bootstraps ), dtype=float )
    self.B = np.zeros( n_bootstraps, dtype=float )
    
    I = bootstraps( indices, n_bootstraps )
    for i in range( n_bootstraps):
      
      
      X_boot = X[I[i],:]
      y_boot = y[I[i]]
      
      model = sklearn.linear_model.Lasso(alpha=self.ridge, fit_intercept=True)
      model.fit(X_boot, y_boot)
      
      #w = np.dot( np.linalg.pinv( np.dot( X_boot.T, X_boot ) + self.ridge*np.eye(self.dim) ), np.dot( X_boot.T, y_boot ) )
      w = np.squeeze( model.coef_ )
      b = np.squeeze( model.intercept_)
      self.W[:,i] = w
      self.B[i] = b
    self.w = self.W.mean(1)
    self.b = self.B.mean()
      
  def predict( self, X ):
    #predictions = np.dot( X, self.W )
    return np.dot( X, self.w ) + self.b
        