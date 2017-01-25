import numpy as np
import scipy as sp
import pylab as pp
from collections import OrderedDict

class LinearDiscriminantAnalysis( object ):
  def __init__(self, epsilon = 1e-12 ):
    self.epsilon = epsilon # used for computing sw
    self.fitted = False
    
  def fit( self, X, y ):
    self.classes = np.unique( y )
    assert len(self.classes) == 2, "only works for 2 classes"
    
    self.n, self.d = X.shape
    assert len(y) == self.n, "X and y should have same length"
    
    self.Sw = np.zeros( (self.d,self.d), dtype=float )
    self.class_ids = OrderedDict()
    self.class_X   = OrderedDict()
    self.class_mean = OrderedDict()
    self.class_X_normed = OrderedDict()
    self.class_S = OrderedDict()
    for k in self.classes:
      self.class_ids[k]  = pp.find( y == k) 
      self.class_X[k]    = X[self.class_ids[k],:].astype(float)
      self.class_mean[k] = self.class_X[k].mean(0).astype(float)
      self.class_X_normed[k]    = X[self.class_ids[k],:].astype(float) - self.class_mean[k].astype(float)
      self.class_S[k]    = np.dot( self.class_X_normed[k].T, self.class_X_normed[k] ).astype(float)
      
      self.Sw += self.class_S[k]
      
    self.mean_dif = self.class_mean[self.classes[1]]-self.class_mean[self.classes[0]]
    
    self.iSw = np.linalg.pinv( self.Sw + self.epsilon*np.eye(self.d) )
    self.w_prop_to = np.dot( self.iSw, self.mean_dif  )
    
    self.fitted = True
    #self.decision_boundary = self.transform( self.mean_dif )
    
  def transform( self, X ):
    if self.fitted is False:
      print "Fit first!!"
      return None
      
    return np.dot( X, self.w_prop_to )
    
  def predict( self, X ):
    x_proj_predicted = self.transform( X ) 
    
    y_predict = np.zeros( len(X), dtype=int)
    
    I0 = pp.find( x_proj_predicted < 0 )
    I1 = pp.find( x_proj_predicted >= 0 )
    
    y_predict[I0] = self.classes[0]
    y_predict[I1] = self.classes[1]
    
    return y_predict
    
    
       