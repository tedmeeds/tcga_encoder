import numpy as np
import scipy as sp
import pylab as pp

from collections import OrderedDict
from sklearn.neighbors import KernelDensity

class LinearDiscriminantAnalysis( object ):
  def __init__(self, epsilon = 1e-12 ):
    self.epsilon = epsilon # used for computing sw
    self.fitted = False
    
  def fit( self, X, y ):
    self.classes = np.unique( y )
    self.use_random_proj = False
    if len(self.classes) < 2:
      print "LDA fit only received one class!!"
      print "Using random projection"
      
      self.use_random_proj = True
      
    
    self.n, self.d = X.shape
    assert len(y) == self.n, "X and y should have same length"
    
    self.Sw             = np.zeros( (self.d,self.d), dtype=float )
    self.class_ids      = OrderedDict()
    self.class_X        = OrderedDict()
    self.class_mean     = OrderedDict()
    self.class_X_normed = OrderedDict()
    self.class_S        = OrderedDict()
    self.class_n        = OrderedDict()
    self.class_pi       = OrderedDict()
    
    for k in self.classes:
      self.class_ids[k]      = pp.find( y == k) 
      self.class_X[k]        = X[self.class_ids[k],:].astype(float)
      self.class_mean[k]     = self.class_X[k].mean(0).astype(float)
      self.class_X_normed[k] = X[self.class_ids[k],:].astype(float) - self.class_mean[k].astype(float)
      self.class_S[k]        = np.dot( self.class_X_normed[k].T, self.class_X_normed[k] ).astype(float)
      self.class_n[k]        = len(self.class_ids[k])
      self.class_pi[k]       = float(self.class_n[k])/self.n
      self.Sw               += self.class_S[k]
      
    if self.use_random_proj is False:
      self.mean_dif = self.class_mean[self.classes[1]]-self.class_mean[self.classes[0]]
    
      self.iSw = np.linalg.pinv( self.Sw + self.epsilon*np.eye(self.d) )
      self.w_prop_to = np.dot( self.iSw, self.mean_dif  )
      self.w_prop_to = self.w_prop_to / np.linalg.norm(self.w_prop_to)
      self.fitted = True
    else:
      self.mean_dif = self.class_mean[self.classes[0]]
      self.iSw = np.linalg.pinv( self.Sw + self.epsilon*np.eye(self.d) )
      self.w_prop_to = np.dot( self.iSw, self.mean_dif  )
      self.w_prop_to = self.w_prop_to / np.linalg.norm(self.w_prop_to)
      self.fitted = True
      
    self.fit_density()
    
    #self.decision_boundary = self.transform( self.mean_dif )
    
  def transform( self, X ):
    if self.fitted is False:
      print "Fit first!!"
      return None
      
    return np.dot( X, self.w_prop_to )
    

  def fit_density( self ):
    if self.use_random_proj is False:
      self.x_proj1 = self.transform( self.class_X[1] )
      self.x_proj0 = self.transform( self.class_X[0] )
      self.h1 = max(1e-12,np.std(self.x_proj1)*(4.0/3.0/self.class_n[1])**(1.0/5.0))
      self.h0 = max(1e-12,np.std(self.x_proj0)*(4.0/3.0/self.class_n[0])**(1.0/5.0))
    
    
    
      self.kde1 = KernelDensity(kernel='gaussian', bandwidth=self.h1).fit(self.x_proj1[:,np.newaxis])
      self.kde0 = KernelDensity(kernel='gaussian', bandwidth=self.h0).fit(self.x_proj0[:,np.newaxis])
    
      self.pi1 = self.class_pi[1]
      self.pi0 = self.class_pi[0]
    
      self.log_pi1 = np.log(self.pi1)
      self.log_pi0 = np.log(self.pi0)
    else:
      self.x_proj1 = self.transform( self.class_X[0] )
      self.x_proj0 = self.transform( self.class_X[0] )
      self.h1 = max(1e-12,np.std(self.x_proj1)*(4.0/3.0/self.class_n[0])**(1.0/5.0))
      self.h0 = max(1e-12,np.std(self.x_proj0)*(4.0/3.0/self.class_n[0])**(1.0/5.0))
    
    
    
      self.kde1 = KernelDensity(kernel='gaussian', bandwidth=self.h1).fit(self.x_proj1[:,np.newaxis])
      self.kde0 = KernelDensity(kernel='gaussian', bandwidth=self.h0).fit(self.x_proj0[:,np.newaxis])
    
      self.pi1 = 0.00001 #self.class_pi[0]
      self.pi0 = 1.0 #self.class_pi[0]
    
      self.log_pi1 = np.log(self.pi1)
      self.log_pi0 = np.log(self.pi0)

  def predict( self, X, ignore_pi = False ):
    x_proj_predicted = self.transform( X ) 
    
    y_predict = np.zeros( len(X), dtype=int)
    
    log_joint_0, log_joint_1 = self.log_joint( x_proj_predicted, ignore_pi=ignore_pi )
    
    I1 = pp.find( log_joint_1 >= log_joint_0 )
    I0 = pp.find( log_joint_1 < log_joint_0 )
    
    #I0 = pp.find( x_proj_predicted < 0 )
    #I1 = pp.find( x_proj_predicted >= 0 )
    
    y_predict[I0] = self.classes[0]
    y_predict[I1] = 1 #self.classes[1]
    
    return y_predict

  def log_prob_1( self, X, ignore_pi = False ):
    x_proj_predicted = self.transform( X ) 
    
    y_predict = np.zeros( len(X), dtype=int)
    
    log_joint_0, log_joint_1 = self.log_joint( x_proj_predicted, ignore_pi = ignore_pi )
    
    log_prob_1 = log_joint_1 - np.log( np.exp(log_joint_0) + np.exp(log_joint_1) )
        
    return log_prob_1
        
  def prob( self, X, ignore_pi = False ):
    x_proj_predicted = self.transform( X ) 
    
    y_predict = np.zeros( len(X), dtype=int)
    
    log_joint_0, log_joint_1 = self.log_joint( x_proj_predicted, ignore_pi = ignore_pi )
    
    log_prob_1 = log_joint_1 - np.log( np.exp(log_joint_0) + np.exp(log_joint_1) )
        
    return np.exp(log_prob_1)
        
  def log_joint( self, x_proj, ignore_pi = False ):
    
    log_prob_0 = self.kde0.score_samples( x_proj[:,np.newaxis] )
    log_prob_1 = self.kde1.score_samples( x_proj[:,np.newaxis] )
    
    if ignore_pi:
      # assume even counts
      log_joint_0 = log_prob_0
      log_joint_1 = log_prob_1
    else:
      log_joint_0 = self.log_pi0 + log_prob_0
      log_joint_1 = self.log_pi1 + log_prob_1
    
    return log_joint_0, log_joint_1
          
  def plot_joint_density( self, x_plot, ax = None, ignore_pi = False ):
    # log_prob_0 = self.kde0.score_samples( x_plot[:,np.newaxis] )
    # log_prob_1 = self.kde1.score_samples( x_plot[:,np.newaxis] )
    #
    # log_joint_0 = self.log_pi0 + log_prob_0
    # log_joint_1 = self.log_pi1 + log_prob_1
    
    log_joint_0, log_joint_1 = self.log_joint( x_plot, ignore_pi )
    
    I1 = pp.find( np.exp( log_joint_1) > 1e-3 )
    I0 = pp.find( np.exp( log_joint_0) > 1e-3)
    #I1 = pp.find( log_joint_1 >= log_joint_0 )
    #I0 = pp.find( log_joint_1 < log_joint_0 )
    if ax is None:
      ax = pp
      
    ax.plot( x_plot[I0], np.exp( log_joint_0[I0]), 'b-')
    ax.plot( x_plot[I1], np.exp( log_joint_1[I1]), 'r-')
    ax.scatter( self.x_proj1[:,np.newaxis], 0.1+0*self.x_proj1[:,np.newaxis], s=80, c="red", marker='+', linewidths=2)
    ax.scatter( self.x_proj0[:,np.newaxis], 0*self.x_proj0[:,np.newaxis], s=80, c="blue", marker='x', linewidths=2)
    

  
    
    
       