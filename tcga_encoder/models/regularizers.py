import tensorflow as tf
from scipy import stats

def Drop( X, keep_rate ):
  keep_mask = stats.bernoulli( keep_rate).rvs( X.shape )
  Y = X*keep_mask + (1-keep_mask)*0
  return Y

def DropColumns( X, cols2drop ):
  Y = X.copy()
  Y[:,cols2drop] = 0
  return Y  
  
class Regularizer(object):
  def __init__( self, lam_value ):
    self.lam = lam_value
    
  def Apply( self, w ):
    raise NotImplemented, "Must derive class"
    
class L2Regularizer(Regularizer):
  def Apply( self, w ):
    return self.lam*tf.reduce_sum( tf.square(w) )
    
class L1Regularizer(Regularizer):
  def Apply( self, w ):
    return self.lam*tf.reduce_sum( tf.abs(w) )
    
class LqRegularizer(Regularizer):
  def __init__( self, lam_value, q ):
    self.lam = lam_value
    self.q   = q
    self.eps = 1e-6
    
  def Apply( self, w ):
    return self.lam*tf.reduce_sum( tf.pow( tf.abs(w), self.q ) + self.eps )
    
class SortedL1RegularizerAxis2(Regularizer):
  def __init__( self, lam_value ):
    self.lam = lam_value
    #self.q   = q
    self.eps = 1e-6
    
  def Apply( self, w ):
    return self.lam*tf.reduce_sum( tf.abs(w[:,:,1:]-w[:,:,:-1]) + self.eps )
    
class SortedL1RegularizerAxis1(Regularizer):
  def __init__( self, lam_value ):
    self.lam = lam_value
    #self.q   = q
    self.eps = 1e-6
    
  def Apply( self, w ):
    return self.lam*tf.reduce_sum( tf.abs(w[:,1:,:]-w[:,:-1,:]) + self.eps )  
    
class SortedL1RegularizerAxis0(Regularizer):
  def __init__( self, lam_value ):
    self.lam = lam_value
    #self.q   = q
    self.eps = 1e-6
    
  def Apply( self, w ):
    return self.lam*tf.reduce_sum( tf.abs(w[1:,:,:]-w[:-1,:,:]) + self.eps )
    
class SortedAbsL1RegularizerAxis2(Regularizer):
  def __init__( self, lam_value ):
    self.lam = lam_value
    #self.q   = q
    self.eps = 1e-6
    
  def Apply( self, w ):
    aw = tf.abs(w)
    return self.lam*tf.reduce_sum( tf.abs(aw[:,:,1:]-aw[:,:,:-1]) + self.eps )
    
class SortedAbsL1RegularizerAxis1(Regularizer):
  def __init__( self, lam_value ):
    self.lam = lam_value
    #self.q   = q
    self.eps = 1e-6
    
  def Apply( self, w ):
    aw = tf.abs(w)
    return self.lam*tf.reduce_sum( tf.abs(aw[:,1:,:]-aw[:,:-1,:]) + self.eps )  
    
class SortedAbsL1RegularizerAxis0(Regularizer):
  def __init__( self, lam_value ):
    self.lam = lam_value
    #self.q   = q
    self.eps = 1e-6
    
  def Apply( self, w ):
    aw = tf.abs(w)
    return self.lam*tf.reduce_sum( tf.abs(aw[1:,:,:]-aw[:-1,:,:]) + self.eps )
    
# ------------------    
class SortedL2RegularizerAxis2(Regularizer):
  def __init__( self, lam_value ):
    self.lam = lam_value
    #self.q   = q
    self.eps = 1e-6
    
  def Apply( self, w ):
    return self.lam*tf.reduce_sum( tf.square(w[:,:,1:]-w[:,:,:-1]) + self.eps )
    
class SortedL2RegularizerAxis1(Regularizer):
  def __init__( self, lam_value ):
    self.lam = lam_value
    #self.q   = q
    self.eps = 1e-6
    
  def Apply( self, w ):
    return self.lam*tf.reduce_sum( tf.square(w[:,1:,:]-w[:,:-1,:]) + self.eps )  
    
class SortedL2RegularizerAxis0(Regularizer):
  def __init__( self, lam_value ):
    self.lam = lam_value
    #self.q   = q
    self.eps = 1e-6
    
  def Apply( self, w ):
    return self.lam*tf.reduce_sum( tf.square(w[1:,:,:]-w[:-1,:,:]) + self.eps )
    
class SortedAbsL2RegularizerAxis2(Regularizer):
  def __init__( self, lam_value ):
    self.lam = lam_value
    #self.q   = q
    self.eps = 1e-6
    
  def Apply( self, w ):
    aw = tf.abs(w)
    return self.lam*tf.reduce_sum( tf.square(aw[:,:,1:]-aw[:,:,:-1]) + self.eps )
    
class SortedAbsL2RegularizerAxis1(Regularizer):
  def __init__( self, lam_value ):
    self.lam = lam_value
    #self.q   = q
    self.eps = 1e-6
    
  def Apply( self, w ):
    aw = tf.abs(w)
    return self.lam*tf.reduce_sum( tf.square(aw[:,1:,:]-aw[:,:-1,:]) + self.eps )  
    
class SortedAbsL2RegularizerAxis0(Regularizer):
  def __init__( self, lam_value ):
    self.lam = lam_value
    #self.q   = q
    self.eps = 1e-6
    
  def Apply( self, w ):
    aw = tf.abs(w)
    return self.lam*tf.reduce_sum( tf.square(aw[1:,:,:]-aw[:-1,:,:]) + self.eps )