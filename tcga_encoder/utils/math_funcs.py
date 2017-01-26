import tensorflow as tf
import numpy as np
import pylab as pp
import scipy as sp
import os
from scipy import special

SQRT2 = np.sqrt(2.0)
LN2PI = np.log(2*np.pi)

def tf_log_beta( a, b ):
  return tf.lgamma(a)+tf.lgamma(b)-tf.lgamma(a+b)

def kl( m1, v1, m2, v2 ):
  lv1 = np.log( v1 + 1e-12 )
  lv2 = np.log( v2 + 1e-12 )
  #return = -0.5*(1 + self.z_logvar - log_p_var - tf.square(self.z_mu-p_mu)/p_var - self.z_var/p_var )
  return -0.5*(1 + lv1 - lv2 - np.square(m1-m2)/v2 - v1/v2 )
  
  