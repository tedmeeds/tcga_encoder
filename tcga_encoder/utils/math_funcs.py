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
