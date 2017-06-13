from tcga_encoder.utils.helpers import *
from tcga_encoder.utils.math_funcs import *

def selu(x, name ):
  alpha = 1.67326324235
  scale = 1.05070098735548
  return scale*tf.where(x>=0.0, x, alpha*tf.exp(x)-alpha)