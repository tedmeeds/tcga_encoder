
import tcga_encoder
import sys, os, yaml
import numpy as np
import scipy as sp
import pylab as pp
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from collections import *
import itertools

def chunks(l, n):
  #Yield successive n-sized chunks from l.
  for i in xrange(0, len(l), n):
    yield l[i:i + n]
        
def load_yaml( filename ):
  with open(filename, 'r') as f:
    data = yaml.load(f)
    return data
    
def check_and_mkdir( path_name, verbose = False ):
  ok = False
  if os.path.exists( path_name ) == True:
    ok = True
  else:
    if verbose:
      print "Making directory: ", path_name
    os.makedirs( path_name )
    ok = True
      
  return ok
  
def ReadH5( fullpath ):
  df = pd.read_hdf( fullpath )  
  return df
  
def OpenHdfStore(location, which_one, mode ):
  store_name = "%s.h5"%(which_one)
  check_and_mkdir( location )
  full_name = os.path.join( location, store_name )
  
  # I think we can just open in 'a' mode for both
  if os.path.exists(full_name) is False:
    print "OpenHdfStore: %s does NOT EXIST, opening in %s mode"%(full_name, mode)
    return pd.HDFStore( full_name, mode )   
  else:
    print "OpenHdfStore: %s does EXISTS, opening in %s mode"%(full_name, mode)
    return pd.HDFStore( full_name, mode )    
  
def CloseHdfStore(store):
  return store.close()
  
# a generator for batch ids
class batch_ids_maker:
  def __init__(self, batchsize, n, randomize = True):
    assert n >= batchsize, "Right now must have batchsize < n"
    
    self.randomize = randomize
    self.batchsize = batchsize    
    self.n         = n
    self.indices   = self.new_indices() 
    self.start_idx = 0

  def __iter__(self):
      return self

  def new_indices(self):
    if self.randomize:
      return np.random.permutation(self.n).astype(int)
    else:
      return np.arange(self.n,dtype=int)
    
  def next(self, weights = None ):
    if weights is not None:
      return self.weighted_next( weights )
      
    if self.start_idx+self.batchsize >= len(self.indices):
      keep_ids = self.indices[self.start_idx:]
      self.indices = np.hstack( (keep_ids, self.new_indices() ))
      self.start_idx = 0
      
    ids = self.indices[self.start_idx:self.start_idx+self.batchsize]
    self.start_idx += self.batchsize
    
    return ids
    
  def weighted_next( self, weights ):
    I = np.argsort( -weights )
    ids = self.indices[ I[:self.batchsize] ]
    return ids