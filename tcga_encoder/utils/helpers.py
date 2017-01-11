import sys, os, yaml
import numpy as np
import scipy as sp
import pylab as pp
import pandas as pd
from collections import *

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