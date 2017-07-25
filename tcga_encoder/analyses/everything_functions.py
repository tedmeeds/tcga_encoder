from tcga_encoder.definitions.locations import *
from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.analyses.dna_functions import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def auc_and_pvalue( true_y, z_values ):
  n_1 = true_y.sum()
  n_0 = len(true_y) - n_1
  
  auc        = roc_auc_score( true_y, z_values )
  
  if auc < 0.5:
    se_auc     = auc_standard_error( auc, n_0, n_1 )
  else:
    se_auc     = auc_standard_error( auc, n_1, n_0 )
    
  
  se_random   = auc_standard_error( 0.5, n_1, n_0 )
  p_value    = auc_p_value( auc, 0.5, se_auc, se_random )
  
  return auc, p_value


class LogisticBinaryClassifierKFold(object):
  def __init__(self, K=5, random_state = None ):
    self.random_state = random_state
    self.K = K
    self.M = []
    for k in range(K):
      self.M.append( LogisticBinaryClassifier() )
    
  def fit_and_prob( self, y, X, penalty = 'l2', C = 0.0 ):
    print "LogisticBinaryClassifierKFold penalty/C = ", penalty, C
    self.folds = StratifiedKFold(n_splits=self.K, shuffle = True, random_state=self.random_state)
    
    y_prob = np.zeros( y.shape )
    k = 0
    for train_split, test_split in self.folds.split( X, y ):
      self.M[k].fit( y[train_split], X[train_split,:], penalty = penalty, C=C )
      y_est = self.M[k].prob( X[test_split,:] )
      #pdb.set_trace()
      # if np.any(np.isnan(y_est)):
      #   pdb.set_trace()
      y_prob[test_split] = y_est
    
    return y_prob

class LogisticBinaryClassifier(object):
  def __init__(self):
    pass
    
  def fit( self, y, X, penalty = 'l2', C = 0.0, fit_intercept=True, class_weight="balanced"  ):
    self.dim = X.shape[1]
    self.n = len(y)
    self.n_1 = y.sum()
    self.n_0 = self.n-self.n_1
    self.penalty = penalty
    self.C = C
    
    self.M = LogisticRegression(penalty=self.penalty,\
                                                     C=self.C, \
                                                     intercept_scaling=1.0, \
                                                     fit_intercept=fit_intercept, \
                                                     class_weight = class_weight)
    self.mean = X.mean(0)
    self.std = X.std(0)
    self.M.fit( self.normalize(X), y )
    self.coef_ = self.M.coef_

  def normalize(self,X):
    return X
    return (X-self.mean)/self.std
    
  def predict( self, X ):
    return self.M.predict(self.normalize(X)).astype(int)
  
  def prob( self, X ):
    return self.M.predict_proba(self.normalize(X))[:,1]
    
  def log_prob( self, X ):
    return self.M.predict_log_proba(self.normalize(X))

class GenerativeBinaryClassifierKFold(object):
  def __init__(self, K=5, random_state = None ):
    self.random_state = random_state
    self.K = K
    self.M = []
    for k in range(K):
      self.M.append( GenerativeBinaryClassifier() )
    
  def fit_and_prob( self, y, X, cov_type = "full", ridge = 0.0 ):
    print "GenerativeBinaryClassifierKFold ridge = ", ridge
    self.folds = StratifiedKFold(n_splits=self.K, shuffle = True, random_state=self.random_state)
    
    y_prob = np.zeros( y.shape )
    k = 0
    for train_split, test_split in self.folds.split( X, y ):
      #print "\t\t\tINFO (%s): running fold %d of %d"%(dna_gene,fold_idx, n_folds)
      self.M[k].fit( y[train_split], X[train_split,:], cov_type, ridge )
      y_est = self.M[k].prob( X[test_split,:] )
      
      if np.any(np.isnan(y_est)):
        pdb.set_trace()
      y_prob[test_split] = y_est
    
    return y_prob

class GenerativeBinaryClassifier(object):
  def __init__(self):
    pass
    
  def fit( self, y, X, cov_type = "full", ridge = 0.0 ):
    self.dim = X.shape[1]
    self.n = len(y)
    self.n_1 = y.sum()
    self.n_0 = self.n-self.n_1
    self.ridge = ridge
    
    self.pi_1 = float(self.n_1)/float(self.n)
    self.pi_0 = float(self.n_0)/float(self.n)
    
    self.log_pi_1 = np.log(self.pi_1)
    self.log_pi_0 = np.log(self.pi_0)
    
    self.class_1 = y==1
    self.class_0 = y==0
    
    self.class_1_ids = pp.find(self.class_1)
    self.class_0_ids = pp.find(self.class_0)
    
    self.mean_1 = X[self.class_1].mean(0)
    self.mean_0 = X[self.class_0].mean(0)
    
    if cov_type == "full":
      self.cov_1 = np.cov( X[self.class_1].T ) + self.ridge*np.eye(self.dim)
      self.cov_0 = np.cov( X[self.class_0].T ) + ridge*np.eye(self.dim)
    elif cov_type == "diag":
      self.cov_1 = np.diag( X[self.class_1].var(0) ) + self.ridge*np.eye(self.dim)
      self.cov_0 = np.diag( X[self.class_0].var(0) )+ self.ridge*np.eye(self.dim)
    elif cov_type == "shared":
      self.cov_1 = np.cov( X.T ) + self.ridge*np.eye(self.dim)
      self.cov_0 = self.cov_1
      
    self.inv_cov_1 = np.linalg.inv(self.cov_1)
    self.inv_cov_0 = np.linalg.inv(self.cov_0)
    #pdb.set_trace()
    
  
  def predict( self, X ):
    return self.prob(X).astype(int)
  
  def prob( self, X ):
    return np.exp(self.log_prob(X))
    
  def log_prob( self, X ):
    log_prob_1 = self.log_prob_class( X, self.log_pi_1, self.cov_1, self.inv_cov_1, self.mean_1 )
    log_prob_0 = self.log_prob_class( X, self.log_pi_0, self.cov_0, self.inv_cov_0, self.mean_0 )
    
    if np.any(np.isnan(log_prob_1)) or np.any(np.isnan(log_prob_0)):
      print self.mean_0
      print self.mean_1
      print self.cov_1
      print self.cov_0
      print self.ridge
      print np.cov( X[self.class_1].T )
      print np.cov( X[self.class_0].T )
      pdb.set_trace()
    #log_denom = np.log( np.exp(log_prob_1)+np.exp(log_prob_0))
    
    max_ = np.maximum( log_prob_1, log_prob_0 )
    log_denom = max_ + np.log( np.exp( log_prob_1-max_ )+np.exp( log_prob_0-max_ ))
    # if log_prob_1 > log_prob_0:
    #   log_denom = log_prob_1 + np.log( 1.0 + np.exp(log_prob_0-log_prob_1) )
    # else:
    #   log_denom = log_prob_0 + np.log( 1.0 + np.exp(log_prob_1-log_prob_0) )
    #pdb.set_trace()
    return log_prob_1 - log_denom
    
   
  def log_prob_class(self, X, log_pi, cov, invcov, mean ):
    dif = X-mean[np.newaxis,:]
    
    a = log_pi
    b = -0.5*np.log( np.linalg.det( 2*np.pi*cov))
    c = -0.5*np.sum( np.dot( dif, invcov )*dif, 1 )
    
    return a+b+c
    
    
class EverythingObject(object):
  def __init__(self):
    self.results = {}
    
    
def load_store( location, name, mode="r" ):
  store_path = os.path.join( HOME_DIR, location )
  store_name = os.path.join( store_path, name )
  
  return pd.HDFStore( store_name, mode )

def load_scaled_data( fill_store, barcodes ):
  
  RNA_scale = fill_store["/scaled/RNA"].loc[barcodes]
  miRNA_scale = fill_store["/scaled/miRNA"].loc[barcodes]
  METH_scale = fill_store["/scaled/METH"].loc[barcodes]
  
  return RNA_scale, miRNA_scale, METH_scale


def load_subtypes( data_store ):
  sub_bcs = np.array([ x+"_"+y for x,y in np.array(data_store["/CLINICAL/data"]["patient.stage_event.pathologic_stage"].index.tolist(),dtype=str)] )
  sub_values = np.array( data_store["/CLINICAL/data"]["patient.stage_event.pathologic_stage"].values, dtype=str )
  subtypes = pd.Series( sub_values, index = sub_bcs, name="subtypes")
  
  return subtypes

def load_latent( fill_store ):
  Z_train = fill_store["/Z/TRAIN/Z/mu"]
  Z_val = fill_store["/Z/VAL/Z/mu"]
  
  Z = pd.concat( [Z_train, Z_val], axis = 0 )
  
  Z_train = fill_store["/Z/TRAIN/Z/var"]
  Z_val = fill_store["/Z/VAL/Z/var"]
  
  Z_var = pd.concat( [Z_train, Z_val], axis = 0 )
  Z_std = np.sqrt(Z_var)
  return Z, Z_std

def load_hidden( fill_store, barcodes ):
  try:
    H = fill_store["hidden"].loc[barcodes]
  except:
    print "found no hidden"
    H = pd.DataFrame( [], index = barcodes )
  return H

def join_weights( W_hidden2z, W_hidden ):
  W = {}
  n_z = W_hidden2z.shape[1]
  columns = np.array( ["z_%d"%i for i in range(n_z)])
  
  for input_source, source_w in W_hidden.iteritems():
    #pdb.set_trace()
    W[ input_source ] = pd.DataFrame( np.dot( source_w, W_hidden2z ), index = source_w.index, columns = columns )
    #pdb.set_trace()
  return W
  
      
def get_hidden2z_weights( model_store ):
  layer = "rec_z_space"
  model_store.open()
  w = model_store[ "%s"%(layer) + "/W/w%d"%(0)].values
  model_store.close()
  return w
  
def get_hidden_weights( model_store, input_sources, data_store ):
  
  rna_genes = data_store["/RNA/FAIR"].columns
  meth_genes = ["M_"+s for s in data_store["/METH/FAIR"].columns]
  mirna_hsas = data_store["/miRNA/FAIR"].columns
  
  post_fix = "_scaled"
  idx=1
  n_sources = len(input_sources)
  W = {}
  for w_idx, input_source in zip( range(n_sources), input_sources ):
    w = model_store[ "rec_hidden" + "/W/w%d"%(w_idx)].values
    #pdb.set_trace()
    
    
    d,k = w.shape
    
    columns = np.array( ["h_%d"%i for i in range(k)])
    if input_source == "RNA":
      rows = rna_genes
      print input_source, w.shape, len(rows), len(columns)
      W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
      
    if input_source == "miRNA":
      rows = mirna_hsas
      print input_source, w.shape, len(rows), len(columns)
      W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
      
    if input_source == "METH":
      rows = meth_genes
      #rows = np.array( [ "M-%s"%g for g in meth_genes], dtype=str )
      print input_source, w.shape, len(rows), len(columns)
      W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
      
    if input_source == "TISSUE":
      rows = tissue_names
      print input_source, w.shape, len(rows), len(columns)
      W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
  model_store.close()
  return W
  
def quantize( Z, q_range =[0,0.2, 0.4,0.6,0.8,1.0] ):
  #n_z = len(Z)
  n_z = len(Z.columns)
  #quantiles = (len(Z)*np.array( [0,0.33, 0.66, 1.0] )).astype(int)
  quantiles = (len(Z)*np.array( q_range )).astype(int)
  #quantiles = (len(Z)*np.array( [0,0.1, 0.2,0.3,0.4,0.6,0.7,0.8,0.9,1.0] )).astype(int)
  n_quantiles = len(quantiles)-1
  start_q_id = -(n_quantiles-1)/2
  #Z=Z.loc[barcodes]
  Z_values = Z.values
  
  argsort_Z = np.argsort( Z_values, 0 )
  
  Z_quantized = np.zeros( Z_values.shape, dtype=int )
  for start_q, end_q in zip( quantiles[:-1], quantiles[1:] ):
    for z_idx in range(n_z):
      z_idx_order = argsort_Z[:,z_idx] 
      Z_quantized[ z_idx_order[start_q:end_q], z_idx] = start_q_id
    start_q_id+=1
    
  Z_quantized = pd.DataFrame(Z_quantized, index=Z.index, columns=Z.columns )
  
  return Z_quantized

def normalize( Z  ):
  
  Z_values = Z.values
  Z_values -= Z_values.mean(0)
  Z_values /= Z_values.std(0)
  Z_normalized = pd.DataFrame(Z_values, index=Z.index, columns=Z.columns )
  
  return Z_normalized
  
def normalize_by_tissue(X,T):
  XV = X.values
  #X2 = 
  for tissue_name in T.columns:
    #print "working ", tissue_name
    ids = pp.find( T[tissue_name]==1 )
    n_ids = len(ids); n_tissue=n_ids
    if n_ids==0:
      continue
      
    XV[ids,:] -= XV[ids,:].mean(0)
    XV[ids,:] /= XV[ids,:].std(0)
    
  return pd.DataFrame( XV, index = X.index, columns = X.columns )
  
def ids_with_at_least_n_mutations( dna, tissue, n = 1 ):
  ok_ids = np.zeros( len(dna), dtype=bool )
  for tissue_name in tissue.columns:
    #print "working ", tissue_name
    ids = pp.find( tissue[tissue_name]==1 )
    n_ids = len(ids); n_tissue=n_ids
    if n_ids==0:
      continue
    
    n_mutations = dna[ids].sum()
    
    if n_mutations >= 1:
      ok_ids[ ids ] = True
  return ok_ids
  
def ids_with_at_least_p_mutations( dna, tissue, p = 1 ):
  ok_ids = np.zeros( len(dna), dtype=bool )
  relevant_tissues=[]
  for tissue_name in tissue.columns:
    #print "working ", tissue_name
    ids = pp.find( tissue[tissue_name]==1 )
    n_ids = len(ids); n_tissue=n_ids
    if n_ids==0:
      continue
    
    n_mutations = dna[ids].sum()
    
    if float(n_mutations)/float(n_ids) >= p:
      ok_ids[ ids ] = True
      relevant_tissues.append(tissue_name)
  return ok_ids,relevant_tissues
#
# def auc_standard_error( theta, nA, nN ):
#   # from: Hanley and McNeil (1982), The Meaning and Use of the Area under the ROC Curve
#   # theta: estimated AUC, can be 0.5 for a random test
#   # nA size of population A
#   # nN size of population N
#
#   Q1=theta/(2.0-theta); Q2=2*theta*theta/(1+theta)
#
#   SE = np.sqrt( (theta*(1-theta)+(nA-1)*(Q1-theta*theta) + (nN-1)*(Q2-theta*theta) )/(nA*nN) )
#
#   return SE

      
    
  
  
    