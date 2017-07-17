from tcga_encoder.definitions.locations import *
from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *

class EverythingObject(object):
  def __init__(self):
    pass
    
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
  
  return Z

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