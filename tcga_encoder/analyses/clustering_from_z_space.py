from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
#from tcga_encoder.algorithms import *
import seaborn as sns
from sklearn.manifold import TSNE, locally_linear_embedding

def main( data_location, results_location ):
  data_path    = os.path.join( HOME_DIR ,data_location ) #, "data.h5" )
  results_path = os.path.join( HOME_DIR, results_location )
  
  data_filename = os.path.join( data_path, "data.h5")
  fill_filename = os.path.join( results_path, "full_vae_fill.h5" )
  
  save_dir = os.path.join( results_path, "clustering_with_z" )
  check_and_mkdir(save_dir)
  size_per_unit = 0.25
  print "HOME_DIR: ", HOME_DIR
  print "data_filename: ", data_filename
  print "fill_filename: ", fill_filename
  
  print "LOADING stores"
  data_store = pd.HDFStore( data_filename, "r" )
  fill_store = pd.HDFStore( fill_filename, "r" )
  
  Z_train = fill_store["/Z/TRAIN/Z/mu"]
  Z_val = fill_store["/Z/VAL/Z/mu"]
  
  Z = np.vstack( (Z_train.values, Z_val.values) )
  n_z = Z.shape[1]
  #pdb.set_trace()
  z_names = ["z_%d"%z_idx for z_idx in range(Z.shape[1])]
  Z = pd.DataFrame( Z, index = np.hstack( (Z_train.index.values, Z_val.index.values)), columns = z_names )
  
  barcodes = np.union1d( Z_train.index.values, Z_val.index.values )
  #quantiles = (len(Z)*np.array( [0,0.2, 0.4,0.6,0.8,1.0] )).astype(int)
  quantiles = (len(Z)*np.array( [0,0.1, 0.2,0.3,0.4,0.6,0.7,0.8,0.9,1.0] )).astype(int)
  n_quantiles = len(quantiles)-1
  start_q_id = -(n_quantiles-1)/2
  Z=Z.loc[barcodes]
  Z_values = Z.values
  
  argsort_Z = np.argsort( Z_values, 0 )
  
  Z_quantized = np.zeros( Z_values.shape, dtype=int )
  for start_q, end_q in zip( quantiles[:-1], quantiles[1:] ):
    for z_idx in range(n_z):
      z_idx_order = argsort_Z[:,z_idx] 
      Z_quantized[ z_idx_order[start_q:end_q], z_idx] = start_q_id
    start_q_id+=1
    
  Z_quantized = pd.DataFrame(Z_quantized, index=barcodes, columns=z_names )
  Z_quantized.to_csv( save_dir + "/Z_quantized.csv")
  
  sub_bcs = np.array([ x+"_"+y for x,y in np.array(data_store["/CLINICAL/data"]["patient.stage_event.pathologic_stage"].index.tolist(),dtype=str)] )
  sub_values = np.array( data_store["/CLINICAL/data"]["patient.stage_event.pathologic_stage"].values, dtype=str )
  subtypes = pd.Series( sub_values, index = sub_bcs, name="subtypes")
  
  
  #pdb.set_trace()
  tissues = data_store["/CLINICAL/TISSUE"].loc[barcodes]
  
  tissue_names = tissues.columns
  tissue_idx = np.argmax( tissues.values, 1 )

  n = len(Z)
  n_tissues = len(tissue_names)
  n_trials = 2*n_z
  trial_names = ["r_%d"%(trial_idx) for trial_idx in range(n_trials)]
  
  aucs_true  = np.ones( (n_tissues,n_z), dtype=float)
  aucs_random  = np.ones( (n_tissues,n_trials), dtype=float)
  
  true_y = np.ones(n, dtype=int)
  for t_idx in range(n_tissues):
    tissue_name = tissue_names[t_idx]
    
    print "working %s"%(tissue_name)
    
    t_ids_cohort = tissue_idx == t_idx
    n_tissue = np.sum(t_ids_cohort)
 
    if n_tissue < 1:
      continue
    
    Z_cohort = Z_quantized[ t_ids_cohort ]
    
    bcs = barcodes[t_ids_cohort]
    
    cohort_subtypes = subtypes.loc[bcs]
    subtype_names = np.unique(cohort_subtypes.values)
    subtype2colors = OrderedDict( zip(subtype_names,sns.color_palette("Blues", len(subtype_names))) ) 
    subtype_colors = np.array( [subtype2colors[subtype] for subtype in cohort_subtypes.values] ) 

    #import scipy.spatial.distance.pdist
    from scipy.spatial.distance import pdist, squareform
    from scipy.spatial.distance import squareform
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    d_mat = pdist( Z_cohort.values )
    s_form = squareform(d_mat)
    csr = csr_matrix(np.triu(s_form))
    Tcsr = minimum_spanning_tree(csr)
    pdb.set_trace()
    # f = pp.figure()
    # ax = f.add_subplot(111)
    #
    # size1 = max( int( n_z*size_per_unit ), 20 )
    # size2 = min( max( int( n_tissue*size_per_unit ), 12 ), 20 )
    #
    # #
    # # if len(subtype_names)>1:
    # #   h = sns.clustermap( Z_cohort, square=False, figsize=(size1,size2), row_colors = subtype_colors  )
    # # else:
    # #   h = sns.clustermap( Z_cohort, square=False, figsize=(size1,size2)  )
    # # pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    # # pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    # # pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
    # # pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
    # # h.ax_row_dendrogram.set_visible(False)
    # # h.ax_col_dendrogram.set_visible(False)
    # # h.cax.set_visible(False)
    #
    #
    # pp.savefig( save_dir + "/Z_clustermap_%s.png"%(tissue_name), fmt="png", dpi=300, bbox_inches='tight')
    # pp.close('all')


  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )