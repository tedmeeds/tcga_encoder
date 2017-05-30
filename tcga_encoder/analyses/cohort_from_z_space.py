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
  
  tissue_dir = os.path.join( results_path, "tissue_prediction" )
  check_and_mkdir(tissue_dir)
  
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
  
  Z=Z.loc[barcodes]
  Z_values = Z.values
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
    t_ids_cohort = tissue_idx == t_idx
    t_ids_other  = tissue_idx != t_idx
    
    true_y[t_ids_cohort] = 1
    true_y[t_ids_other]  = 0
    
    tissue_name = tissue_names[t_idx]
    
    print "working %s"%(tissue_name)
    #
    n_tissue = len(t_ids_cohort)
    
    if n_tissue < 1:
      continue
      
    for z_idx in range(n_z):
      z = Z_values[:,z_idx]
      aucs_true[t_idx,z_idx] = roc_auc_score( true_y, z )
      
    for trial_idx in range(n_trials):
      I = np.random.permutation(n_tissue)
      z = Z_values[:,z_idx][I]
      aucs_random[t_idx,trial_idx] = roc_auc_score( true_y, z )
      
    #events = Z["E"].loc[]
  
  #
  aucs_true  = pd.DataFrame( aucs_true, index = tissue_names, columns=z_names )
  aucs_random = pd.DataFrame( aucs_random, index = tissue_names, columns=trial_names )
  
  aucs_true.to_csv( tissue_dir + "/aucs_true.csv" )
  aucs_random.to_csv( tissue_dir + "/aucs_random.csv" )
  
  binses = [20,50,100,500]
  for bins in binses:
    pp.figure()
    pp.hist( aucs_true.values.flatten(), bins, range=(0,1), normed=True, histtype="step", lw=3, label="True" )
    pp.hist( aucs_random.values.flatten(), bins, color="red",range=(0,1), normed=True, histtype="step", lw=3, label="Random" )
    #pp.plot( [0,1.0],[0.5,0.5], 'r-', lw=3)
    pp.legend()
    pp.xlabel("Area Under the ROC")
    pp.ylabel("Pr(AUC)")
    pp.title("Comparison between AUC using latent space and random")
    pp.savefig( tissue_dir + "/auc_comparison_%dbins.png"%(bins), format='png', dpi=300 )

  pp.close('all')
  #pdb.set_trace()
  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )