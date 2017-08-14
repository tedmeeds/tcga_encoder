from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
#from tcga_encoder.algorithms import *
import seaborn as sns
from sklearn.manifold import TSNE, locally_linear_embedding

from scipy import stats

def auc_standard_error( theta, nA, nN ):
  # from: Hanley and McNeil (1982), The Meaning and Use of the Area under the ROC Curve
  # theta: estimated AUC, can be 0.5 for a random test
  # nA size of population A
  # nN size of population N
  
  Q1=theta/(2.0-theta); Q2=2*theta*theta/(1+theta)
  
  SE = np.sqrt( (theta*(1-theta)+(nA-1)*(Q1-theta*theta) + (nN-1)*(Q2-theta*theta) )/(nA*nN) )
  
  return SE
  
def auc_p_value( auc1, auc2, std_error1, std_error2 ):
  
  se_combined = np.sqrt( std_error1**2 + std_error2**2 )
  
  difference = auc1 - auc2
  z_values = difference / se_combined 
  sign_difference = np.sign(difference)
  
  #p_values = 1.0 - stats.norm.cdf( np.abs(z_values) ) 
  p_values = 1.0 - stats.norm.cdf( z_values ) 
  
  return p_values
  
  
  
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
  aucs_true_best  = np.ones( (n_tissues,n_z), dtype=float)
  aucs_random_best  = np.ones( (n_tissues,n_trials), dtype=float)

  z_true_se       = np.ones( (n_tissues,n_z), dtype=float)
  z_true_p_value  = np.ones( (n_tissues,n_z), dtype=float)
  
  null_se  = np.ones( (n_tissues), dtype=float)
  
  null_auc=0.5
  true_y = np.ones(n, dtype=int)
  for t_idx in range(n_tissues):
    t_ids_cohort = tissue_idx == t_idx
    t_ids_other  = tissue_idx != t_idx
    
    true_y[t_ids_cohort] = 1
    true_y[t_ids_other]  = 0
    
    tissue_name = tissue_names[t_idx]
    if tissue_name == "gbm":
      print "skipping gbm"
      continue
    
    print "working %s"%(tissue_name)
    #
    n_tissue = len(t_ids_cohort)
    
    if n_tissue < 1:
      continue
    
    n1 =  true_y.sum()
    n0 = n_tissue - true_y.sum()
    null_se[t_idx] = auc_standard_error( null_auc, n1, n0 )
    for z_idx in range(n_z):
      z = Z_values[:,z_idx]
      aucs_true[t_idx,z_idx] = roc_auc_score( true_y, z )
      aucs_true_best[t_idx,z_idx] = max(aucs_true[t_idx,z_idx],1.0-aucs_true[t_idx,z_idx])
      
      z_true_auc = aucs_true[t_idx,z_idx]
      z_true_se[t_idx,z_idx] = auc_standard_error( z_true_auc, n1, n0 )
      
      z_true_p_value[t_idx,z_idx] = auc_p_value( null_auc, z_true_auc, null_se[t_idx], z_true_se[t_idx,z_idx] )
      
    for trial_idx in range(n_trials):
      I = np.random.permutation(n_tissue)
      z = Z_values[:,z_idx][I]
      aucs_random[t_idx,trial_idx] = roc_auc_score( true_y, z )
      aucs_random_best[t_idx,trial_idx] = max(aucs_random[t_idx,trial_idx],1.0-aucs_random[t_idx,trial_idx])
    #events = Z["E"].loc[]
  
  #
  aucs_true  = pd.DataFrame( aucs_true, index = tissue_names, columns=z_names )
  aucs_random = pd.DataFrame( aucs_random, index = tissue_names, columns=trial_names )
  #
  aucs_true_best  = pd.DataFrame( aucs_true_best, index = tissue_names, columns=z_names )
  aucs_random_best = pd.DataFrame( aucs_random_best, index = tissue_names, columns=trial_names )

  null_se         = pd.Series( null_se, index = tissue_names )
  z_true_se       = pd.DataFrame( z_true_se, index = tissue_names, columns=z_names )
  z_true_p_value  = pd.DataFrame( z_true_p_value, index = tissue_names, columns=z_names )
  
  aucs_true.drop("gbm",inplace=True)
  aucs_random.drop("gbm",inplace=True)
  aucs_true_best.drop("gbm",inplace=True)
  aucs_random_best.drop("gbm",inplace=True)
  z_true_se.drop("gbm",inplace=True)
  z_true_p_value.drop("gbm",inplace=True)
  null_se.drop("gbm",inplace=True)
  
  aucs_true.to_csv( tissue_dir + "/aucs_true.csv", index_label = "tissue" )
  aucs_random.to_csv( tissue_dir + "/aucs_random.csv", index_label = "tissue" )
  aucs_true_best.to_csv( tissue_dir + "/aucs_true_best.csv", index_label = "tissue" )
  aucs_random_best.to_csv( tissue_dir + "/aucs_random_best.csv", index_label = "tissue" )
  z_true_se.to_csv( tissue_dir + "/z_true_se.csv", index_label = "tissue" )
  z_true_p_value.to_csv( tissue_dir + "/z_true_p_value.csv", index_label = "tissue" )
  null_se.to_csv( tissue_dir + "/null_se.csv", index_label = "tissue" )
  
  binses = [20,50,100,500]
  for bins in binses:
    pp.figure()
    pp.hist( aucs_true.values.flatten(), bins=np.linspace(0,1,bins+1), normed=True, histtype="step", lw=2, label="True" )
    pp.hist( aucs_random.values.flatten(), bins=np.linspace(0,1,bins+1), color="red", normed=True, histtype="step", lw=2, label="Random" )
    #pp.plot( [0,1.0],[0.5,0.5], 'r-', lw=3)
    pp.legend()
    pp.xlabel("Area Under the ROC")
    pp.ylabel("Pr(AUC)")
    pp.title("Comparison between AUC using latent space and random")
    pp.savefig( tissue_dir + "/auc_comparison_%dbins.png"%(bins), format='png', dpi=300 )
    
    pp.figure()
    pp.hist( aucs_true_best.values.flatten(), bins=np.linspace(0.5,1,bins/2+1), normed=True, histtype="step", lw=2, label="True" )
    pp.hist( aucs_random_best.values.flatten(), bins=np.linspace(0.5,1,bins/2+1), color="red", normed=True, histtype="step", lw=2, label="Random" )
    #pp.plot( [0,1.0],[0.5,0.5], 'r-', lw=3)
    pp.legend()
    pp.xlabel("Area Under the ROC")
    pp.ylabel("Pr(AUC)")
    pp.title("Comparison between AUC using latent space and random")
    pp.savefig( tissue_dir + "/auc_comparison_%dbins_best.png"%(bins), format='png', dpi=300 )

    pp.figure()
    pp.plot( [0,1.0],[1.0,1.0], 'r--', lw=2)
    pp.hist( z_true_p_value.values.flatten(), bins=np.linspace(0,1,bins+1), normed=True, histtype="step", lw=2, label="True" )
    #pp.hist( aucs_random.values.flatten(), bins=np.linspace(0,1,bins+1), color="red", normed=True, histtype="step", lw=2, label="Random" )
    #pp.plot( [0,1.0],[0.5,0.5], 'r-', lw=3)
    pp.legend()
    pp.xlabel("p-value")
    pp.ylabel("Pr(p-value)")
    pp.title("Distribution of AUC p-values")
    pp.savefig( tissue_dir + "/auc_p_values_%dbins.png"%(bins), format='png', dpi=300 )


  pp.close('all')
  #pdb.set_trace()
  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )