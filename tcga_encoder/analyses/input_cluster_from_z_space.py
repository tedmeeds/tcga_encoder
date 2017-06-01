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
  
  save_dir = os.path.join( results_path, "input_clustering" )
  check_and_mkdir(save_dir)
  
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
  barcodes = data_store["/CLINICAL/observed"][ data_store["/CLINICAL/observed"][["RNA","miRNA","METH","DNA"]].sum(1)==4 ].index.values
  
  Z=Z.loc[barcodes]
  Z_values = Z.values
  tissues = data_store["/CLINICAL/TISSUE"].loc[barcodes]
  
  rna   = np.log(1+data_store["/RNA/RSEM"].loc[ barcodes ])
  mirna = np.log(1+data_store["/miRNA/READS"].loc[ barcodes ])
  meth  = np.log(0.1+data_store["/METH/METH"].loc[ barcodes ])
  dna   = data_store["/DNA/channel/0"].loc[ barcodes ]
  
  
  tissue_names = tissues.columns
  tissue_idx = np.argmax( tissues.values, 1 )
  n = len(Z)
  n_tissues = len(tissue_names)
  
  rna_normed = rna; mirna_normed = mirna; meth_normed = meth
  for t_idx in range(n_tissues):
    t_query = tissue_idx == t_idx
    
    X = rna[t_query]
    X -= X.mean(0)
    X /= X.std(0)
    rna_normed[t_query] = X

    X = mirna[t_query]
    X -= X.mean(0)
    X /= X.std(0)
    mirna_normed[t_query] = X

    X = meth[t_query]
    X -= X.mean(0)
    X /= X.std(0)
    meth_normed[t_query] = X
    
  
  for z_idx in range(5):
    z_values = Z_values[:,z_idx]
    order_z = np.argsort(z_values)
    rna_sorted = pd.DataFrame( rna_normed.values[order_z,:], index = barcodes[order_z], columns = rna.columns )
    mirna_sorted = pd.DataFrame( mirna_normed.values[order_z,:], index = barcodes[order_z], columns = mirna.columns )
    meth_sorted = pd.DataFrame( meth_normed.values[order_z,:], index = barcodes[order_z], columns = meth.columns )
    pdb.set_trace()
  #
  # binses = [20,50,100,500]
  # for bins in binses:
  #   pp.figure()
  #   pp.hist( aucs_true.values.flatten(), bins, range=(0,1), normed=True, histtype="step", lw=3, label="True" )
  #   pp.hist( aucs_random.values.flatten(), bins, color="red",range=(0,1), normed=True, histtype="step", lw=3, label="Random" )
  #   #pp.plot( [0,1.0],[0.5,0.5], 'r-', lw=3)
  #   pp.legend()
  #   pp.xlabel("Area Under the ROC")
  #   pp.ylabel("Pr(AUC)")
  #   pp.title("Comparison between AUC using latent space and random")
  #   pp.savefig( tissue_dir + "/auc_comparison_%dbins.png"%(bins), format='png', dpi=300 )
  #
  # pp.close('all')
  #pdb.set_trace()
  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )