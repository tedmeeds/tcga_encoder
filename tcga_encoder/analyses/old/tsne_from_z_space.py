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
  
  tsne_dir = os.path.join( results_path, "tsne" )
  check_and_mkdir(tsne_dir)
  
  print "HOME_DIR: ", HOME_DIR
  print "data_filename: ", data_filename
  print "fill_filename: ", fill_filename
  
  print "LOADING stores"
  data_store = pd.HDFStore( data_filename, "r" )
  fill_store = pd.HDFStore( fill_filename, "r" )
  
  Z_train = fill_store["/Z/TRAIN/Z/mu"]
  Z_val = fill_store["/Z/VAL/Z/mu"]
  tissue = data_store["/CLINICAL/TISSUE"]
  
  barcodes_train = Z_train.index.values
  tissue_train = data_store["/CLINICAL/TISSUE"].loc[barcodes_train]
  
  tissues = tissue_train.columns
  tissue_idx = np.argmax( tissue_train.values, 1 )
  #pdb.set_trace()
  #class sklearn.manifold.TSNE(n_components=2, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000, n_iter_without_progress=30, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)
  print "Running TSNE"
  
  Z_normed = Z_train.values# - Z_train.values.mean(0)
  #Z_normed = Z_normed - Z_train.values.std(0)
  Z_normed = Z_normed[:,:100]
  perplexity=30
  nbr = 2000
  np.random.seed(1)
  I = np.random.permutation( len(Z_normed ))[:nbr]
  
  tsne = TSNE(n_components=3,verbose=1, learning_rate=1000, perplexity=perplexity, method='exact')
  #embedded,dummy = locally_linear_embedding(Z_normed[I,:], n_neighbors=10, n_components=4)
  n_components =5
  w = np.random.randn( Z_normed.shape[1],n_components)
  embedded = np.dot( Z_normed[I,:], w )
  #pdb.set_trace()
  
  np.savetxt(  tsne_dir + "/z.csv", Z_normed[I,:], fmt='%.3f',delimiter=',')
  labels = [tissues[idx] for idx in tissue_idx[I]]
  np.savetxt(  tsne_dir + "/labels.csv", labels, fmt='%s',delimiter=',')
  embedded = tsne.fit_transform( embedded )
  
  
  print "DONE!"
   
  # z_2d = bh_sne(Z_n,perplexity=30)
  
  colors = "bgrkycmbgrkycmbgrkycmbgrkycmbgrkycmbgrkycmbgrkycmbgrkycmbgrkycm"
  markers = "ooooooosssssssvvvvvvvppppppphhhhhhhDDDDDDDooooooosssssssvvvvvvvppppppphhhhhhhDDDDDDD"
  pp.figure( figsize=(12,12))
  for t_idx in range( len(tissues) ):
    ids = tissue_idx[I] == t_idx 
    #'o', mec="r", mew="2",ms=30,fillstyle="none"
    if len(ids) >=10:
      pp.plot( embedded[ids,:][:10,0], embedded[ids,:][:10,1], markers[t_idx], mec=colors[t_idx], mew="2", ms=10, fillstyle="none", alpha=0.5 )
  #pp.show()
  
  pp.savefig( tsne_dir + "/tsne_perplexity_%d.png"%(perplexity), format='png', dpi=300 )
  
  #pdb.set_trace()
  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )