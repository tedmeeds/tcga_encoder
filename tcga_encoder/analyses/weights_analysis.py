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
  model_filename = os.path.join( results_path, "full_vae_model.h5" )
  
  weights_dir = os.path.join( results_path, "weights" )
  check_and_mkdir(weights_dir)
  
  print "HOME_DIR: ", HOME_DIR
  print "data_filename: ", data_filename
  print "model_filename: ", model_filename
  
  print "LOADING stores"
  data_store = pd.HDFStore( data_filename, "r" )
  
  rna_genes = data_store["/RNA/FAIR"].columns
  meth_genes = data_store["/METH/FAIR"].columns
  mirna_hsas = data_store["/miRNA/FAIR"].columns
  
  model_store = pd.HDFStore( model_filename, "r" )
  
  input_sources = ["METH","RNA","miRNA"] 
  #f = pp.figure()
  n_sources = 3
  
  size_per_unit = 0.25
  
  post_fix = "_scaled"
  idx=1
  
  W = {}
  for w_idx, input_source in zip( range(n_sources-1), input_sources[:-1] ):
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
      rows = np.array( [ "M-%s"%g for g in meth_genes], dtype=str )
      print input_source, w.shape, len(rows), len(columns)
      W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
      
    if input_source == "TISSUE":
      rows = tissue_names
      print input_source, w.shape, len(rows), len(columns)
      W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
  model_store.close()
  
  #pp.show()
  
  cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
  
  W_all = pd.concat( W.values(), axis=0 )
  rownames = W_all.index.values
  
  W_corr_hidden = W_all.corr()
  W_corr_inputs = W_all.T.corr()
  
  n_hidden = len(W_corr_hidden)
  size = max( int( n_hidden*size_per_unit ), 12 )
  f = pp.figure(figsize=(size,size))
  ax=f.add_subplot(111)
  # mask = np.zeros_like(W_corr, dtype=np.bool)
  # mask[np.triu_indices_from(mask)] = True
  
  # htmap = sns.clustermap ( W_corr_hidden, cmap=cmap, square=True, figsize=(size,size) )
  # #htmap.set_yticklabels( list(rownames), rotation='horizontal', fontsize=8 )
  # #htmap.set_xticklabels( list(rownames), rotation='vertical', fontsize=8 )
  #
  # pp.setp(htmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  # pp.setp(htmap.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  # pp.setp(htmap.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  # pp.setp(htmap.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  #
  # pp.savefig( weights_dir + "/corr_heatmap_hidden.png", fmt="png", bbox_inches = "tight")
  #
  n_inputs = len(W_corr_inputs)
  # size = max( int( n_inputs*size_per_unit ), 12 )
  # print "SIZE=",size
  # f2 = pp.figure(figsize=(size,size))
  # ax2=f.add_subplot(111)
  #
  #
  # htmap2 = sns.clustermap ( W_corr_inputs, cmap=cmap, square=True, figsize=(size,size) )
  #
  # pp.setp(htmap2.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  # pp.setp(htmap2.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  # pp.setp(htmap2.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  # pp.setp(htmap2.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  #
  # pp.savefig( weights_dir + "/corr_heatmap_inputs.png", fmt="png", bbox_inches = "tight")
  #
  # ax3=f.add_subplot(111)
  # size1 = max( int( n_hidden*size_per_unit ), 12 )
  # size2 = max( int( n_inputs*size_per_unit ), 12 )
  # htmap3 = sns.clustermap ( W_all, cmap=cmap, square=True, figsize=(size1,size2) )
  # pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  # pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  # pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  # pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  #
  # pp.savefig( weights_dir + "/weights_heatmap.png", fmt="png", bbox_inches = "tight")

  ax3=f.add_subplot(111)
  size1 = max( int( n_hidden*size_per_unit ), 12 )
  size2 = max( int( n_inputs*size_per_unit ), 12 )
  cmap = sns.palplot(sns.light_palette((260, 75, 60), input="husl"))
  htmap3 = sns.clustermap ( np.abs(W_all), cmap=cmap, square=True, figsize=(size1,size2) )
  pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  
  pp.savefig( weights_dir + "/weights_abs_heatmap.png", fmt="png", bbox_inches = "tight")

  #
  #
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

  pp.close('all')
  #pdb.set_trace()
  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )