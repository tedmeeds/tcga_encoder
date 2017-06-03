from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.data.pathway_data import Pathways
from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
#from tcga_encoder.algorithms import *
import seaborn as sns
from sklearn.manifold import TSNE, locally_linear_embedding

def find_keepers(z, X, name, nbr2keep):
  inner = pd.Series( np.dot( z, X ), index = X.columns, name=name )
  inner.sort_values(inplace=True)
  inner = inner / np.max(np.abs(inner))
  #signed = np.sign( inner )
  abs_inner = np.abs( inner )
  ordered = np.argsort( -abs_inner.values )
  ordered = pd.Series( inner.values[ordered], index =inner.index[ordered],name=name )
  
  keepers = ordered[:nbr2keep]
  keepers = keepers.sort_values()
  
  return keepers
  
def main( data_location, results_location ):
  pathway_info = Pathways()
  data_path    = os.path.join( HOME_DIR ,data_location ) #, "data.h5" )
  results_path = os.path.join( HOME_DIR, results_location )
  
  data_filename = os.path.join( data_path, "data.h5")
  fill_filename = os.path.join( results_path, "full_vae_fill.h5" )
  
  save_dir = os.path.join( results_path, "input_clustering" )
  check_and_mkdir(save_dir)
  z_dir = os.path.join( save_dir, "z_pics" )
  check_and_mkdir(z_dir)
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
    
  nbr = 10
  Z_keep_rna=[]
  Z_keep_mirna=[]
  Z_keep_meth=[]
  for z_idx in range(n_z):
    z_values = Z_values[:,z_idx]
    order_z = np.argsort(z_values)
    rna_sorted = pd.DataFrame( rna_normed.values[order_z,:], index = barcodes[order_z], columns = rna.columns )
    mirna_sorted = pd.DataFrame( mirna_normed.values[order_z,:], index = barcodes[order_z], columns = mirna.columns )
    meth_sorted = pd.DataFrame( meth_normed.values[order_z,:], index = barcodes[order_z], columns = meth.columns )
    #
    # inner_rna = pd.Series( np.dot( z_values, rna_normed ), index = rna_normed.columns, name="rna" )
    # inner_rna.sort_values(inplace=True)
    # inner_rna = inner_rna / np.max(np.abs(inner_rna))
    # sign_rna = np.sign( inner_rna )
    # abs_rna = np.abs( inner_rna )
    # ordered_rna = np.argsort( -abs_rna.values )
    # ordered_rna = pd.Series( inner_rna.values[ordered_rna], index =inner_rna.index[ordered_rna],name="rna" )
    #
    # keep_rna = ordered_rna[:nbr]
    # keep_rna = keep_rna.sort_values()
    
    keep_rna = find_keepers( z_values, rna_normed, "z_%d"%(z_idx), nbr )
    keep_mirna = find_keepers( z_values, mirna_normed, "z_%d"%(z_idx), nbr )
    keep_meth = find_keepers( z_values, meth_normed, "z_%d"%(z_idx), nbr )
    
    keep_rna_big = find_keepers( z_values, rna_normed, "z_%d"%(z_idx), 3*nbr )
    keep_mirna_big = find_keepers( z_values, mirna_normed, "z_%d"%(z_idx), 3*nbr )
    keep_meth_big = find_keepers( z_values, meth_normed, "z_%d"%(z_idx), 3*nbr )
    
    Z_keep_rna.append( keep_rna )
    Z_keep_mirna.append( keep_mirna )
    Z_keep_meth.append( keep_meth )
    
    f = pp.figure( figsize = (14,8))
    ax1 = f.add_subplot(234);ax2 = f.add_subplot(235);ax3 = f.add_subplot(236)
    ax_pie1 = f.add_subplot(231); ax_pie3 = f.add_subplot(233)
    
    h1=keep_rna.plot(kind='bar',ax=ax1); h1.set_ylim(-1,1); ax1.set_title("RNA")
    h2=keep_mirna.plot(kind='bar',ax=ax2);h2.set_ylim(-1,1);ax2.set_title("miRNA")
    h3=keep_meth.plot(kind='bar',ax=ax3);h3.set_ylim(-1,1);ax3.set_title("METH")
    
    rna_kegg,rna_readable = pathway_info.Enrichment(keep_rna_big.index)
    meth_kegg,meth_readable = pathway_info.Enrichment(keep_meth_big.index)
    
    rna_readable.name=""
    meth_readable.name=""
    rna_readable[:8].plot.pie( ax=ax_pie1, fontsize=8 )
    meth_readable[:8].plot.pie( ax=ax_pie3, fontsize =8 )
    #pp.show()
    #pdb.set_trace()
    
    #f.suptitle( "z %d"%(z_idx) ); 
    f.subplots_adjust(bottom=0.5);
    pp.savefig( z_dir + "/z%d.png"%(z_idx), format="png", dpi=300 )
    #print h
    pp.close('all')
    #pdb.set_trace()
    #kept_rna = pd.DataFrame( rna_sorted[keep_rna.index], index=rna_sorted.index, columns = keep_rna.index )
    #kept_mirna = pd.DataFrame( mirna_sorted[keep_mirna.index], index=mirna_sorted.index, columns = keep_mirna.index )
    #kept_meth = pd.DataFrame( meth_sorted[keep_meth.index], index=meth_sorted.index, columns = keep_meth.index )
  merged_rna   = pd.concat(Z_keep_rna,axis=1) 
  merged_mirna = pd.concat(Z_keep_mirna,axis=1) 
  merged_meth  = pd.concat(Z_keep_meth,axis=1)  
  
  merged_rna.to_csv( save_dir + "/z_to_rna.csv" )
  merged_mirna.to_csv( save_dir + "/z_to_mirna.csv" )
  merged_meth.to_csv( save_dir + "/z_to_meth.csv" )
  
  f = sns.clustermap(merged_rna.fillna(0), figsize=(8,6))
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), fontsize=8)
  pp.savefig( save_dir + "/clustermap_z_to_rna.png", format="png", dpi=300 )
  f = sns.clustermap(merged_mirna.fillna(0), figsize=(8,6))
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), fontsize=8)
  pp.savefig( save_dir + "/clustermap_z_to_mirna.png", format="png", dpi=300 )
  
  f = sns.clustermap(merged_meth.fillna(0), figsize=(8,6))
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), fontsize=8)
  pp.savefig( save_dir + "/clustermap_z_to_meth.png", format="png", dpi=300 )
  
  
  #pdb.set_trace()
    #pdb.set_trace()
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