
from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.analyses.everything_functions import *
from tcga_encoder.analyses.everything_long import *
from tcga_encoder.analyses.survival_functions import *
import networkx as nx
#try:
from networkx.drawing.nx_agraph import graphviz_layout as g_layout

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import squareform
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
#except:
#  print "could not import graphviz "
#from networkx.drawing.nx_agraph import spring_layout 
  
from scipy import stats  


if __name__ == "__main__":
  
  name1 = "RNA"
  name2 = "Z"

  dir_1_short = "results/tcga_vae_post_recomb9/medium/xval_nn_tissue/z_100_h_500_anti_100/fold_1_of_5/everything/correct_by_tissue_dna_and_RNA_tissue_0.00_p_spear_1_logreg"
  dir_2_short = "results/tcga_vae_post_recomb9/medium/xval_nn_tissue/z_100_h_500_anti_100/fold_1_of_5/everything/correct_by_tissue_dna_and_Z_tissue_0.00_p_spear_1_logreg"
  
  dir_1 = os.path.join( HOME_DIR, dir_1_short )
  dir_2 = os.path.join( HOME_DIR, dir_2_short )
  
  tissues_1 = np.array( os.listdir( dir_1 ), dtype=str )
  tissues_2 = np.array( os.listdir( dir_2 ), dtype=str )
  
  common_tissues = np.intersect1d( tissues_1, tissues_2 )
  
  w_auc_1 = 0.0; w_auc_2 = 0.0
  w_prc_1 = 0.0; w_prc_2 = 0.0
  w_s = 0.0
  
  weighted_tissues_auc = []
  weighted_tissues_prc = []
  
  for tissue in common_tissues:
    if tissue[-3:] == "png":
      continue
    tissue_dir_1 = os.path.join( dir_1, tissue )
    tissue_dir_2 = os.path.join( dir_2, tissue )
    
    perf1 = pd.read_csv( tissue_dir_1 + "/performance.csv", index_col = "measure" ).T
    perf2 = pd.read_csv( tissue_dir_2 + "/performance.csv", index_col = "measure" ).T
    
    common_genes = np.intersect1d( perf1.index.values, perf2.index.values )
    print tissue
    print perf1
    print perf2
    print common_genes
    
    #pdb.set_trace()
    auc1 = perf1[ "AUROC"]
    auc2 = perf2[ "AUROC"]
    pr1 = perf1[ "AUPRC"]
    pr2 = perf2[ "AUPRC"]
    
    min_auc = max( min( auc1.min(), auc2.min() ) - 0.1*min( auc1.min(), auc2.min() ), 0 )
    min_prc = max( min( pr1.min(), pr2.min() ) - 0.1*min( pr1.min(), pr2.min() ), 0 )
    max_auc = min( max( auc1.max(), auc2.max() ) + 0.1*max( auc1.max(), auc2.max() ), 1 )
    max_prc = min( max( pr1.max(), pr2.max() ) + 0.1*max( pr1.max(), pr2.max() ), 1 )
    
    n = perf1[ "n"].values;
    mutations = perf1[ "mutations"].values; 
    #pan_muts = mutations #mutations.loc["PAN"]
    wildtypes = perf1[ "wildtype"].values; 
    #pan_wild = wildtypes
    #n_pan = pan_wild + pan_muts
    
    real_weight = mutations.astype(float) / n.astype(float)
    
    weights = np.maximum(5, 50*real_weight )
    
    f = pp.figure(figsize=(12,6))
    ax_auc = f.add_subplot( 121 )
    ax_prc = f.add_subplot( 122 )
  
    ax_auc.plot( [0,1],[0,1], 'k--' )
    ax_prc.plot( [0,1],[0,1], 'k--' )
   
    for g_idx,gene in zip(range(len(common_genes)),common_genes):
      ax_auc.plot( auc1.values[g_idx], auc2.values[g_idx], 'o', ms=weights[g_idx], alpha=0.75 )
      ax_auc.text( auc1.values[g_idx], auc2.values[g_idx], common_genes[g_idx], fontsize=6 )
      ax_prc.plot( pr1.values[g_idx], pr2.values[g_idx], 'o', ms=weights[g_idx], alpha=0.75 )
      ax_prc.text( pr1.values[g_idx], pr2.values[g_idx], common_genes[g_idx], fontsize=6 )

    ax_auc.set_xlabel(  name1 ); ax_auc.set_ylabel(  name2 )
    ax_auc.set_title("AUROC")
    #ax_prc.legend(loc='right')
    
    ax_auc.set_xlim( min_auc, max_auc )
    ax_prc.set_xlim( min_prc, max_prc )
    ax_auc.set_ylim( min_auc, max_auc )
    ax_prc.set_ylim( min_prc, max_prc )
    ax_prc.set_xlabel(  name1 ); ax_auc.set_ylabel(  name2 )
    ax_prc.set_title("AUPRC")
    
    pp.suptitle(tissue)
    
    f.savefig( tissue_dir_1 + "/comparison.png", fmt='png', dpi=300)
    f.savefig( tissue_dir_2 + "/comparison.png", fmt='png', dpi=300)
    
    weighted_auc1 = np.dot( real_weight, auc1.values ) / real_weight.sum()
    weighted_auc2 = np.dot( real_weight, auc2.values ) / real_weight.sum()
    weighted_prc1 = np.dot( real_weight, pr1.values ) / real_weight.sum()
    weighted_prc2 = np.dot( real_weight, pr2.values ) / real_weight.sum()
    
    weighted_tissues_auc.append( pd.Series( [weighted_auc1,weighted_auc2], index = [name1,name2], name=tissue ) )
    weighted_tissues_prc.append( pd.Series( [weighted_prc1,weighted_prc2], index = [name1,name2], name=tissue ) )

    pp.close('all')
  
  weighted_aucs = pd.concat( weighted_tissues_auc, axis=1 ).T
  weighted_prcs = pd.concat( weighted_tissues_prc, axis=1 ).T
  #
  min_auc = max( min( weighted_aucs[name1].min(), weighted_aucs[name2].min() ) - 0.1*min( weighted_aucs[name1].min(), weighted_aucs[name2].min() ), 0 )
  min_prc = max( min( weighted_prcs[name1].min(), weighted_prcs[name2].min() ) - 0.1*min( weighted_prcs[name1].min(), weighted_prcs[name2].min() ), 0 )
  max_auc = min( max( weighted_aucs[name1].max(), weighted_aucs[name2].max() ) + 0.1*max( weighted_aucs[name1].max(), weighted_aucs[name2].max() ), 1 )
  max_prc = min( max( weighted_prcs[name1].max(), weighted_prcs[name2].max() ) + 0.1*max( weighted_prcs[name1].max(), weighted_prcs[name2].max() ), 1 )
  #
  #
  f = pp.figure(figsize=(12,6))
  ax_auc = f.add_subplot( 121 )
  ax_prc = f.add_subplot( 122 )

  ax_auc.plot( [0,1],[0,1], 'k--' )
  ax_prc.plot( [0,1],[0,1], 'k--' )
  #
  #
  for tissue in common_tissues:
    if tissue[-3:] == "png":
      continue
    #gene_dir1 = os.path.join( dir_1, gene )
    #gene_dir2 = os.path.join( dir_2, gene )

    auc1_ = weighted_aucs.loc[tissue][name1]; auc2_ = weighted_aucs.loc[tissue][name2]
    prc1_ = weighted_prcs.loc[tissue][name1]; prc2_ = weighted_prcs.loc[tissue][name2]

    #real_weight = float(mutations.loc[tissue]) / float( pan_muts )
    weight = 20 #max(5, 0.025*real_weight )

    ax_auc.plot( auc1_, auc2_, 'o', mec='k', mew=1, ms = weight, alpha=0.75, label = tissue.upper() )
    ax_prc.plot( prc1_, prc2_, 'o', mec='k', mew=1, ms = weight, alpha=0.75, label = tissue.upper()  )

    #if float(mutations.loc[tissue])  > float(0.05*pan_muts):
    ax_auc.text(  auc1_, auc2_, tissue.upper(), fontsize=6 )
    ax_prc.text( prc1_, prc2_, tissue.upper(), fontsize=6 )


  ax_auc.set_xlim( min_auc, max_auc )
  ax_prc.set_xlim( min_prc, max_prc )
  ax_auc.set_ylim( min_auc, max_auc )
  ax_prc.set_ylim( min_prc, max_prc )
  ax_prc.set_xlabel(  name1 ); ax_auc.set_ylabel(  name2 )
  ax_prc.set_title("AUPRC")
  ax_auc.set_title("AUROC")
  pp.suptitle("Weighted")

  f.savefig( dir_1 + "/comparison.png", fmt='png', dpi=300)
  f.savefig( dir_2 + "/comparison.png", fmt='png', dpi=300)
  #
  # #weighted_aucs.append( pd.Series( [w_auc_1, w_auc_2], index = [name1, name2], name=gene ) )
  # #weighted_prcs.append( pd.Series( [w_prc_1, w_prc_2], index = [name1, name2], name=gene ) )
  #weights.append( pan_muts )