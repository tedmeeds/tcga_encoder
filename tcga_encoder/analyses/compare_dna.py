
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
  dir_1_short = "results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_100_h_1000_anti_5000/fold_1_of_50/everything/correct_deeper_meaning_dna_and_rna_fair_tissue_0.00_p_spear_1_logreg"
  dir_2_short = "results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_100_h_1000_anti_5000/fold_1_of_50/everything/correct_deeper_meaning_dna_and_z_p_tissue_0.00_p_spear_1_logreg"
  
  #dir_1_short = "results/tcga_vae_post_recomb9/medium/xval_nn_tissue/z_100_h_500_anti_100/fold_1_of_5/everything/correct_deeper_meaning_dna_and_rna_fair_tissue_0.00_p_spear_1_logreg"
  #dir_2_short = "results/tcga_vae_post_recomb9/medium/xval_nn_tissue/z_100_h_500_anti_100/fold_1_of_5/everything/correct_deeper_meaning_dna_and_z_p_tissue_0.00_p_spear_1_logreg"
  
  dir_1 = os.path.join( HOME_DIR, dir_1_short )
  dir_2 = os.path.join( HOME_DIR, dir_2_short )
  
  performance_1 = pd.read_csv( dir_1 + "/performances.csv", index_col = "measure" ).T
  performance_2 = pd.read_csv( dir_2 + "/performances.csv", index_col = "measure" ).T

  common_genes = np.intersect1d( performance_1.index.values, performance_2.index.values )
  print   performance_1
  print   performance_2
  print common_genes
  
  for gene in ["TP53"]: #common_genes:
    gene_dir1 = os.path.join( dir_1, gene )
    gene_dir2 = os.path.join( dir_2, gene )
    
    perf1 =  pd.read_csv( gene_dir1 + "/tissue_performance.csv", index_col = "tissue" )
    perf2 =  pd.read_csv( gene_dir2 + "/tissue_performance.csv", index_col = "tissue" )
    
    auc1 = perf1[ "AUROC"]
    auc2 = perf2[ "AUROC"]
    pr1 = perf1[ "AUPRC"]
    pr2 = perf2[ "AUPRC"]
    
    min_auc = max( min( auc1.min(), auc2.min() ) - 0.1*min( auc1.min(), auc2.min() ), 0 )
    min_prc = max( min( pr1.min(), pr2.min() ) - 0.1*min( pr1.min(), pr2.min() ), 0 )
    max_auc = min( max( auc1.max(), auc2.max() ) + 0.1*max( auc1.max(), auc2.max() ), 1 )
    max_prc = min( max( pr1.max(), pr2.max() ) + 0.1*max( pr1.max(), pr2.max() ), 1 )
    
    #print gene, min_auc,max_auc
    #pdb.set_trace()
    wildtypes = perf1[ "wildtype"];   mutations = perf1[ "mutations"]; 
    pan_wild = wildtypes.loc["PAN"]; pan_muts = mutations.loc["PAN"]
    n_pan = pan_wild + pan_muts
    
    f = pp.figure(figsize=(12,6))
    ax_auc = f.add_subplot( 121 )
    ax_prc = f.add_subplot( 122 )
    
    ax_auc.plot( [0,1],[0,1], 'k--' )
    ax_prc.plot( [0,1],[0,1], 'k--' )
    
    for tissue in auc1.index.values[1:]:
      weight = max(5, 200*float(mutations.loc[tissue]) / float( pan_muts ) )
      #print tissue, weight
      ax_auc.plot( auc1.loc[tissue], auc2.loc[tissue], 'o', mec='k', mew=1, ms = weight, alpha=0.75, label = tissue )
      ax_prc.plot( pr1.loc[tissue], pr2.loc[tissue], 'o', mec='k', mew=1, ms = weight, alpha=0.75, label = tissue  )
      
      if float(mutations.loc[tissue])  > float(0.05*pan_muts):
        ax_auc.text( auc1.loc[tissue], auc2.loc[tissue], tissue.upper(), fontsize=6 )
        ax_prc.text( pr1.loc[tissue], pr2.loc[tissue], tissue.upper(), fontsize=6 )
    ax_auc.set_xlabel(  name1 ); ax_auc.set_ylabel(  name2 )
    ax_auc.set_title("AUROC")
    #ax_prc.legend(loc='right')
    
    ax_auc.set_xlim( min_auc, max_auc )
    ax_prc.set_xlim( min_prc, max_prc )
    ax_auc.set_ylim( min_auc, max_auc )
    ax_prc.set_ylim( min_prc, max_prc )
    ax_prc.set_xlabel(  name1 ); ax_auc.set_ylabel(  name2 )
    ax_prc.set_title("AUPRC")
    
    pp.suptitle(gene)
    
    f.savefig( gene_dir1 + "/comparison.png", fmt='png', dpi=300)
    f.savefig( gene_dir2 + "/comparison.png", fmt='png', dpi=300)
    print perf1, perf2
    
  pp.show()