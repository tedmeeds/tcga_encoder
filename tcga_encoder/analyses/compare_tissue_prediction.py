
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
  
  names = ["no scale","scale","entropy"]
  
  short_dirs = []
  short_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers_no_cohort/23_z_100_h_1000_anti_0/fold_1_of_50/tissue_prediction/")
  short_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/22_z_100_h_1000_anti_0/fold_1_of_50/tissue_prediction/")
  short_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_100_h_1000_anti_5000/fold_1_of_50/tissue_prediction/")

  dirs = []
  for short_dir in short_dirs:
    dirs.append( os.path.join( HOME_DIR, short_dir ) )
  
  random_auc = []
  aucs       = []
  pvalues    =[]
  for dir_1 in dirs:
    aucs_true      = pd.read_csv( dir_1 + "/aucs_true.csv", index_col = "tissue" )
    aucs_random    = pd.read_csv( dir_1 + "/aucs_random.csv", index_col = "tissue" )
    z_true_p_value = pd.read_csv( dir_1 + "/z_true_p_value.csv", index_col = "tissue" )
    
    aucs.append( aucs_true.values.flatten() )
    pvalues.append( z_true_p_value.values.flatten() )
    random_auc.extend( list(aucs_random.values.flatten()) )
    
  binses = [20,50,100,500]
  for bins in binses:
    pp.figure()
    pp.hist( random_auc, bins=np.linspace(0,1,bins+1), color="red", normed=True, histtype="step", lw=2, label="random" )
    for name,values in zip(names,aucs_true):
      pp.hist( values, bins=np.linspace(0,1,bins+1), normed=True, histtype="step", lw=2, label=name )
    
    #pp.plot( [0,1.0],[0.5,0.5], 'r-', lw=3)
    pp.legend()
    pp.xlabel("Area Under the ROC")
    pp.ylabel("Pr(AUC)")
    pp.title("Comparison of distributions Area under the ROC")
    for dir_1 in dirs:
      pp.savefig( dir_1 + "/model_auc_comparison_%dbins.png"%(bins), format='png', dpi=300 )

    pp.figure()
    pp.plot( [0,1.0],[1.0,1.0], 'r-', lw=2, label="random")
    for name,values in zip(names,pvalues):
      pp.hist( values, bins=np.linspace(0,1,bins+1), normed=True, histtype="step", lw=2, label=name )

    pp.legend()
    pp.xlabel("p-value")
    pp.ylabel("Pr(p-value)")
    pp.title("Comparison of distributions AUC p-values")
    for dir_1 in dirs:
      pp.savefig( dir_1 + "/model_auc_p_values_%dbins.png"%(bins), format='png', dpi=300 )


  pp.close('all')