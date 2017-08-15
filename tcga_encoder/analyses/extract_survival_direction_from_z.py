from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.locations import *

if __name__ == "__main__":
  
  #names = ["no scale","scale","entropy"]
  
  # short_coefs_dirs = []
  # short_coefs_dirs.append("results/tcga_vae_post_recomb9/medium/xval_nn_tissue/z_100_h_500_anti_100/fold_1_of_5/everything2/survival_regression_global_Z_K_5_Cox2")
  # short_latent_dirs = []
  # short_latent_dirs.append("results/tcga_vae_post_recomb9/medium/xval_nn_tissue/z_100_h_500_anti_100/fold_1_of_5/everything2/A_spearmans_latent_tissue")
  # short_weighted_dirs = []
  # short_weighted_dirs.append("results/tcga_vae_post_recomb9/medium/xval_nn_tissue/z_100_h_500_anti_100/fold_1_of_5/everything2/A_weighted_latent_tissue")


  short_coefs_dirs = []
  short_coefs_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_100_h_1000_anti_5000/fold_1_of_50/everything2/survival_regression_global_Z_K_5_Cox2")
  short_latent_dirs = []
  short_latent_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_100_h_1000_anti_5000/fold_1_of_50/everything2/A_spearmans_latent_tissue")
  short_weighted_dirs = []
  short_weighted_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_100_h_1000_anti_5000/fold_1_of_50/everything2/A_weighted_latent_tissue")

  short_coefs_dirs = []
  short_coefs_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_500_h_5000_anti_5000/fold_1_of_50/everything2/survival_regression_global_Z_K_2_Cox2")
  short_latent_dirs = []
  short_latent_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_500_h_5000_anti_5000/fold_1_of_50/everything2/A_spearmans_latent_tissue")
  short_weighted_dirs = []
  short_weighted_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_500_h_5000_anti_5000/fold_1_of_50/everything2/A_weighted_latent_tissue")


  #short_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/22_z_100_h_1000_anti_0/fold_1_of_50/tissue_prediction/")
  #short_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_100_h_1000_anti_5000/fold_1_of_50/tissue_prediction/")

  coefs_dirs = []
  latent_dirs = []
  weighted_dirs=[]
  for short_coefs_dir, short_latent_dir, short_weighted_dir in zip( short_coefs_dirs, short_latent_dirs, short_weighted_dirs ):
    coefs_dirs.append( os.path.join( HOME_DIR, short_coefs_dir ) )
    latent_dirs.append( os.path.join( HOME_DIR, short_latent_dir ) )
    weighted_dirs.append( os.path.join( HOME_DIR, short_weighted_dir ) )
  
  random_auc = []
  aucs       = []
  pvalues    =[]
  for coefs_dir, latent_dir, weighted_dir in zip( coefs_dirs, latent_dirs, weighted_dirs ):
    coefs      = pd.read_csv( coefs_dir + "/coefs.csv", index_col = "feature" )
    
    rna_rho   = np.log(1e-12+pd.read_csv( latent_dir + "/rna_z_p.csv", index_col = "gene" ))
    dna_rho   = np.log(1e-12+pd.read_csv( latent_dir + "/dna_z_p.csv", index_col = "gene" ))
    mirna_rho = np.log(1e-12+pd.read_csv( latent_dir + "/mirna_z_p.csv", index_col = "gene" ))
    meth_rho  = np.log(1e-12+pd.read_csv( latent_dir + "/meth_z_p.csv", index_col = "gene" ))
    weighted  = pd.read_csv( weighted_dir + "/all_z_weighted.csv", index_col = "gene" )
    
    mean_coef = coefs["mean"] / np.sqrt(np.sum(np.square(coefs["mean"])))
    
    rna_projection   = pd.Series( np.dot( rna_rho, mean_coef ), index = rna_rho.index, name="RNA" ).sort_values()
    dna_projection   = pd.Series( np.dot( dna_rho, mean_coef ), index = dna_rho.index, name="DNA" ).sort_values()
    mirna_projection = pd.Series( np.dot( mirna_rho, mean_coef ), index = mirna_rho.index, name="miRNA" ).sort_values()
    meth_projection  = pd.Series( np.dot( meth_rho, mean_coef ), index = meth_rho.index, name="METH" ).sort_values()
    all_projection  = pd.Series( np.dot( weighted, mean_coef ), index = weighted.index, name="ALL" ).sort_values()
    
    rna_projection.to_csv( latent_dir + "/rna_z_rho_projection.csv" )
    dna_projection.to_csv( latent_dir + "/dna_z_rho_projection.csv" )
    mirna_projection.to_csv( latent_dir + "/mirna_z_rho_projection.csv" )
    meth_projection.to_csv( latent_dir + "/meth_z_rho_projection.csv" )
    all_projection.to_csv( weighted_dir + "/all_z_weighted_projection.csv" )
    
    print "RNA-----------"
    print rna_projection[:20]
    print rna_projection[-20:]
    print "DNA-----------"
    print dna_projection[:20]
    print dna_projection[-20:]
    print "miRNA-----------"
    print mirna_projection[:20]
    print mirna_projection[-20:]
    print "METH-----------"
    print meth_projection[:20]
    print meth_projection[-20:]
    print "ALL-----------"
    print all_projection[:20]
    print all_projection[-20:]
  #pdb.set_trace()
