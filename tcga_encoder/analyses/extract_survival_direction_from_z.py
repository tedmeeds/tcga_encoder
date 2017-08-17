from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.locations import *
from scipy import stats

from tcga_encoder.analyses.everything_functions import *
from tcga_encoder.analyses.everything_long import *
from tcga_encoder.analyses.survival_functions import *
if __name__ == "__main__":
  
  #names = ["no scale","scale","entropy"]
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  
  data_locations = [data_location]
  results_locations=[results_location]
  short_coefs_dirs = []
  short_coefs_dirs.append("%s/everything2/survival_regression_global_Z_K_2_Cox2"%(results_locations[-1]))
  short_latent_dirs = []
  short_latent_dirs.append("%s/everything2/A_spearmans_latent_tissue"%(results_locations[-1]))
  short_weighted_dirs = []
  short_weighted_dirs.append("%s/everything2/A_weighted_latent_tissue"%(results_locations[-1]))
  
  #data_locations = ["data/broad_processed_june_2017/20160128/pan_large"]
  #results_locations=["results/tcga_vae_post_recomb9/large/xval_rec_not_blind_fix_outliers/22_z_100_h_1000_anti_5000/fold_1_of_50"]
  short_coefs_dirs = []
  short_coefs_dirs.append("%s/everything2/survival_regression_global_Z_K_5_Cox2"%(results_locations[-1]))
  short_latent_dirs = []
  short_latent_dirs.append("%s/everything2/A_spearmans_latent_tissue"%(results_locations[-1]))
  short_weighted_dirs = []
  short_weighted_dirs.append("%s/everything2/A_weighted_latent_tissue"%(results_locations[-1]))


  #data_locations = ["data/broad_processed_june_2017/20160128/pan_large"]
  #results_locations=["results/tcga_vae_post_recomb9/large/xval_rec_not_blind_fix_outliers/22_z_200_h_2000_anti_5000/fold_1_of_50"]
  short_coefs_dirs = []
  short_coefs_dirs.append("%s/everything2/survival_regression_global_Z_K_5_Cox2"%(results_locations[-1]))
  short_latent_dirs = []
  short_latent_dirs.append("%s/everything2/A_spearmans_latent_tissue"%(results_locations[-1]))
  short_weighted_dirs = []
  short_weighted_dirs.append("%s/everything2/A_weighted_latent_tissue"%(results_locations[-1]))
  # short_coefs_dirs = []
  # short_coefs_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_100_h_1000_anti_5000/fold_1_of_50/everything2/survival_regression_global_Z_K_5_Cox2")
  # short_latent_dirs = []
  # short_latent_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_100_h_1000_anti_5000/fold_1_of_50/everything2/A_spearmans_latent_tissue")
  # short_weighted_dirs = []
  # short_weighted_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_100_h_1000_anti_5000/fold_1_of_50/everything2/A_weighted_latent_tissue")

  # short_coefs_dirs = []
  # short_coefs_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_500_h_5000_anti_5000/fold_1_of_50/everything2/survival_regression_global_Z_K_2_Cox2")
  # short_latent_dirs = []
  # short_latent_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_500_h_5000_anti_5000/fold_1_of_50/everything2/A_spearmans_latent_tissue")
  # short_weighted_dirs = []
  # short_weighted_dirs.append("results/tcga_vae_post_recomb9/xlarge/xval_rec_not_blind_fix_outliers/20_z_500_h_5000_anti_5000/fold_1_of_50/everything2/A_weighted_latent_tissue")


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
  for coefs_dir, latent_dir, weighted_dir,data_location, results_location in zip( coefs_dirs, latent_dirs, weighted_dirs,data_locations, results_locations ):
    coefs      = pd.read_csv( coefs_dir + "/coefs.csv", index_col = "feature" )
    
    data = load_data_and_fill( data_location, results_location )
    dna = data.dna
    scores      = pd.read_csv( coefs_dir + "/global_survival.csv", index_col = "barcode" )
  
    RNA_scale   = data.RNA_scale #tanh(data.RNA_scale) 
    miRNA_scale = data.miRNA_scale #tanh(data.miRNA_scale) 
    METH_scale  = data.METH_scale #tanh(data.METH_scale )
    
    RNA_scale   = tanh(data.RNA_scale) 
    miRNA_scale = tanh(data.miRNA_scale) 
    METH_scale  = tanh(data.METH_scale )
    
    
    rho_rna_score = stats.spearmanr( RNA_scale.loc[scores.index].values, scores["weighted_death"].values[:,np.newaxis] )
    rna_score_rho = pd.Series( rho_rna_score[0][-1,:-1], index=RNA_scale.columns, name="RNA_score_rho")
    rna_score_p   = pd.Series( rho_rna_score[1][-1,:-1], index=RNA_scale.columns, name="RNA_score_p")
    
    rho_mirna_score = stats.spearmanr( miRNA_scale.loc[scores.index].values, scores["weighted_death"].values[:,np.newaxis] )
    mirna_score_rho = pd.Series( rho_mirna_score[0][-1,:-1], index=miRNA_scale.columns, name="miRNA_score_rho")
    mirna_score_p   = pd.Series( rho_mirna_score[1][-1,:-1], index=miRNA_scale.columns, name="miRNA_score_p")
    
    rho_meth_score = stats.spearmanr( METH_scale.loc[scores.index].values, scores["weighted_death"].values[:,np.newaxis] )
    meth_score_rho = pd.Series( rho_meth_score[0][-1,:-1], index=METH_scale.columns, name="METH_score_rho")
    meth_score_p   = pd.Series( rho_meth_score[1][-1,:-1], index=METH_scale.columns, name="METH_score_p")
    
    rho_dna_score = stats.spearmanr(2*dna.loc[scores.index].values-1, scores["weighted_death"].values[:,np.newaxis] )
    dna_score_rho = pd.Series( rho_dna_score[0][-1,:-1], index=dna.columns, name="DNA_score_rho")
    dna_score_p   = pd.Series( rho_dna_score[1][-1,:-1], index=dna.columns, name="DNA_score_p")
    
    dna_score_rho.to_csv( latent_dir + "/dna_score_rho.csv" )
    dna_score_p.to_csv( latent_dir + "/dna_score_p.csv" )
    
    rna_score_rho.to_csv( latent_dir + "/rna_score_rho.csv" )
    rna_score_p.to_csv( latent_dir + "/rna_score_p.csv" )
    
    mirna_score_rho.to_csv( latent_dir + "/mirna_score_rho.csv" )
    mirna_score_p.to_csv( latent_dir + "/mirna_score_p.csv" )
    
    meth_score_rho.to_csv( latent_dir + "/meth_score_rho.csv" )
    meth_score_p.to_csv( latent_dir + "/meth_score_p.csv" )
    
    
    #pdb.set_trace()
    
    rna_rho   = pd.read_csv( latent_dir + "/rna_z_rho.csv", index_col = "gene" )
    dna_rho   = pd.read_csv( latent_dir + "/dna_z_rho.csv", index_col = "gene" )
    mirna_rho = pd.read_csv( latent_dir + "/mirna_z_rho.csv", index_col = "gene" )
    meth_rho  = pd.read_csv( latent_dir + "/meth_z_rho.csv", index_col = "gene" )
    
    rna_c   = pd.read_csv( latent_dir + "/consensus_rna_scores.csv", index_col = "gene" )
    dna_c   = pd.read_csv( latent_dir + "/consensus_dna_scores.csv", index_col = "gene" )
    mirna_c = pd.read_csv( latent_dir + "/consensus_mirna_scores.csv", index_col = "gene" )
    meth_c  = pd.read_csv( latent_dir + "/consensus_meth_scores.csv", index_col = "gene" )
    
    # rna_rho   = 1-pd.read_csv( latent_dir + "/rna_z_p.csv", index_col = "gene" )
    # dna_rho   = 1-pd.read_csv( latent_dir + "/dna_z_p.csv", index_col = "gene" )
    # mirna_rho = 1-pd.read_csv( latent_dir + "/mirna_z_p.csv", index_col = "gene" )
    # meth_rho  = 1-pd.read_csv( latent_dir + "/meth_z_p.csv", index_col = "gene" )
    
    # Stouffer's Z
    # rna_z   = stats.norm.ppf(pd.read_csv( latent_dir + "/rna_z_p.csv", index_col = "gene" ))
    # dna_z   = stats.norm.ppf(pd.read_csv( latent_dir + "/dna_z_p.csv", index_col = "gene" ))
    # mirna_z = stats.norm.ppf(pd.read_csv( latent_dir + "/mirna_z_p.csv", index_col = "gene" ))
    # meth_z  = stats.norm.ppf(pd.read_csv( latent_dir + "/meth_z_p.csv", index_col = "gene" ))
    
    #weighted  = pd.read_csv( weighted_dir + "/all_z_weighted.csv", index_col = "gene" )
    
    mean_coef = coefs["mean"] # / np.sqrt(np.sum(np.square(coefs["mean"])))
    
    rna_projection   = pd.Series( np.dot( rna_rho, mean_coef ), index = rna_rho.index, name="RNA" ).sort_values()
    dna_projection   = pd.Series( np.dot( dna_rho, mean_coef ), index = dna_rho.index, name="DNA" ).sort_values()
    mirna_projection = pd.Series( np.dot( mirna_rho, mean_coef ), index = mirna_rho.index, name="miRNA" ).sort_values()
    meth_projection  = pd.Series( np.dot( meth_rho, mean_coef ), index = meth_rho.index, name="METH" ).sort_values()
    #all_projection  = pd.Series( np.dot( weighted, mean_coef ), index = weighted.index, name="ALL" ).sort_values()
    
    
    rna_projection_c   = pd.Series( np.dot( rna_c, mean_coef ), index = rna_c.index, name="RNA" ).sort_values()
    dna_projection_c   = pd.Series( np.dot( dna_c, mean_coef ), index = dna_c.index, name="DNA" ).sort_values()
    mirna_projection_c = pd.Series( np.dot( mirna_c, mean_coef ), index = mirna_c.index, name="miRNA" ).sort_values()
    meth_projection_c  = pd.Series( np.dot( meth_c, mean_coef ), index = meth_c.index, name="METH" ).sort_values()
    
    rna_projection.to_csv( latent_dir + "/rna_z_rho_projection.csv" )
    dna_projection.to_csv( latent_dir + "/dna_z_rho_projection.csv" )
    mirna_projection.to_csv( latent_dir + "/mirna_z_rho_projection.csv" )
    meth_projection.to_csv( latent_dir + "/meth_z_rho_projection.csv" )
    #all_projection.to_csv( weighted_dir + "/all_z_weighted_projection.csv" )
    
    
    rna_projection_c.to_csv( latent_dir + "/rna_z_concensus_projection.csv" )
    dna_projection_c.to_csv( latent_dir + "/dna_z_concensus_projection.csv" )
    mirna_projection_c.to_csv( latent_dir + "/mirna_z_concensus_projection.csv" )
    meth_projection_c.to_csv( latent_dir + "/meth_z_concensus_projection.csv" )
    
    print "RNA-----------"
    print rna_score_p.sort_values()[:20]
    #print rna_projection_c[-20:]
    #print "DNA-----------"
    #print dna_projection_c[:20]
    #print dna_projection_c[-20:]
    print "miRNA-----------"
    print mirna_score_p.sort_values()[:20]
    print "METH-----------"
    print meth_score_p.sort_values()[:20]
    #print meth_projection_c[-20:]
    # print "ALL-----------"
    # print all_projection[:20]
    # print all_projection[-20:]
    
    sorted_scores = scores["weighted_death"].sort_values()
    rna_sorted   = RNA_scale.loc[sorted_scores.index][ rna_score_p.sort_values()[:200].index.values ]
    mirna_sorted = miRNA_scale.loc[sorted_scores.index][ mirna_score_p.sort_values()[:200].index.values ]
    meth_sorted  = METH_scale.loc[sorted_scores.index][ meth_score_p.sort_values()[:200].index.values ]
    
    rna_sorted.to_csv( latent_dir + "/rna_sorted_by_hazard.csv", index_label='barcode' )
    mirna_sorted.to_csv( latent_dir + "/mirna_sorted_by_hazard.csv", index_label='barcode' )
    meth_sorted.to_csv( latent_dir + "/meth_sorted_by_hazard.csv", index_label='barcode' )
    #pdb.set_trace()
    # f=pp.figure()
    # X_sorted = pd.DataFrame( XV[patient_order,:][:,z_ids2use], index = X.index.values[patient_order], columns=z_names2use )
    # X_sorted2 = pd.DataFrame( XV2[patient_order,:][:,z_ids2use2], index = X_sorted.index.values, columns=XV2cols )
    # h = sns.clustermap( X_sorted2, \
    #                     row_colors=k_colors_global, \
    #                     row_cluster=False, \
    #                     col_cluster=False, \
    #                     figsize=(10,10),\
    #                     yticklabels = False )
    #
    # pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    # pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    # pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
    # pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
    # h.ax_row_dendrogram.set_visible(False)
    # h.ax_col_dendrogram.set_visible(False)
    # h.cax.set_visible(False)
    # pp.savefig( heatmap_fig_dir + "/PAN.png", format="png" )#, dpi=300, bbox_inches='tight')
    # pp.close('all')
    
    
  #pdb.set_trace()
