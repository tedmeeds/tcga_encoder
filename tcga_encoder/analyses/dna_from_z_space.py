from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
#from tcga_encoder.algorithms import *
import seaborn as sns
from sklearn.manifold import TSNE, locally_linear_embedding
from scipy import stats, special

def auc_standard_error( theta, nA, nN ):
  # from: Hanley and McNeil (1982), The Meaning and Use of the Area under the ROC Curve
  # theta: estimated AUC, can be 0.5 for a random test
  # nA size of population A
  # nN size of population N
  
  Q1=theta/(2.0-theta); Q2=2*theta*theta/(1+theta)
  
  SE = np.sqrt( (theta*(1-theta)+(nA-1)*(Q1-theta*theta) + (nN-1)*(Q2-theta*theta) )/(nA*nN) )
  
  return SE
  
def main( data_location, results_location, alpha=0.02 ):
  data_path    = os.path.join( HOME_DIR ,data_location ) #, "data.h5" )
  results_path = os.path.join( HOME_DIR, results_location )
  
  data_filename = os.path.join( data_path, "data.h5")
  fill_filename = os.path.join( results_path, "full_vae_fill.h5" )
  
  dna_dir = os.path.join( results_path, "dna_prediction" )
  check_and_mkdir(dna_dir)
  
  aucs_dir = os.path.join( dna_dir, "aucs_viz" )
  check_and_mkdir(aucs_dir)
  print "HOME_DIR: ", HOME_DIR
  print "data_filename: ", data_filename
  print "fill_filename: ", fill_filename
  
  #dna_genes = ["TP53","APC","PIK3CA"]
  dna_genes = ["APC","TP53","KRAS","PIK3CA","FBXW7","SMAD4","NRAS","ARID1A","ATM","CTNNB1"]
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
  
  
  dna_observed_bcs = data_store["/CLINICAL/observed"][ data_store["/CLINICAL/observed"]["DNA"]==1 ].index.values
  
  barcodes = np.intersect1d( dna_observed_bcs, barcodes )
  
  
  Z=Z.loc[barcodes]
  Z_values = Z.values
  #pdb.set_trace()
  
  dna = data_store["/DNA/channel/0"].loc[barcodes][dna_genes]
  
  
  
  
  tissues = data_store["/CLINICAL/TISSUE"].loc[barcodes]
  
  tissue_names = tissues.columns
  tissue_idx = np.argmax( tissues.values, 1 )

  n = len(Z)
  n_tissues = len(tissue_names)
  n_trials = 10
  trial_names = ["r_%d"%(trial_idx) for trial_idx in range(n_trials)]
  
  aucs_true  = {} #np.ones( (n_tissues,n_z), dtype=float)
  aucs_true_not  = {} 
  aucs_random  = {} #np.ones( (n_tissues,n_trials), dtype=float)
  se_auc_true = {}
  se_auc_random=  {}
  se_auc_true_not={}
  true_y = np.ones(n, dtype=int)
  
  auc_pvalues = {}
  for dna_gene in dna_genes:
    aucs_true[dna_gene] = 0.5*np.ones((n_tissues,n_z))
    aucs_true_not[dna_gene] = 0.5*np.ones((n_tissues,n_z))
    aucs_random[dna_gene] = 0.5*np.ones((n_tissues,n_z,n_trials))
    se_auc_true[dna_gene] = np.ones((n_tissues,n_z))
    se_auc_true_not[dna_gene] = np.ones((n_tissues,n_z))
    se_auc_random[dna_gene] = np.ones(n_tissues)
    
    auc_pvalues[dna_gene] = np.ones((n_tissues,n_z))
    
  for t_idx in range(n_tissues):
    tissue_name = tissue_names[t_idx]
    print "working %s"%(tissue_name)
    
    t_ids_cohort = tissue_idx == t_idx
    
    bcs_cohort = barcodes[pp.find(t_ids_cohort)]
    
    z_cohort   = Z.loc[bcs_cohort]
    dna_cohort = dna.loc[bcs_cohort]
    
    
    for dna_gene in dna_genes:
      
      dna_values = dna_cohort[dna_gene]
      n_1 = dna_values.sum()
      n_0 = len(dna_values)-n_1
      
      true_y = dna_values.values
      
      if n_1 > 5:
        print "     %s"%(dna_gene)

        for z_idx in range(n_z):
          z_values = z_cohort.values[:,z_idx]
          #pdb.set_trace()
          if true_y.sum() == 0 or true_y.sum() == len(true_y):
            pdb.set_trace()
          aucs_true[dna_gene][t_idx,z_idx] = roc_auc_score( true_y, z_values )
          aucs_true_not[dna_gene][t_idx,z_idx] = roc_auc_score( 1-true_y, z_values )
          se_auc_true[dna_gene][t_idx,z_idx] = auc_standard_error( aucs_true[dna_gene][t_idx,z_idx] , n_1, n_0 )
          se_auc_true_not[dna_gene][t_idx,z_idx] = auc_standard_error( aucs_true_not[dna_gene][t_idx,z_idx] , n_0, n_1 )
        
          # for trial_idx in range(n_trials):
          #   I = np.random.permutation(n_1+n_0)
          #   z = z_values[I]
          #   aucs_random[dna_gene][t_idx,z_idx,trial_idx] = roc_auc_score( true_y, z )
          
        se_auc_random[dna_gene][t_idx] = auc_standard_error( 0.5, n_1, n_0 )
  
  print "summarizing..."
  sorted_pan = []
  sorted_pan_sig=[]
  auc_sigs = {}
  for dna_gene in dna_genes:   
    aucs =  aucs_true[dna_gene]
    ses  =  se_auc_true[dna_gene]
    ses_not  =  se_auc_true_not[dna_gene]
    #aucs_r = aucs_random[dna_gene]
    se_r = se_auc_random[dna_gene]
    
    
    for t_idx in range(n_tissues):
      f = pp.figure()
      tissue_name = tissue_names[t_idx]
      print "working %s"%(tissue_name)
      
      aucs_tissue = aucs[t_idx]
      ses_tissue = ses[t_idx]
      ses_not_tissue = ses_not[t_idx]
      #aucs_r_tissue = aucs_r[t_idx]
      se_r_tissue = se_r[t_idx]
      
      se_combined = np.sqrt( ses_tissue**2 + se_r_tissue**2 )
      se_combined_not = np.sqrt( ses_not_tissue**2 + se_r_tissue**2 )
      
      difference = aucs_tissue - 0.5
      z_values = difference / se_combined 
      z_values_not = difference / se_combined_not 
      sign_difference = np.sign(difference)
      
      
      p_values = np.maximum( 1.0 - stats.norm.cdf( np.abs(z_values) ), 1.0 - stats.norm.cdf( np.abs(z_values_not) ) ) 
      
      ranked_zs = np.argsort(p_values)
      
      ax = f.add_subplot(111)
      best_aucs = np.maximum( aucs_tissue[ranked_zs], 1-aucs_tissue[ranked_zs])
      best_ses = ses_tissue[ ranked_zs ]
      
      #ax.plot( aucs_r_tissue.mean(1)[ranked_zs], 'r-', label="Random")
      
      ax.fill_between( np.arange(n_z), 0.5*np.ones(n_z)-2*se_r_tissue, 0.5*np.ones(n_z)+2*se_r_tissue, color="red", alpha=0.5)
      
      ax.fill_between( np.arange(n_z), best_aucs-2*best_ses, best_aucs+2*best_ses, color="blue", alpha=0.5)
      ax.plot( best_aucs, 'bo-', label="True" )
      
      x_tick_names = []
      for z_idx in ranked_zs:
        if sign_difference[z_idx]<0:
          x_tick_names.append( "-z_%d"%z_idx)
        else:
          x_tick_names.append( "z_%d"%z_idx)
      
      ax.set_xticks(np.arange(n_z))
      ax.set_xticklabels(x_tick_names, rotation=90, fontsize=6)
      
      auc_pvalues[dna_gene][t_idx,:] = p_values
      #ax.set_p  
      #ax.plot( aucs_r_tissue[ranked_zs,:], 'r.', label="Random")
      pp.ylim(0.5,1)
      pp.title( "Predicting %s on cohort %s"%(dna_gene,tissue_name ) )#,n_1,n_0,n_1+n_0) )
      pp.xlabel( "Ranked z")
      pp.ylabel( "AUC")
      if np.any(p_values<alpha):
        pp.savefig( aucs_dir + "/auc_tests_%s_%s"%(dna_gene,tissue_name))
      pp.close('all')
      #pp.show()
    auc_pvalues[dna_gene] = pd.DataFrame( auc_pvalues[dna_gene], index=tissue_names, columns = z_names )
    auc_sigs[dna_gene] = pd.DataFrame( (auc_pvalues[dna_gene].values<alpha).astype(int), index=tissue_names, columns = z_names )
    auc_sigs[dna_gene].to_csv( dna_dir + "/pan_sig_z_for_dna_%s.csv"%(dna_gene) )
    reduced_ = auc_sigs[dna_gene]
    rows = reduced_.sum(1)[ reduced_.sum(1)>0 ].index.values
    cols = reduced_.sum(0)[ reduced_.sum(0)>0 ].index.values
    reduced_ = reduced_.loc[rows]
    reduced_ = reduced_[cols]
    size_per_unit = 0.25
    size1 = max( int( len(rows)*size_per_unit ), 12 )
    size2 = max( int( len(cols)*size_per_unit ), 12 )
    f = sns.clustermap( reduced_, square=False, figsize=(size1,size2) )
    
    pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
    pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
    
    pp.savefig( dna_dir + "/z_for_dna_clustermap_sig_%s.png"%(dna_gene), format="png", dpi=300)
    
    
    sorted_pan_sig.append(auc_sigs[dna_gene].sum(0))
    sorted_pan.append( np.log( auc_pvalues[dna_gene] + 1e-12 ).sum(0) ) #.sort_values()
  sorted_pan = pd.concat(sorted_pan,axis=1)
  sorted_pan.columns = dna_genes
  sorted_pan.to_csv( dna_dir + "/pan_logpvals_z_for_dna.csv" )
  
  sorted_pan_sig = pd.concat(sorted_pan_sig,axis=1)
  sorted_pan_sig.columns = dna_genes
  sorted_pan_sig.to_csv( dna_dir + "/pan_sig_z_for_dna.csv" )
  size1 = max( int( n_z*size_per_unit ), 12 )
  size2 = max( int( len(dna_genes)*size_per_unit ), 12 )
  f = sns.clustermap( sorted_pan.T, figsize=(size1,size2) )
  
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  pp.savefig( dna_dir + "/z_for_dna_clustermap_logpval.png", format="png")
  
  f = sns.clustermap( sorted_pan_sig.T, figsize=(size1,size2) )
  
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  pp.savefig( dna_dir + "/z_for_dna_clustermap_sig.png", format="png")
  #pdb.set_trace()
        
  #   #events = Z["E"].loc[]
  #
  # #
  # aucs_true  = pd.DataFrame( aucs_true, index = tissue_names, columns=z_names )
  # aucs_random = pd.DataFrame( aucs_random, index = tissue_names, columns=trial_names )
  #
  # aucs_true.to_csv( tissue_dir + "/aucs_true.csv" )
  # aucs_random.to_csv( tissue_dir + "/aucs_random.csv" )
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