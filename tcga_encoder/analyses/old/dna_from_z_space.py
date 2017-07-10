from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
#from tcga_encoder.algorithms import *
import seaborn as sns
from sklearn.manifold import TSNE, locally_linear_embedding
from scipy import stats, special

def view_results( results_location, store, gene, n_permutations, source, method, disease_string, title_str = "", max_nbr = 100, zoom = True ):
  mean_aucs = store["/%s/%s/%s/%s/labels_0/xval_aucs"%(disease_string,gene,source, method)]
  mean_auc = store["/%s/%s/%s/%s/labels_0/xval_aucs"%(disease_string,gene,source, method)].mean()
  var_auc  = store["/%s/%s/%s/%s/labels_0/xval_aucs"%(disease_string,gene,source, method)].var()
  
  barcodes = store["/%s/%s/%s/%s/labels_0/xval_predictions"%(disease_string,gene,source, method)].index.values
  diseases = np.array( [s.split("_")[0] for s in barcodes])
  u_diseases = np.unique( diseases )
  
  disease_aucs = store[ "/%s/%s/%s/%s/labels_0/xval_disease_aucs"%(disease_string,gene,source, method)]
  mean_disease_aucs = disease_aucs.mean(1)
  var_disease_aucs = disease_aucs.var(1)
  
  std_auc  = np.sqrt( var_auc )
  ordered_mean_aucs = store["/%s/%s/%s/%s/labels_0/xval_aucs_elementwise"%(disease_string,gene,source, method)].mean(1).sort_values(ascending=False)
  ordered_source_genes = ordered_mean_aucs.index.values
  ordered_var_aucs = store["/%s/%s/%s/%s/labels_0/xval_aucs_elementwise"%(disease_string,gene,source, method)].loc[ordered_source_genes].var(1)
  order_std_aucs = np.sqrt(ordered_var_aucs)
  D = len(ordered_mean_aucs.values)
  
  element_aucs = store["/%s/%s/%s/%s/labels_0/xval_aucs_elementwise"%(disease_string,gene,source, method)]
  element_aucs=element_aucs.T
  element_aucs["ALL"] = mean_aucs
  element_aucs=element_aucs.T
  orientation = "horizontal"
  if zoom is True:
    marker = 'o'
  else:
    marker = '.'
    
  nD = np.minimum( D, max_nbr )
  
  if orientation == "vertical":
    f1=pp.figure( figsize=(6,16))
  else:
    f1=pp.figure( figsize=(16,6))
    
  ax11 = f1.add_subplot(111)
  #
  # disease_aucs = []
  # for disease in u_diseases:
  #   disease_aucs.append( store[ "/%s/%s/%s/%s/labels_0/diseases/%s/xval_aucs_elementwise"%(disease_string,gene,source, method, disease)].mean(1) )
  #
  # disease_aucs = pd.concat(disease_aucs,axis=1)
  #pdb.set_trace() 
  if orientation == "vertical":
    # for d_idx in range(len(u_diseases)):
   #    disease = u_diseases[d_idx]
   #    results_store[ "/%s/%s/%s/labels_%d/diseases/%s/xval_aucs_elementwise"%(dna_gene,source, method,label_permutation_idx,disease)] = pd.DataFrame( elementwise_aucs_by_disease[d_idx,:,:], index = source_data.columns, columns=xval_columns )
   #    results_store[ "/%s/%s/%s/labels_%d/diseases/%s/xval_aucs"%(dna_gene,source, method,label_permutation_idx,disease)] = pd.DataFrame( np.array(aucs_by_disease).T, index = u_diseases, columns=xval_columns )
   #  
    ax11.vlines( mean_disease_aucs.values, 0, nD-1, color='g' )
    
    for disease in u_diseases:
      aucs =store[ "/%s/%s/%s/%s/labels_0/diseases/%s/xval_aucs_elementwise"%(disease_string,gene,source, method, disease)].mean(1)
      ax11.plot( aucs.loc[ordered_source_genes].values[:nD], nD-np.arange(nD)-1, '.-', mec = 'k', label = "%s"%(disease) )
      
    #pdb.set_trace()
    ax11.plot( ordered_mean_aucs.values[:nD], nD-np.arange(nD)-1, 'b'+marker+"-", mec = 'k', label = "True" )
    
    ax11.fill_betweenx( nD-np.arange(nD), \
                        ordered_mean_aucs.values[:nD] + 2*order_std_aucs.values[:nD], \
                        ordered_mean_aucs.values[:nD] - 2*order_std_aucs.values[:nD], facecolor='blue', edgecolor = 'k', alpha=0.5 )
    ax11.plot( ordered_mean_aucs.values[:nD], nD-np.arange(nD)-1, 'b'+marker+"-", mec = 'k', label = "True" )

    ax11.fill_betweenx( nD-np.arange(nD), \
                        mean_auc*np.ones(nD) -2*std_auc, \
                        mean_auc*np.ones(nD) +2*std_auc, facecolor='blue',edgecolor='k', alpha=0.5 )

    ax11.vlines( mean_auc, 0, nD-1, color='b' )
    if zoom is True:
      ax11.set_yticks( nD-1-np.arange(nD) )
      ax11.set_yticklabels( ordered_source_genes[:nD], rotation='horizontal', fontsize=8 )
    
  else:
    #ax11.fill_between( 2+np.arange(nD), \
    #                    ordered_mean_aucs.values[:nD] + 2*order_std_aucs.values[:nD], \
    #                    ordered_mean_aucs.values[:nD] - 2*order_std_aucs.values[:nD], facecolor='blue', edgecolor = 'k', alpha=0.5 )
    #ax11.plot( np.arange(nD)+2, ordered_mean_aucs.values[:nD], 'b'+marker+"-", mec = 'k', label = "True" )
    ax11.plot( np.arange(nD)+2, ordered_mean_aucs.values[:nD], 'b-', mec = 'k', label = "True" )

    # ax11.fill_between( 1+np.arange(nD), \
    #                     mean_auc*np.ones(nD) -2*std_auc, \
    #                     mean_auc*np.ones(nD) +2*std_auc, facecolor='blue',edgecolor='k', alpha=0.5 )
    #
    # ax11.hlines( mean_auc, 1, nD, color='b' )
    if zoom is True:
      ax11.set_xticks( 2+np.arange(nD) )
      ax11.set_xticklabels( ordered_source_genes[:nD], rotation='vertical', fontsize=8 )
  
  #
  #pdb.set_trace()
  #ax11.plot( np.ones( len(mean_aucs.values)), mean_aucs.values, 'o', ms=10, color='orange', mec='k', alpha=0.75) 
  #ax11.plot( [1], [mean_auc], 'd', color='orchid',mec='orchid' ,ms=30, mew=2, lw=2, alpha=0.75 )
  permutations = []
  combined_permutations = []
  for permutation_idx in range(n_permutations):
    mean_auc_p = store["/%s/%s/%s/%s/labels_%d/xval_aucs"%(disease_string,gene,source, method,permutation_idx+1)].mean()
    combined_permutations.append( mean_auc_p)
  combined_permutations = pd.Series( np.array(combined_permutations), index = np.arange(n_permutations) )
  
  #permutations.append(combined_permutations )
  for permutation_idx in range(n_permutations):
    mean_auc_p = store["/%s/%s/%s/%s/labels_%d/xval_aucs"%(disease_string,gene,source, method,permutation_idx+1)].mean()
    var_auc_p  = store["/%s/%s/%s/%s/labels_%d/xval_aucs"%(disease_string,gene,source, method, permutation_idx+1)].var()
    std_auc_p  = np.sqrt( var_auc_p )
    
    
    mean_aucs = store["/%s/%s/%s/%s/labels_%d/xval_aucs_elementwise"%(disease_string,gene,source, method,permutation_idx+1)].loc[ordered_source_genes].mean(1)
    
    #permutations.append( store["/%s/%s/%s/%s/labels_%d/xval_aucs_elementwise"%(disease_string,gene,source, method,permutation_idx+1)].loc[ordered_source_genes] )
    permutations.append( mean_aucs )
    
    # if orientation == "vertical":
    #   ax11.vlines( mean_auc_p, 0, nD-1, color='r' )
    #   #ax11.plot( mean_aucs[:nD], nD-1-np.arange(nD),  'o', color='orange', mec='k', alpha=0.5)
    # else:
    #   ax11.hlines( mean_auc_p, 0, nD-1, color='r' )
    #   #ax11.plot( nD-1-np.arange(nD), mean_aucs[:nD], 'o', color='orange', mec='k', alpha=0.5)
  #
  
  permutations = pd.concat( permutations,axis=1 )
  permutations = permutations.T
  permutations["ALL"] = combined_permutations
  new_order = ["ALL"]
  new_order.extend(ordered_source_genes[:nD] )
  permutations = permutations.T.loc[new_order]
  element_aucs=element_aucs.loc[new_order]
  print permutations
  #pdb.set_trace()
  correct_labels = store["/%s/%s/%s/%s/labels_%d/xval_aucs_elementwise"%(disease_string,gene,source, method,0)].loc[ordered_source_genes]
  if orientation == "vertical":
    color = dict(boxes='DarkRed', whiskers='DarkOrange',  medians='Red', caps='Black')
    color2 = dict(boxes='DarkBlue', whiskers='DarkBlue',  medians='DarkBlue', caps='Cyan')
    permutations.T.boxplot(ax=ax11,color=color)
    element_aucs.T.boxplot(ax=ax11,color=color2)
  else:
    color = dict(boxes='LightCoral', whiskers='DarkRed',  medians='DarkRed', caps='LightCoral')
    color2 = dict(boxes='SkyBlue', whiskers='DarkBlue',  medians='DarkBlue', caps='SkyBlue')
    permutations.T.plot.box(ax=ax11,color=color,patch_artist=True)
    element_aucs.T.plot.box(ax=ax11,color=color2,patch_artist=True, widths=0.25)
    if zoom is True:
      ax11.set_xticks( 1+np.arange(len(new_order)) )
      ax11.set_xticklabels( new_order, rotation='vertical', fontsize=8 )
    
  #pdb.set_trace()
  t_tests = []
  for this_gene in ordered_source_genes[:nD]:
    k = n_permutations
    
    p_value = ( np.sum( correct_labels.loc[this_gene].values.mean() < permutations.loc[this_gene].values ) + 1.0 )/ (k+1.0)
    #t_tests.append( [gene,stats.ttest_ind( permutations.loc[gene].values, correct_labels.loc[gene], equal_var=False )] )
    t_tests.append(p_value)
  #pdb.set_trace()
  pp.grid('on')
  pp.title( "%s using %s of %s with %s mean AUC = %0.3f"%(gene,disease_string, source, method, mean_auc))
  pp.subplots_adjust(bottom=0.2)
  figname1 = os.path.join( HOME_DIR, os.path.dirname(results_location) ) + "/aucs_%s_%s_%s_%s_%s.png"%(gene,source, method,disease_string,title_str)
  f1.savefig(  figname1, dpi=300 )


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