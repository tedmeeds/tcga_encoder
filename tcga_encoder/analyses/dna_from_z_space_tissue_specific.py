from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
from tcga_encoder.analyses.dna_functions import *

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



  
def main( data_location, results_location, alpha=0.02 ):
  data_path    = os.path.join( HOME_DIR ,data_location ) #, "data.h5" )
  results_path = os.path.join( HOME_DIR, results_location )
  
  data_filename = os.path.join( data_path, "data.h5")
  fill_filename = os.path.join( results_path, "full_vae_fill.h5" )
  
  dna_dir = os.path.join( results_path, "dna_prediction_by_tissue" )
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
  
  dna = data_store["/DNA/channel/0"].loc[barcodes] #[dna_genes]
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
  
  aucs_true       = OrderedDict()
  aucs_true_not   = OrderedDict() 
  se_auc_true     = OrderedDict() 
  se_auc_true_not = OrderedDict() 
  se_auc_random   = OrderedDict() 
  
  all_auc_series = []
  all_p_series = []
  for t_idx in range(n_tissues):
    tissue_name = tissue_names[t_idx]
    if tissue_name == "gbm" or tissue_name == "meso":
      print "skipping ",tissue_name
      continue
    print "working %s"%(tissue_name)
    
    t_ids_cohort = tissue_idx == t_idx
    
    bcs_cohort = barcodes[pp.find(t_ids_cohort)]
    
    z_cohort   = Z.loc[bcs_cohort]
    dna_cohort = dna.loc[bcs_cohort]
    dna_genes = dna_cohort.sum(0).sort_values(ascending=False)[:100].index.values
    #pdb.set_trace()
    for dna_gene in dna_genes:
      
      dna_values = dna_cohort[dna_gene]
      n_1 = dna_values.sum()
      n_0 = len(dna_values)-n_1
      
      if n_1 == 0:
        continue
      true_y = dna_values.values
      
      print "     %s"%(dna_gene)
    
      aucs        = np.zeros( n_z, dtype=float )
      p_values        = np.zeros( n_z, dtype=float )
      #aucs_not    = np.zeros( n_z, dtype=float )
      se_aucs     = np.zeros( n_z, dtype=float )
      #se_aucs_not = np.zeros( n_z, dtype=float )
      se_random   = np.zeros( n_z, dtype=float )
      for z_idx in range(n_z):
        z_values = z_cohort.values[:,z_idx]
        #pdb.set_trace()
        if true_y.sum() == 0 or true_y.sum() == len(true_y):
          pdb.set_trace()
          
        aucs[z_idx]        = roc_auc_score( true_y, z_values )
        aucs[z_idx] = max(aucs[z_idx],1.0-aucs[z_idx])
        se_aucs[z_idx]     = auc_standard_error( aucs[z_idx], n_1, n_0 )
        se_random[t_idx]   = auc_standard_error( 0.5, n_1, n_0 )
        p_values[z_idx] = auc_p_value( aucs[z_idx], 0.5, se_aucs[z_idx], se_random[t_idx] )
      
      aucs_series = pd.Series( aucs, index = z_names, name=dna_gene)
      p_series = pd.Series( p_values, index = z_names, name=dna_gene)
      
      all_auc_series.append( [tissue_name,aucs_series] )
      all_p_series.append( [tissue_name,p_series] )
  
  z_names = all_auc_series[0][1].index.values
  multi_index = []
  for tissue, series  in all_auc_series:
    #tissue = x[0]; series = x[1]
    multi_index.append( (tissue, series.name) )
  #multi_index = np.array(multi_index).T
  index = pd.MultiIndex.from_tuples(multi_index, names=['tissue', 'gene'])
  
  auc_df = pd.DataFrame( index = index, columns = z_names )
  p_df = pd.DataFrame( index = index, columns = z_names )
  
  for tissue, series  in all_auc_series:
    gene = series.name
    auc_df.loc[tissue,gene] = series
  for tissue, series  in all_p_series:
    gene = series.name
    p_df.loc[tissue,gene] = series
  
  auc_df.to_csv( dna_dir + "/auc_values.csv")  
  p_df.to_csv( dna_dir + "/p_values.csv")      
  
  f=pp.figure()
  ax1 = f.add_subplot(121)
  ax2 = f.add_subplot(122)
  ax1.hist( p_df.values.flatten(), bins = np.linspace(0,0.5,21), lw=2, histtype="step", normed=True, range=(0,0.5) )
  ax2.hist( auc_df.values.flatten(), bins = np.linspace(0.5,1,21), lw=2, histtype="step", normed=True, range=(0.5,1) )
  pp.savefig( dna_dir + "/p_values_and_auc.png", fmt = "png", dpi=300)
  f=pp.figure()
  ax1 = f.add_subplot(121)
  ax2 = f.add_subplot(122)
  ax1.hist( p_df.values.flatten(), bins = np.linspace(0,0.5,61), lw=2, histtype="step", normed=True, range=(0,0.5) )
  ax2.hist( auc_df.values.flatten(), bins = np.linspace(0.5,1,61), lw=2, histtype="step", normed=True, range=(0.5,1) )
  pp.savefig( dna_dir + "/p_values_and_auc2.png", fmt = "png", dpi=300)
  
  f=pp.figure()
  for z in auc_df.columns:
    pp.semilogy( auc_df[z].values.flatten(), p_df[z].values.flatten(), '.' )
  pp.xlabel('AUC');pp.ylabel('p-value')
  pp.savefig( dna_dir + "/p_values_and_auc_scatter.png", fmt = "png", dpi=300)
  pp.close('all')
  return auc_df, p_df
  #pdb.set_trace()
  # sorted_pan = pd.concat(sorted_pan,axis=1)
  # sorted_pan.columns = dna_genes
  # sorted_pan.to_csv( dna_dir + "/pan_logpvals_z_for_dna.csv" )
  #
  # sorted_pan_sig = pd.concat(sorted_pan_sig,axis=1)
  # sorted_pan_sig.columns = dna_genes
  # sorted_pan_sig.to_csv( dna_dir + "/pan_sig_z_for_dna.csv" )
  # size1 = max( int( n_z*size_per_unit ), 12 )
  # size2 = max( int( len(dna_genes)*size_per_unit ), 12 )
  # f = sns.clustermap( sorted_pan.T, figsize=(size1,size2) )
  #
  # pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  # pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  # pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  # pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  # pp.savefig( dna_dir + "/z_for_dna_clustermap_logpval.png", format="png")
  #
  # f = sns.clustermap( sorted_pan_sig.T, figsize=(size1,size2) )
  #
  # pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  # pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  # pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  # pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  # pp.savefig( dna_dir + "/z_for_dna_clustermap_sig.png", format="png")

  #pp.close('all')

  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  a, p = main( data_location, results_location )
  # z_names = a[0][1].index.values
  # multi_index = []
  # for tissue, series  in a:
  #   #tissue = x[0]; series = x[1]
  #   multi_index.append( (tissue, series.name) )
  # #multi_index = np.array(multi_index).T
  # index = pd.MultiIndex.from_tuples(multi_index, names=['tissue', 'gene'])
  #
  # auc_df = pd.DataFrame( index = index, columns = z_names )
  # p_df = pd.DataFrame( index = index, columns = z_names )
  #
  # for tissue, series  in a:
  #   gene = series.name
  #   auc_df.loc[tissue,gene] = series
  # for tissue, series  in p:
  #   gene = series.name
  #   p_df.loc[tissue,gene] = series
  