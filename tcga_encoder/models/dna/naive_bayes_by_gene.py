from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from tcga_encoder.models.dna.models import *


BETA_METHOD    = "beta"
BETA_METHOD2    = "beta2"
POISSON_METHOD = "poisson"
GAUSSIAN_METHOD = "gaussian"
KDE_METHOD  = "kde"
NEGBIN_METHOD  = "negbin"
log_prior = 1e-2

def model_by_method( method ):
  if method == BETA_METHOD or method == BETA_METHOD2:
    return BetaNaiveBayesModel()
  elif method == POISSON_METHOD:
    return PoissonNaiveBayesModel()
  elif method == KDE_METHOD:
    return KernelDensityNaiveBayesModel()
  elif method == GAUSSIAN_METHOD:
    return GaussianNaiveBayesModel()
  elif method == NEGBIN_METHOD:
    return NegBinNaiveBayesModel()
  else:
    assert False, "No model called %s"%(method)
  return None

def compute_auc_by_disease( true_y, est_y, diseases ):
  u_diseases = np.unique(diseases)
  
  aucs = np.nan*np.ones( len(u_diseases) )
  d_idx = 0
  for disease in u_diseases:
    I = pp.find( disease == diseases )
    y = true_y[I]
    x = est_y[I]
    
    n = len(I)
    n_1 = y.sum()
    n_0 = n-n_1
    
    if n_1 > 0 and n_0 < n:
      aucs[ d_idx ] = roc_auc_score( y, x )
    
    d_idx += 1
  return aucs
    
    
  
def run_method( data, results_location, results_store, \
                dna_gene, source, method, disease_string, \
                n_folds, n_xval_repeats, \
                randomize_labels, label_permutation_idx ):
  
  colors = "rgkcmrgkcmrgkcmrgkcmrgkcmrgkcmrgkcmrgkcmrgkcmrgkcmrgkcmrgkcmrgkcmrgkcmrgkcmrgkcmrgkcm"
  
  dna_data, source_data = data              
  barcodes = source_data.index.values
  
  targets = dna_data.loc[ barcodes ].values
  inputs  = source_data.loc[ barcodes ].values
  
  # permute the original targets
  permuted_targets = targets.copy()
  if randomize_labels == True:
    np.random.seed( label_permutation_idx )
    np.random.shuffle( permuted_targets )
  
  
  labels = np.vstack((targets,permuted_targets)).T
  N =  len(permuted_targets) 
  D = source_data.shape[1]
  test_predictions = np.zeros( (N,n_xval_repeats), dtype=float )
  test_predictions_elementwise = np.zeros( (N,D), dtype=float )
  diseases = np.array( [s.split("_")[0] for s in barcodes])
  

  u_diseases = np.unique( diseases )
  n_diseases = len(u_diseases)
  one_hot_diseases = np.zeros( (N,n_diseases) )
  
  aucs = []
  aucs_by_disease=[]
  elementwise_aucs = np.zeros( (D,n_xval_repeats) )
  elementwise_aucs_by_disease = np.zeros( (n_diseases, D, n_xval_repeats) )
  
  weights = np.ones( len(diseases))
  k=0
  for disease in u_diseases:
    I = disease == diseases
    weights[I] = permuted_targets[I].sum()
    one_hot_diseases[I,k] = 1
    k+=1
  weights = np.array(weights)  
  if label_permutation_idx == 0:
    f = pp.figure()
    ax_roc = f.add_subplot(111)
  for xval_repeat_idx in range( n_xval_repeats ):
    print "\t\tINFO (%s): running xval repeat %d of %d"%(dna_gene,xval_repeat_idx+1, n_xval_repeats)
    
    # get train/test splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle = True, random_state=xval_repeat_idx+1)
    
    
    fold_idx = 1
    
    for train_index, test_index in skf.split(inputs, permuted_targets):
      #print "\t\t\tINFO (%s): running fold %d of %d"%(dna_gene,fold_idx, n_folds)
      model = model_by_method( method )
      model.fit( inputs[train_index,:], permuted_targets[train_index] )#, one_hot_diseases[train_index,:] )
      
      #fold_test_predictions             = model.predict( inputs[test_index,:], one_hot_groups=one_hot_diseases[test_index,:] )
      #fold_test_predictions_elementwise = model.predict( inputs[test_index,:], elementwise=True, one_hot_groups=one_hot_diseases[test_index,:] )
      fold_test_predictions             = model.predict( inputs[test_index,:]  )
      fold_test_predictions_elementwise = model.predict( inputs[test_index,:], elementwise=True )
      
      test_predictions[:,xval_repeat_idx][ test_index ] = fold_test_predictions
      test_predictions_elementwise[test_index,:] = fold_test_predictions_elementwise
      #pdb.set_trace()
      fold_idx+=1
    
    #pdb.set_trace()
    aucs_by_disease.append( compute_auc_by_disease( permuted_targets, test_predictions[:, xval_repeat_idx], diseases ) )  
    aucs.append( roc_auc_score( permuted_targets, test_predictions[:, xval_repeat_idx], sample_weight=weights ) )  
    print "\t\tINFO (%s): running AUC compute repeat %d of %d"%(dna_gene,xval_repeat_idx+1, n_xval_repeats)
    for d in xrange(D):
      elementwise_aucs[d,xval_repeat_idx] = roc_auc_score( permuted_targets, test_predictions_elementwise[:, d], sample_weight=np.array(weights) )
      elementwise_aucs_by_disease[:,d,xval_repeat_idx] = compute_auc_by_disease( permuted_targets, test_predictions_elementwise[:, d], diseases )
      
      if d==3 and label_permutation_idx==0:
        d_idx = 0
        
        for disease in u_diseases:
          I = disease == diseases
          fpr,tpr,thre = roc_curve( permuted_targets[I], test_predictions_elementwise[:, d][I] )
          #pdb.set_trace()
          #print d_idx, colors
          ax_roc.plot( fpr, tpr, colors[d_idx]+'.-', alpha=0.4 )
          d_idx+=1


        fpr,tpr,thre = roc_curve( permuted_targets, test_predictions_elementwise[:, d], sample_weight=np.array(weights) )
        ax_roc.plot( fpr, tpr, 'b.-', alpha = 0.4, lw=0.5)
        #pdb.set_trace()
        
        
        pp.title( source_data.columns[d])
        #pdb.set_trace()
    #pdb.set_trace()
    
  if label_permutation_idx == 0:
    figname1 = os.path.join( HOME_DIR, os.path.dirname(results_location) ) + "/rocs_%s_%s_%s_%s.png"%(dna_gene,source, method, disease_string)
    f.savefig(  figname1, dpi=300 )
  xval_columns = np.array( ["seed_%d"%(seed+1) for seed in range(n_xval_repeats) ] )
  
  results_store[ "/%s/%s/%s/%s/labels_%d/xval_aucs_elementwise"%(disease_string,dna_gene,source, method,label_permutation_idx)] = pd.DataFrame( elementwise_aucs, index = source_data.columns, columns=xval_columns )
  results_store[ "/%s/%s/%s/%s/labels_%d/xval_aucs"%(disease_string,dna_gene,source, method,label_permutation_idx)]        = pd.Series( np.array(aucs), index=xval_columns )
  results_store[ "/%s/%s/%s/%s/labels_%d/xval_predictions"%(disease_string,dna_gene,source, method,label_permutation_idx)] = pd.DataFrame( test_predictions, index = barcodes, columns = xval_columns )
  results_store[ "/%s/%s/%s/%s/labels_%d/xval_targets"%(disease_string,dna_gene,source, method,label_permutation_idx)]     = pd.DataFrame( labels, index = barcodes, columns = ["true","permuted"])

  for d_idx in range(len(u_diseases)):
    disease = u_diseases[d_idx]
    results_store[ "/%s/%s/%s/%s/labels_%d/diseases/%s/xval_aucs_elementwise"%(disease_string,dna_gene,source, method,label_permutation_idx,disease)] = pd.DataFrame( elementwise_aucs_by_disease[d_idx,:,:], index = source_data.columns, columns=xval_columns )
  results_store[ "/%s/%s/%s/%s/labels_%d/xval_disease_aucs"%(disease_string,dna_gene,source, method,label_permutation_idx)] = pd.DataFrame( np.array(aucs_by_disease).T, index = u_diseases, columns=xval_columns )

def prepare_results_store( results_location, mode = "a" ):
  # create directory path for results
  check_and_mkdir( os.path.join( HOME_DIR, os.path.dirname(results_location) ) )
  
  # open store in append mode, incase it already exists
  results_store = pd.HDFStore( os.path.join( HOME_DIR, results_location ), mode )
  
  return results_store
  
def prepare_data_store( data_file, dna_gene, source, method, restricted_diseases ):
  # later add restrictions on tissue type
  #source = source.upper()
  
  data_store = pd.HDFStore( data_file, "r" )
  #data_store.open()
  
  # first get columns on observed, then get intersection
  # sources = [DNA,source]
  sources = [DNA, source]
  
  
  observed = data_store["/CLINICAL/observed"][ sources ] 
  barcodes = observed[ observed.sum(1)==len(sources) ].index.values
  
  
  dna_data    = data_store["/DNA/channel/0"].loc[ barcodes ][ dna_gene ]
  source_data = None
  
  if source == RNA:
    if method == BETA_METHOD:
      source_data = data_store["/RNA/FAIR"].loc[ barcodes ]
    elif method == POISSON_METHOD or method == GAUSSIAN_METHOD or method == KDE_METHOD:
      source_data = np.log2( data_store["/RNA/RSEM"].loc[ barcodes ] + log_prior )
    elif method == NEGBIN_METHOD:
      #source_data = np.log2( np.maximum( 2.0, data_store["/RNA/RSEM"].loc[ barcodes ]+ log_prior ) )
      source_data = data_store["/RNA/RSEM"].loc[ barcodes ]
      
  elif source == miRNA:
    if method == BETA_METHOD:
      source_data = data_store["/miRNA/FAIR"].loc[ barcodes ]
    elif method == POISSON_METHOD or method == GAUSSIAN_METHOD or method == KDE_METHOD:
      source_data = np.log2( data_store["/miRNA/READS"].loc[ barcodes ] + log_prior )
    elif method == NEGBIN_METHOD:
      source_data = data_store["/miRNA/READS"].loc[ barcodes ]
      
  elif source == METH:
    if method == BETA_METHOD:
      source_data = data_store["/METH/FAIR"].loc[ barcodes ]
    elif method == POISSON_METHOD or method == GAUSSIAN_METHOD or method == KDE_METHOD:
      source_data = np.log2( data_store["/METH/METH"].loc[ barcodes ]  )
    elif method == BETA_METHOD2:
      source_data = data_store["/METH/METH"].loc[ barcodes ]
    elif method == NEGBIN_METHOD:
      source_data = data_store["/METH/METH"].loc[ barcodes ]
  data_store.close()
  
  if source_data is None or dna_data is None:
    return None 
  else:
    
    if len(restricted_diseases)>0:
      diseases = np.array( [s.split("_")[0] for s in barcodes])
      ok = np.zeros( len(diseases), dtype = bool)
      for disease in restricted_diseases:
        ok |= diseases == disease
      
      source_data = source_data[ ok ]
      dna_data = dna_data[ ok ]
      #pdb.set_trace()
      barcodes = barcodes[ok]
    print "\tINFO: %s has %d of %d mutated (%0.2f percent)"%( dna_gene, dna_data.sum(), len(barcodes), 100.0*dna_data.sum() / float(len(barcodes)) )
    
    
    return dna_data, source_data

def run_train( data_file, results_location, dna_gene, source, method, n_folds, n_xval_repeats, n_permutations, restricted_diseases ):
  
  disease_string = "ALL"
  if len(restricted_diseases)>0:
    disease_string = restricted_diseases[0]
    for d in restricted_diseases[1:]:
      disease_string += "_%s"%(d)
    
  # extract in for the dna_gene
  data = prepare_data_store( data_file, dna_gene, source, method, restricted_diseases )
  
  if data is None:
    print "Skipping gene %s"%dna_gene
    return
  
  # prepare HDF store
  results_store = prepare_results_store( results_location )
   
  # run train with correct labels
  print "..............................................................."
  print "\tINFO (%s): Running with correct labels..."%(dna_gene)
  run_method( data, results_location, results_store, dna_gene, source, method, disease_string, n_folds, n_xval_repeats, randomize_labels = False, label_permutation_idx = 0)

  # run a nbr of permutated xval repeats
  for permutation_idx in range(n_permutations):

    print "..............................................................."
    print "\tINFO (%s): Running with permuted labels...%d of %d"%(dna_gene,permutation_idx+1, n_permutations)
    run_method( data, results_location, results_store, dna_gene, source, method, disease_string, n_folds, n_xval_repeats, randomize_labels = True, label_permutation_idx = permutation_idx+1)
  
  view_results( results_location, results_store, dna_gene, n_permutations, source, method, disease_string, title_str = "all", max_nbr=1000, zoom = False )
  view_results( results_location, results_store, dna_gene, n_permutations, source, method, disease_string, title_str = "zoom", max_nbr=5, zoom=True )
  print "... done run_train."  

def view_results( location, store, gene, n_permutations, source, method, disease_string, title_str = "", max_nbr = 100, zoom = True ):
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


def main( data_file, results_location, dna_gene, source, method, n_folds, n_xval_repeats, n_permutations, train, restricted_diseases ):
  print "***************************************************************"
  print "Data:     ", data_file
  print "Results:  ", results_location
  print "DNA Gene: ", dna_gene
  print "source:   ", source
  print "method:   ", method
  print "folds:    ", n_folds
  print "n xvals:  ", n_xval_repeats
  print "permutes: ", n_permutations
  if train == True:
    print "TRAINING"
  else:
    print "REPORTING"
  if len(restricted_diseases) > 0:
    print "diseases: ", restricted_diseases
  else:
    print "diseases: ALL"
  print "***************************************************************"
  
  if train:
    run_train( data_file, results_location, dna_gene, source, method, n_folds, n_xval_repeats, n_permutations, restricted_diseases )
  else:
    run_report( data_file, results_location, dna_gene, source, method, n_folds, n_xval_repeats, n_permutations, restricted_diseases )
    

if __name__ == "__main__":
  #assert len(sys.argv) >= 2, "Must pass yaml file."
  data_file        = sys.argv[1]
  results_location = sys.argv[2]
  dna_gene         = sys.argv[3]
  source           = sys.argv[4]
  method           = sys.argv[5]
  
  
  n_folds        = 4   # nbr of folds per xval repeat
  n_xval_repeats = 5   # nbr of xval permutations/repeats to try
  n_permutations = 10  # nbr of random label assignments to try
  
  if len(sys.argv) >= 9:
    n_folds        = int( sys.argv[6] )
    n_xval_repeats = int( sys.argv[7] )
    n_permutations = int( sys.argv[8] )
    
  #train = False
  #if len(sys.argv) == 10:
  train = bool(int( sys.argv[9]))
    
  restricted_diseases = []
  idx = 10
  while len(sys.argv) > idx:
    restricted_diseases.append( sys.argv[idx] )
    idx += 1
  
  
  main( data_file, results_location, dna_gene, source, method, n_folds, n_xval_repeats, n_permutations, train, restricted_diseases )
  #pdb.set_trace()