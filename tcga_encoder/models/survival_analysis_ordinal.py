from tcga_encoder.utils.helpers import *
from tcga_encoder.definitions.locations import *
import sklearn
from sklearn.model_selection import KFold
from tcga_encoder.models.lda import LinearDiscriminantAnalysis
from tcga_encoder.models.ordinal_regression import *
import pdb
from lifelines import KaplanMeierFitter

def bootstraps( x, m, replace = True ):
  # samples from arange(n) with replacement, m times.
  #x = np.arange(n, dtype=int)
  if m == 0:
    m = 1
    replace = False
    
  n = len(x)
  N = np.zeros( (m,n), dtype=int)
  for i in range(m):
    N[i,:] = sklearn.utils.resample( x, replace = replace )
    
  return N
  
def xval_folds( n, K, randomize = False, seed = None ):
  if randomize:
    if seed is not None:
      np.random.seed(seed)
      
    x = np.random.permutation(n)
  else:
    x = np.arange(n,dtype=int)
    
  kf = KFold( K )
  train = []
  test = []
  for train_ids, test_ids in kf.split( x ):
    #train_ids = np.setdiff1d( x, test_ids )
    
    train.append( train_ids )
    test.append( test_ids )
  
  return train, test
  
def ordinal_regression_with_xval_and_bootstrap( X, e, t, k_fold = 10, n_bootstraps = 10, randomize = True, seed = 0, epsilon = 1e-12, l1=0,l2=0 ):
  
  n,d = X.shape
  assert len(e) == n, "incorrect sizes"
  assert len(t) == n, "incorrect sizes"
  
  train_folds, test_folds = xval_folds( n, k_fold, randomize = randomize, seed = seed )
  
  avg_projection = np.zeros( n, dtype=float )
  avg_probability = np.zeros( n, dtype=float )
  
  mean_projections = np.zeros( n, dtype=float )
  var_projections  = np.zeros( n, dtype=float )
  
  mean_probabilities = np.zeros( n, dtype=float )
  var_probabilities  = np.zeros( n, dtype=float )
  
  # for each fold, compute mean and variances
  w_mean = np.zeros( (k_fold,d), dtype = float )
  w_var = np.zeros( (k_fold,d), dtype = float )
  
  for k, train_ids, test_ids in zip( range(k_fold), train_folds, test_folds ):
    X_test = X[test_ids,:]
    bootstrap_ids = bootstraps( train_ids, n_bootstraps )
    
    for bootstrap_train_ids in bootstrap_ids:
      #pdb.set_trace()
      X_train = X[bootstrap_train_ids,:]
      t_train = t[bootstrap_train_ids]
      e_train = e[bootstrap_train_ids]
      
      w_ord,b_ord = ordinal_regression( X_train, e_train, t_train, l1=l1, l2=l2 )
      
      lda = LinearDiscriminantAnalysis(epsilon=epsilon)
      lda.fit( X_train, e_train )
      lda.w_prop_to = np.squeeze(w_ord)
      lda.fit_density()
      w = lda.w_prop_to
      
      test_proj = lda.transform( X_test )
      #ranked = np.argsort(test_proj).astype(float) / len(test_proj)
      #test_proj = ranked
      test_prob = lda.prob( X_test )
      train_prob = lda.prob( X_train )
      I=pp.find( np.isinf(test_prob) )
      test_prob[I] = 1
      
      #auc = roc_auc_score( e[test_ids], test_prob )
      #auc_tr = roc_auc_score( e_train, train_prob )
      
      
      test_predict = lda.predict( X_test )
      
      #print "AUCS: ", auc_tr, auc
      # if auc < 0.5:
      #   test_prob = 1.0-test_prob
      #   test_proj *= -1
      #   w *= -1
        
      mean_projections[ test_ids ]   += test_proj
      mean_probabilities[ test_ids ] += test_prob
      
      var_projections[ test_ids ]   += np.square( test_proj )
      var_probabilities[ test_ids ] += np.square( test_predict )
      
      w_mean[k] += w
      w_var[k] += np.square(w)
      #pdb.set_trace()
    # w_mn = w_mean[k] / n_bootstraps
    #
    # lda = LinearDiscriminantAnalysis(epsilon=epsilon)
    # lda.fit( X[train_ids,:], y[train_ids] )
    # lda.w_prop_to =   w_mn
    # lda.fit_density()
    #
    # avg_projection[ test_ids ] = lda.transform( X_test )
    # avg_probability[ test_ids ] = lda.prob( X_test )
   
  if n_bootstraps == 0:
    n_bootstraps = 1
    
  w_mean /= n_bootstraps
  w_var   /= n_bootstraps 
  w_var   -= np.square( w_mean )
    
  mean_projections /= n_bootstraps
  var_projections   /= n_bootstraps
  mean_probabilities /= n_bootstraps
  var_probabilities   /= n_bootstraps
  
  var_projections   -= np.square( mean_projections )
  var_probabilities -= np.square( mean_probabilities )
  
  return (mean_projections,var_projections),(mean_probabilities,var_probabilities),(w_mean,w_var),(avg_projection,avg_probability)

def run_survival_analysis( disease_list, fill_store, data_store, k_fold = 10, n_bootstraps = 10, epsilon = 1e-12, l1=0, l2=0 ):
  fill_store.open()
  data_store.open()
  ALL_SURVIVAL = data_store["/CLINICAL/data"][["patient.days_to_last_followup","patient.days_to_death"]]
  tissue_barcodes = np.array( ALL_SURVIVAL.index.tolist(), dtype=str )
  surv_barcodes = np.array([ x+"_"+y for x,y in tissue_barcodes])
  NEW_SURVIVAL = pd.DataFrame( ALL_SURVIVAL.values, index =surv_barcodes, columns = ALL_SURVIVAL.columns ) 
  val_survival  = pd.concat( [NEW_SURVIVAL, fill_store["/Z/VAL/Z/mu"]], axis=1, join = 'inner' )
  
  fill_store.close()
  data_store.close()
  #-------
  predict_survival_train = val_survival #pd.concat( [test_survival, val_survival], axis=0, join = 'outer' )
  predict_barcodes_train = predict_survival_train.index
  splt = np.array( [ [s.split("_")[0], s.split("_")[1]] for s in predict_barcodes_train ] )
  predict_survival_train = pd.DataFrame( predict_survival_train.values, index = splt[:,1], columns = predict_survival_train.columns )
  predict_survival_train["disease"] = splt[:,0]

  Times_train = predict_survival_train[ "patient.days_to_last_followup" ].fillna(0).values.astype(int)+predict_survival_train[ "patient.days_to_death" ].fillna(0).values.astype(int)
  predict_survival_train["T"] = Times_train
  Events_train = (1-np.isnan( predict_survival_train[ "patient.days_to_death" ].astype(float)) ).astype(int)
  predict_survival_train["E"] = Events_train
  
  
  X_columns = val_survival.columns[2:]
  X = predict_survival_train[X_columns].values.astype(float)
  e = predict_survival_train["E"].values.astype(int)
  t = predict_survival_train["T"].values.astype(float)
  t /= 365.0
  projections, probabilties, weights, averages = ordinal_regression_with_xval_and_bootstrap( X, e, t,  k_fold = k_fold, n_bootstraps = n_bootstraps, l1 = l1, l2 = l2 )
  
  return projections, probabilties, weights, averages, X, e, t, Events_train, Times_train

if __name__ == "__main__":
  
  disease = "brca"
  data_file = "pan_tiny_multi_set"
  experiment_name = "tiny_leave_%s_out"%(disease)
  
  if len(sys.argv) == 4:
    disease   = sys.argv[1]
    data_file = sys.argv[2]
    #experiment_name = sys.argv[3]
    
    data_location = os.path.join( HOME_DIR, "data/broad_processed_post_recomb/20160128/%s/data.h5"%(data_file) )
    fill_location = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out/medium/leave_out_%s/full_vae_fill.h5"%(disease) )
    survival_location = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out/medium/leave_out_%s/full_vae_survival.h5"%(disease) )
    savename = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out/medium/leave_out_%s/survival_xval_ordinal.png"%(disease))
  else:
    data_location = os.path.join( HOME_DIR, "data/broad_processed_post_recomb/20160128/%s/data.h5"%(data_file) )
    fill_location = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out_sandbox/tiny/leave_out_%s/full_vae_fill.h5"%(disease) )
    survival_location = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out_sandbox/tiny/leave_out_%s/full_vae_survival.h5"%(disease) )
    savename = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out/tiny/leave_out_%s/survival_xval_ordinal.png"%(disease))
  
  s=pd.HDFStore( survival_location, "r" )
  d=pd.HDFStore( data_location, "r" )
  f=pd.HDFStore( fill_location, "r" ) 
  
  l1 = 0.01
  l2 = 0.0
  projections, probabilties, weights, averages, X, e, t, E_train, T_train = run_survival_analysis( [disease], f, d, k_fold = 20, n_bootstraps = 10, epsilon= 0.1, l1 = l1, l2 = l2 )  
  
  avg_proj = averages[0]
  avg_prob = averages[1]
  
  f = pp.figure()
  mn_proj = projections[0]
  std_proj = np.sqrt(projections[1])
  mn_prob = probabilties[0]
  std_prob = np.sqrt(probabilties[1])
  mn_w = weights[0]
  std_w = np.sqrt(weights[1])
  
  ax1 = f.add_subplot(211)
  I = np.argsort(-mn_proj)
  ax1.plot( mn_proj[I], mn_prob[I], 'o')
  ax2 = f.add_subplot(212)
  ax2.plot( mn_w.T, 'o-')
  
  #I = np.argsort( mn_prob )
  I1 = pp.find( mn_prob > np.median(mn_prob) )
  I0 = pp.find( mn_prob <= np.median(mn_prob) )
  #I1 = pp.find( mn_prob > 0.4 )
  #I0 = pp.find( mn_prob <= 0.4 )
  #I1 = pp.find( avg_prob > np.median(avg_prob) )
  #I0 = pp.find( avg_prob <= np.median(avg_prob) )
  
  f = pp.figure()
  ax3 = f.add_subplot(121)
  ax4 = f.add_subplot(122)
  
  kmf = KaplanMeierFitter()
  if len(I1) > 0:
    kmf.fit(T_train[I1], event_observed=E_train[I1], label =  "lda_1 E=%d C=%d"%(E_train[I1].sum(),len(I1)-E_train[I1].sum()))
    ax3=kmf.plot(ax=ax3,at_risk_counts=False,show_censors=True, color='red')
  if len(I0) > 0:
    kmf.fit(T_train[I0], event_observed=E_train[I0], label = "lda_0 E=%d C=%d"%(E_train[I0].sum(),len(I0)-E_train[I0].sum()))
    ax3=kmf.plot(ax=ax3,at_risk_counts=False,show_censors=True, color='blue')
  I1 = pp.find( mn_prob > 0.5 )
  I0 = pp.find( mn_prob <= 0.5 )
  kmf = KaplanMeierFitter()
  if len(I1) > 0:
    kmf.fit(T_train[I1], event_observed=E_train[I1], label =  "lda_1 E=%d C=%d"%(E_train[I1].sum(),len(I1)-E_train[I1].sum()))
    ax4=kmf.plot(ax=ax4,at_risk_counts=False,show_censors=True, color='red')
  if len(I0) > 0:
    kmf.fit(T_train[I0], event_observed=E_train[I0], label = "lda_0 E=%d C=%d"%(E_train[I0].sum(),len(I0)-E_train[I0].sum()))
    ax4=kmf.plot(ax=ax4,at_risk_counts=False,show_censors=True, color='blue')

  
  
  print "ROC mn_prob ", roc_auc_score(e,mn_prob)
  pp.savefig(savename, dpi=300, format='png')
  #print "ROC avg_prob ", roc_auc_score(e,avg_prob)
  pp.show()
  
   