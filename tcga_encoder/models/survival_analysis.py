from tcga_encoder.utils.helpers import *
from tcga_encoder.definitions.locations import *
import sklearn
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import KFold
from tcga_encoder.models.lda import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sk_LinearDiscriminantAnalysis
from tcga_encoder.models.survival import *
import pdb
from lifelines import KaplanMeierFitter
#import autograd.numpy as np
#from autograd import grad

#def cost_lineat_reg_ard( X, y, w, b, h );
def linear_reg_gprior_ard( X, y, alpha, lr = 1e-4, iters = 10, verbose = False, eps = 1e-6 ):
  n,d = X.shape
  
  XTX = np.dot( X.T, X )
  XXT = np.dot( X, X.T )
  
  XX = X*X
  sX = XX.sum(0)
  w = np.zeros(d,dtype=float)
  h = np.zeros(d,dtype=float)
  z = np.exp(h)
  b = np.mean(y)
  
  for i in range(iters):
    y_hat = np.dot( X, w ) + b
    if verbose:
      print "Error = ", np.sum( np.square( y - y_hat ) ), w[:5], b, z[:5]
    g_w = - np.dot( X.T, (y-y_hat) ) + (z+eps)*np.sign(w)
    g_b = np.sum(y-np.dot( X, w ))
    ##g_h = np.abs(w)*z - alpha*np.dot( X.T, np.dot( X, (z+eps)**-2 ))*z
    g_h = np.abs(w)*z - np.dot( sX.T, alpha*z*(z+eps)**-2 )
    #g_w = - np.dot( X.T, (y-y_hat) ) + z*np.sign(w)
    #g_b = np.sum(y-np.dot( X, w ))
    #g_h = np.abs(w)*z - alpha*np.dot( X.T, np.dot( X, z**-1 ))
    #g_h = np.abs(w)*z - np.dot( sX.T, alpha*z**-1 )

    if np.any(np.isnan(g_w)):
      pdb.set_trace()
    if np.any(np.isinf(g_w)):
      pdb.set_trace()
    if np.any(np.isnan(g_h)):
      pdb.set_trace()
    if np.any(np.isinf(g_h)):
      pdb.set_trace()

    
    h = np.maximum( h - lr*g_h, -20 )
    w = w - lr*g_w
    b = b - lr*g_b
    z = np.exp(h)
    
    if np.any(np.isnan(w)):
      pdb.set_trace()
  
  y_hat = np.dot( X, w ) + b
  if verbose:
   print "Final Error = ", np.sum( np.square( y - y_hat ) )
  #pdb.set_trace()
  return w, b
    
def bootstraps( x, m ):
  # samples from arange(n) with replacement, m times.
  #x = np.arange(n, dtype=int)
  n = len(x)
  N = np.zeros( (m,n), dtype=int)
  for i in range(m):
    N[i,:] = sklearn.utils.resample( x, replace = True )
    
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

def kmeans_survival( X, y, K ):
  
  kmeans = KMeans(n_clusters=K ).fit(X.astype(float))
  predictions = kmeans.predict(X)
  # f = pp.figure()
  # kmf = KaplanMeierFitter()
  # ax1 = f.add_subplot(311)
  # ax2 = f.add_subplot(312)
  # ax3 = f.add_subplot(313)
  #
  # test_labels = []
  # if len(Z_test) > 0:
  #   test_labels = kmeans.predict( Z_test.astype(float) )
  #   #pdb.set_trace()
  #
  # colours = "brgkmcbrgkmcbrgkmcbrgkmcbrgkmcbrgkmcbrgkmc"
  # for k in range(K):
  #   I = pp.find( kmeans.labels_==k)
  #   Ti=T_train[I]
  #   Ei=E_train[I]
  #
  #   if len(Ti)>0:
  #     kmf.fit(Ti, event_observed=Ei, label = "train_k=%d"%k)
  #     ax1=kmf.plot(ax=ax1, color=colours[k])
  #
  #   if len(test_labels) > 0:
  #     I_test = pp.find( test_labels==k)
  #     Ti_test=T_test[I_test]
  #     Ei_test=E_test[I_test]
  #
  #     if len(Ti_test)>0:
  #       kmf.fit(Ti_test, event_observed=Ei_test, label = "test_k=%d"%k)
  #       ax2=kmf.plot(ax=ax2, color=colours[k])
  #
  #     T = np.hstack( (Ti,Ti_test))
  #     E = np.hstack( (Ei,Ei_test))
  #     if len(T)>0:
  #       kmf.fit(T, event_observed=E, label = "all_k=%d"%k)
  #       ax3=kmf.plot(ax=ax3, color=colours[k])
  #   #pdb.set_trace()
  # pp.suptitle("%s"%(disease))
  
  return predictions
  
  return (mean_projections,var_projections),(mean_probabilities,var_probabilities),(w_mean,w_var),(avg_projection,avg_probability)

def lda_with_xval_and_bootstrap( X, y, k_fold = 10, n_bootstraps = 10, randomize = True, seed = 0, epsilon = 1e-12 ):
  
  print "epsilon", epsilon
  n,d = X.shape
  assert len(y) == n, "incorrect sizes"
  
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
      y_train = y[bootstrap_train_ids]
      
      
      lda = LinearDiscriminantAnalysis(epsilon=epsilon)
      lda.fit( X_train, y_train )
      
      
      
      w = lda.w_prop_to

      sk_lda = sk_LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
      sk_lda.fit( X_train, y_train )
      try:
        sk_test_proj = np.squeeze(sk_lda.predict_log_proba( X_test ))[:,1]
      
        test_proj = sk_test_proj #lda.transform( X_test )
        test_prob = np.squeeze(sk_lda.predict_proba( X_test )[:,1]) # lda.predict( X_test )
      except:
        test_proj = lda.transform( X_test )
        test_prob = lda.prob( X_test )
        
      
      #pdb.set_trace()
      #ranked = np.argsort(test_proj).astype(float) / len(test_proj)
      #test_proj = ranked
      #test_prob = lda.prob( X_test )
      #test_proj = sk_test_proj
      #I=pp.find( np.isinf(test_prob) )
      #test_prob[I] = 1
      #test_prob = np.squeeze(sk_lda.predict_proba( X_test )[:,1]) # lda.predict( X_test )
      #test_predic
      mean_projections[ test_ids ]   += test_proj
      mean_probabilities[ test_ids ] += test_prob
      
      var_projections[ test_ids ]   += np.square( test_proj )
      var_probabilities[ test_ids ] += np.square( test_prob )
      
      w_mean[k] += w
      w_var[k] += np.square(w)
    w_mn = w_mean[k] / n_bootstraps
   
    lda = LinearDiscriminantAnalysis(epsilon=epsilon)
    lda.fit( X[train_ids,:], y[train_ids] )
    lda.w_prop_to =   w_mn
    lda.fit_density()
   
    avg_projection[ test_ids ] = lda.transform( X_test )
    avg_probability[ test_ids ] = lda.prob( X_test )
  I=pp.find( np.isinf(avg_probability) )
  avg_probability[I] = 1 
    
  w_mean /= n_bootstraps
  w_var   /= n_bootstraps 
  w_var   -= np.square( w_mean )
    
  print "xval w = ", w_mean.mean(0), w_var.mean(0)
  
  mean_projections /= n_bootstraps
  var_projections   /= n_bootstraps
  mean_probabilities /= n_bootstraps
  var_probabilities   /= n_bootstraps
  
  var_projections   -= np.square( mean_projections )
  var_probabilities -= np.square( mean_probabilities )
  
  return (mean_projections,var_projections),(mean_probabilities,var_probabilities),(w_mean,w_var),(avg_projection,avg_probability)

def predict_groups_with_loo_with_regression_gprior( X, y, C ):
  
  #print "epsilon", epsilon
  n,d = X.shape
  assert len(y) == n, "incorrect sizes"
  
  #train_folds, test_folds = xval_folds( n, k_fold, randomize = randomize, seed = seed )
  
  avg_projection = np.zeros( n, dtype=float )
  avg_probability = np.zeros( n, dtype=float )
  
  mean_projections = np.zeros( n, dtype=float )
  var_projections  = np.zeros( n, dtype=float )
  
  mean_probabilities = np.zeros( n, dtype=float )
  var_probabilities  = np.zeros( n, dtype=float )
  
  # for each fold, compute mean and variances
  w_mean = np.zeros( (n,d), dtype = float )
  w_var = np.zeros( (n,d), dtype = float )
  
  all_I = np.arange(n,dtype=int)
  
  Ws = []
  bs = []
  for i in xrange(n):
    train_ids = np.setdiff1d( all_I, i )
    test_ids = [i]
    
    X_test = X[test_ids,:]
    #bootstrap_ids = bootstraps( train_ids, n_bootstraps )
    
    X_train = X[train_ids,:]
    y_train = y[train_ids]
    
    #sklearn.linear_model.LogisticRegression()
    penalty="l2"
    #
    #sk_lda = sklearn.linear_model.ARDRegression(alpha=0.5, fit_intercept=True, verbose=True)
    #sk_lda = sklearn.linear_model.ElasticNet(alpha=0.5, fit_intercept=True)
    #sk_lda = sklearn.linear_model.Ridge(alpha=1.5, fit_intercept=True)
    sk_lda = sklearn.linear_model.Lasso(alpha=C, fit_intercept=True)
    #sklearn.linear_model.BayesianRidge
    #sk_lda = sklearn.linear_model.BayesianRidge(fit_intercept=False, verbose=True)
    sk_lda.fit( X_train, y_train )
    #pdb.set_trace()
    sk_test_proj = np.squeeze(sk_lda.predict( X_test ))
    test_proj = sk_test_proj #lda.transform( X_test )
    #test_prob = np.squeeze(sk_lda.predict_proba( X_test ))[1] # lda.predict( X_test )


    mean_projections[ i ]   += test_proj
    #mean_probabilities[ i ] += test_prob
    
    var_projections[ i ]   += np.square( test_proj )
    #var_probabilities[ i ] += np.square( test_prob )
    
    w = np.squeeze( sk_lda.coef_ )
    Ws.append( w )
    bs.append( sk_lda.intercept_ )
    w_mean[i] += w
    w_var[i] += np.square(w)
  
  Ws = np.array(Ws)
  bs = np.array(bs)
  w_mn = w_mean.mean(0)
 
  w_var   = w_mean.var(0)
    
  #print "loo     w = ", w_mn
  #print "loo w_var = ", w_var
  
  var_projections   -= np.square( mean_projections )
  #var_probabilities -= np.square( mean_probabilities )
  avg_projection=mean_projections
  #avg_probability=mean_probabilities
  return (mean_projections,var_projections),(w_mn,w_var,Ws,bs),(avg_projection,)
  
def predict_groups_with_loo_with_regression( X, y, C ):
  
  #print "epsilon", epsilon
  n,d = X.shape
  assert len(y) == n, "incorrect sizes"
  
  #train_folds, test_folds = xval_folds( n, k_fold, randomize = randomize, seed = seed )
  
  avg_projection = np.zeros( n, dtype=float )
  avg_probability = np.zeros( n, dtype=float )
  
  mean_projections = np.zeros( n, dtype=float )
  var_projections  = np.zeros( n, dtype=float )
  
  mean_probabilities = np.zeros( n, dtype=float )
  var_probabilities  = np.zeros( n, dtype=float )
  
  # for each fold, compute mean and variances
  w_mean = np.zeros( (n,d), dtype = float )
  w_var = np.zeros( (n,d), dtype = float )
  
  all_I = np.arange(n,dtype=int)
  
  Ws = []
  bs = []
  for i in xrange(n):
    train_ids = np.setdiff1d( all_I, i )
    test_ids = [i]
    
    X_test = X[test_ids,:]
    #bootstrap_ids = bootstraps( train_ids, n_bootstraps )
    
    X_train = X[train_ids,:]
    y_train = y[train_ids]
    
    # mn_x = X_train.mean(0)
    # std_x = X_train.std(0); i_bad = pp.find( std_x == 0 ); std_x[i_bad]=1.0
    # mn_y = y_train.mean()
    # std_y = y_train.std()
    #X_train -= mn_x;
    #X_train /= std_x

    #X_test -= mn_x;
    #X_test /= std_x
    
    #X_train -= 0.5
    #X_test -= 0.5
    # I_ = X_test < -0.5
    # X_test = (1-I_)*X_test - I_
    # I_ = X_test > 0.5
    # X_test = (1-I_)*X_test + I_
    #y_train -= mn_y;# y_train /= std_y
    
    #sklearn.linear_model.LogisticRegression()
    penalty="l2"
    #
    #sk_lda = sklearn.linear_model.ARDRegression(alpha=0.5, fit_intercept=True, verbose=True)
    #sk_lda = sklearn.linear_model.ElasticNet(alpha=0.5, fit_intercept=True)
    #sk_lda = sklearn.linear_model.Ridge(alpha=1.5, fit_intercept=True)
    sk_lda = sklearn.linear_model.Lasso(alpha=C, fit_intercept=True)
    #sklearn.linear_model.BayesianRidge
    #sk_lda = sklearn.linear_model.BayesianRidge(fit_intercept=False, verbose=True)
    sk_lda.fit( X_train, y_train )
    
    w_ard, b_ard = linear_reg_gprior_ard( X_train, y_train, C, lr = 0.0001, iters=2000, verbose = False )
    
    
    sk_test_proj = np.squeeze(sk_lda.predict( X_test ))
    sk_test_proj = np.dot( X_test, w_ard ) + b_ard
    test_proj = sk_test_proj #lda.transform( X_test )
    #test_prob = np.squeeze(sk_lda.predict_proba( X_test ))[1] # lda.predict( X_test )


    mean_projections[ i ]   += test_proj
    #mean_probabilities[ i ] += test_prob
    
    var_projections[ i ]   += np.square( test_proj )
    #var_probabilities[ i ] += np.square( test_prob )
    
    w = w_ard #np.squeeze( sk_lda.coef_ )
    #pdb.set_trace()
    Ws.append( w )
    bs.append( sk_lda.intercept_ )
    w_mean[i] += w
    w_var[i] += np.square(w)
  
  Ws = np.array(Ws)
  bs = np.array(bs)
  w_mn = w_mean.mean(0)
 
  w_var   = w_mean.var(0)
    
  #print "loo     w = ", w_mn
  #print "loo w_var = ", w_var
  
  var_projections   -= np.square( mean_projections )
  #var_probabilities -= np.square( mean_probabilities )
  avg_projection=mean_projections
  #avg_probability=mean_probabilities
  return (mean_projections,var_projections),(w_mn,w_var,Ws,bs),(avg_projection,)
  
def predict_groups_with_loo( X, y, C ):
  
  #print "epsilon", epsilon
  n,d = X.shape
  assert len(y) == n, "incorrect sizes"
  
  #train_folds, test_folds = xval_folds( n, k_fold, randomize = randomize, seed = seed )
  
  avg_projection = np.zeros( n, dtype=float )
  avg_probability = np.zeros( n, dtype=float )
  
  mean_projections = np.zeros( n, dtype=float )
  var_projections  = np.zeros( n, dtype=float )
  
  mean_probabilities = np.zeros( n, dtype=float )
  var_probabilities  = np.zeros( n, dtype=float )
  
  # for each fold, compute mean and variances
  w_mean = np.zeros( (n,d), dtype = float )
  w_var = np.zeros( (n,d), dtype = float )
  
  all_I = np.arange(n,dtype=int)
  Ws = []
  for i in xrange(n):
    train_ids = np.setdiff1d( all_I, i )
    test_ids = [i]
    
    X_test = X[test_ids,:]
    #bootstrap_ids = bootstraps( train_ids, n_bootstraps )
    
    X_train = X[train_ids,:]
    y_train = y[train_ids]
    # X_train -= 0.5
    # X_test -= 0.5
    
    # mn_x = X_train.mean(0)
    # std_x = X_train.std(0); i_bad = pp.find( std_x == 0 ); std_x[i_bad]=1.0
    # mn_y = y_train.mean()
    # std_y = y_train.std()
    # X_train -= mn_x;
    # X_train /= std_x
    #
    # X_test -= mn_x;
    # X_test /= std_x
    
    
    #sklearn.linear_model.LogisticRegression()
    penalty="l2"
    if penalty == "l1":
      sk_lda = sklearn.linear_model.LogisticRegression(C=C, penalty='l1',solver='liblinear', fit_intercept=True)
    else:
      sk_lda = sklearn.linear_model.LogisticRegression(solver='liblinear', C=C, penalty='l2', fit_intercept=True)
    sk_lda.fit( X_train, y_train )
    #pdb.set_trace()
    sk_test_proj = np.squeeze(sk_lda.predict_log_proba( X_test ))[1]
    test_proj = sk_test_proj #lda.transform( X_test )
    test_prob = np.squeeze(sk_lda.predict_proba( X_test ))[1] # lda.predict( X_test )


    mean_projections[ i ]   += test_proj
    mean_probabilities[ i ] += test_prob
    
    var_projections[ i ]   += np.square( test_proj )
    var_probabilities[ i ] += np.square( test_prob )
    
    w = np.squeeze( sk_lda.coef_ )
    Ws.append( w )
    w_mean[i] += w
    w_var[i] += np.square(w)
  
  w_mn = w_mean.mean(0)
 
  w_var   = w_mean.var(0)
  Ws = np.array(Ws)  
  #print "loo     w = ", w_mn
  #print "loo w_var = ", w_var
  
  var_projections   -= np.square( mean_projections )
  var_probabilities -= np.square( mean_probabilities )
  avg_projection=mean_projections
  avg_probability=mean_probabilities
  return (mean_projections,var_projections),(mean_probabilities,var_probabilities),(w_mn,w_var,Ws),(avg_projection,avg_probability)
  
def lda_with_loo( X, y, epsilon = 1e-12 ):
  
  print "epsilon", epsilon
  n,d = X.shape
  assert len(y) == n, "incorrect sizes"
  
  #train_folds, test_folds = xval_folds( n, k_fold, randomize = randomize, seed = seed )
  
  avg_projection = np.zeros( n, dtype=float )
  avg_probability = np.zeros( n, dtype=float )
  
  mean_projections = np.zeros( n, dtype=float )
  var_projections  = np.zeros( n, dtype=float )
  
  mean_probabilities = np.zeros( n, dtype=float )
  var_probabilities  = np.zeros( n, dtype=float )
  
  # for each fold, compute mean and variances
  w_mean = np.zeros( (n,d), dtype = float )
  w_var = np.zeros( (n,d), dtype = float )
  
  all_I = np.arange(n,dtype=int)
  
  for i in xrange(n):
    train_ids = np.setdiff1d( all_I, i )
    test_ids = [i]
    
    X_test = X[test_ids,:]
    #bootstrap_ids = bootstraps( train_ids, n_bootstraps )
    
    X_train = X[train_ids,:]
    y_train = y[train_ids]
    
    sk_lda = sk_LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    sk_lda.fit( X_train, y_train )
    #pdb.set_trace()
    sk_test_proj = np.squeeze(sk_lda.predict_log_proba( X_test ))[1]
    test_proj = sk_test_proj #lda.transform( X_test )
    test_prob = np.squeeze(sk_lda.predict_proba( X_test ))[1] # lda.predict( X_test )


    mean_projections[ i ]   += test_proj
    mean_probabilities[ i ] += test_prob
    
    var_projections[ i ]   += np.square( test_proj )
    var_probabilities[ i ] += np.square( test_prob )
    
    w = np.squeeze( sk_lda.coef_ )
    w_mean[i] += w
    w_var[i] += np.square(w)
  
  w_mn = w_mean.mean(0)
 
  w_var   = w_mean.var(0)
    
  #print "loo     w = ", w_mn
  #print "loo w_var = ", w_var
  
  var_projections   -= np.square( mean_projections )
  var_probabilities -= np.square( mean_probabilities )
  avg_projection=mean_projections
  avg_probability=mean_probabilities
  return (mean_projections,var_projections),(mean_probabilities,var_probabilities),(w_mn,w_var),(avg_projection,avg_probability)
        
def lda_on_train( X, y, k_fold = 10, n_bootstraps = 10, randomize = True, seed = 0, epsilon = 1e-12 ):
  
  n,d = X.shape
  assert len(y) == n, "incorrect sizes"
  
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
  
  lda = LinearDiscriminantAnalysis(epsilon=epsilon)
  lda.fit( X, y )
  w = lda.w_prop_to
  
  sk_lda = sk_LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
  sk_lda.fit( X, y )
  sk_test_proj = np.squeeze(sk_lda.predict_log_proba( X ))
  
  w = lda.w_prop_to
      
  
  test_proj = lda.transform( X )
      
  #pdb.set_trace()
  ranked = np.argsort(test_proj).astype(float) / len(test_proj)
  #test_proj = ranked
  test_prob = lda.prob( X )
  I=pp.find( np.isinf(test_prob) )
  test_prob[I] = 1
  test_predict = lda.predict( X )
  mean_projections = test_proj
  mean_probabilities = test_prob
  
  var_projections = np.square( test_proj )
  var_probabilities = np.square( test_predict )
  
  w_mean = w
  w_var = np.square(w)
      
  print "train w = ",  w
  I=pp.find( np.isinf(avg_probability) )
  avg_probability[I] = 1 
    
  return (mean_projections,var_projections),(mean_probabilities,var_probabilities),(w_mean,w_var),(avg_projection,avg_probability)

def run_survival_analysis( disease_list, fill_store, data_store, k_fold = 10, n_bootstraps = 10, epsilon = 1e-12 ):
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
  i_event = pp.find(predict_survival_train["E"].values)
  #median_time = np.median( predict_survival_train["T"].values[i_event] )
  median_time = np.mean( predict_survival_train["T"].values )
  i_less = pp.find(predict_survival_train["T"].values<median_time)
  
  #y = predict_survival_train["E"].values.astype(int)
  y = np.zeros( len(predict_survival_train["T"].values) )
  y[i_less] = 1
  projections, probabilties, weights, averages = lda_with_xval_and_bootstrap( X, y, k_fold = k_fold, n_bootstraps = n_bootstraps )
  
  return projections, probabilties, weights, averages, X, y, Events_train, Times_train

def run_survival_analysis_lda( disease_list, fill_store, data_store, k_fold = 10, n_bootstraps = 10, epsilon = 1e-12 ):
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
  i_event = pp.find(predict_survival_train["E"].values)
  #median_time = np.median( predict_survival_train["T"].values[i_event] )
  median_time = np.mean( predict_survival_train["T"].values )
  i_less = pp.find(predict_survival_train["T"].values<median_time)
  
  y = predict_survival_train["E"].values.astype(int)
  #y = np.zeros( len(predict_survival_train["T"].values) )
  #y[i_less] = 1
  projections, probabilties, weights, averages = lda_with_xval_and_bootstrap( X, y, k_fold = k_fold, n_bootstraps = n_bootstraps, epsilon=epsilon )
  
  return projections, probabilties, weights, averages, X, y, Events_train, Times_train

def run_survival_analysis_lda_loo( disease_list, fill_store, data_store, k_fold = 10, n_bootstraps = 10, epsilon = 1e-12 ):
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
  i_event = pp.find(predict_survival_train["E"].values)
  #median_time = np.median( predict_survival_train["T"].values[i_event] )
  median_time = np.mean( predict_survival_train["T"].values )
  i_less = pp.find(predict_survival_train["T"].values<median_time)
  
  y = predict_survival_train["E"].values.astype(int)
  #y = np.zeros( len(predict_survival_train["T"].values) )
  #y[i_less] = 1
  projections, probabilties, weights, averages = lda_with_loo( X, y, epsilon=epsilon )
  
  return projections, probabilties, weights, averages, X, y, Events_train, Times_train

def run_survival_prediction_loo( disease_list, fill_store, data_store, group0, group1, data_keys, data_names, C = 1 ):
  fill_store.open()
  data_store.open()
  ALL_SURVIVAL = data_store["/CLINICAL/data"][["patient.days_to_last_followup","patient.days_to_death"]]
  tissue_barcodes = np.array( ALL_SURVIVAL.index.tolist(), dtype=str )
  surv_barcodes = np.array([ x+"_"+y for x,y in tissue_barcodes])
  NEW_SURVIVAL = pd.DataFrame( ALL_SURVIVAL.values, index =surv_barcodes, columns = ALL_SURVIVAL.columns ) 
  val_survival  = pd.concat( [NEW_SURVIVAL, fill_store["/Z/VAL/Z/mu"]], axis=1, join = 'inner' )
  
  datas = []
  for data_key, data_name in zip( data_keys, data_names):
    datas.append( data_store[data_key].loc[val_survival.index].fillna(0) )
    data_columns = {}
    for b in data_store[data_key].columns:
      if len(data_keys)>1:
        data_columns[b] = "%s_%s"%(data_name,b)
      else:
        data_columns[b] = "%s"%(b)
    
    datas[-1].rename( columns = data_columns, inplace=True)
  
  data_train = pd.concat(datas, axis=1)  
  fill_store.close()
  data_store.close()
  
  #pdb.set_trace()
  X_columns = data_train.columns
  X = data_train[X_columns].values.astype(float)  
  y = np.zeros(len(X),dtype=int)
  y[group1] = 1
  #pdb.set_trace()
  predictions, probabilties, weights, averages = predict_groups_with_loo( X, y, C  )
  
  return predictions, probabilties, weights, averages, data_train, y

def run_survival_prediction_loo_regression( disease_list, fill_store, data_store, targets, data_keys, data_names, C = 1 ):
  fill_store.open()
  data_store.open()
  ALL_SURVIVAL = data_store["/CLINICAL/data"][["patient.days_to_last_followup","patient.days_to_death"]]
  tissue_barcodes = np.array( ALL_SURVIVAL.index.tolist(), dtype=str )
  surv_barcodes = np.array([ x+"_"+y for x,y in tissue_barcodes])
  NEW_SURVIVAL = pd.DataFrame( ALL_SURVIVAL.values, index =surv_barcodes, columns = ALL_SURVIVAL.columns ) 
  val_survival  = pd.concat( [NEW_SURVIVAL, fill_store["/Z/VAL/Z/mu"]], axis=1, join = 'inner' )
  
  datas = []
  for data_key, data_name in zip( data_keys, data_names):
    datas.append( data_store[data_key].loc[val_survival.index].fillna(0) )
    data_columns = {}
    for b in data_store[data_key].columns:
      if len(data_keys)>1:
        data_columns[b] = "%s_%s"%(data_name,b)
      else:
        data_columns[b] = "%s"%(b)
    
    datas[-1].rename( columns = data_columns, inplace=True)
  
  data_train = pd.concat(datas, axis=1)  
  fill_store.close()
  data_store.close()
  
  #pdb.set_trace()
  X_columns = data_train.columns
  X = data_train[X_columns].values.astype(float)  
  #y = np.zeros(len(X),dtype=int)
  #y[group1] = 1
  y=targets
  #pdb.set_trace()
  assert len(y) == len(X), "made different sizes"
  predictions, weights, averages = predict_groups_with_loo_with_regression( X, y, C  )
  #(mean_projections,var_projections),(w_mn,w_var),(avg_projection,)
  return predictions, weights, averages, data_train, y

    
def run_survival_analysis_lda_train( disease_list, fill_store, data_store, k_fold = 10, n_bootstraps = 10, epsilon = 1e-12 ):
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
  i_event = pp.find(predict_survival_train["E"].values)
  #median_time = np.median( predict_survival_train["T"].values[i_event] )
  median_time = np.mean( predict_survival_train["T"].values )
  i_less = pp.find(predict_survival_train["T"].values<median_time)
  
  y = predict_survival_train["E"].values.astype(int)
  #y = np.zeros( len(predict_survival_train["T"].values) )
  #y[i_less] = 1
  projections, probabilties, weights, averages = lda_on_train( X, y, k_fold = k_fold, n_bootstraps = n_bootstraps, epsilon=epsilon )
  
  return projections, probabilties, weights, averages, X, y, Events_train, Times_train
  
def run_survival_analysis_kmeans( disease_list, fill_store, data_store, k_fold, K ):
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
  i_event = pp.find(predict_survival_train["E"].values)
  #median_time = np.median( predict_survival_train["T"].values[i_event] )
  median_time = np.mean( predict_survival_train["T"].values )
  i_less = pp.find(predict_survival_train["T"].values<median_time)
  
  y = predict_survival_train["E"].values.astype(int)
  #y = np.zeros( len(predict_survival_train["T"].values) )
  #y[i_less] = 1
  #projections, probabilties, weights, averages = lda_with_xval_and_bootstrap( X, y, k_fold = k_fold, n_bootstraps = n_bootstraps )
  predictions = kmeans_survival( X, y, K = K )
  
  return predictions, X, y, Events_train, Times_train
    
if __name__ == "__main__":
  
  disease = "blca"
  data_file = "pan_tiny_multi_set"
  experiment_name = "tiny_leave_%s_out"%(disease)
  
  if len(sys.argv) == 4:
    disease   = sys.argv[1]
    data_file = sys.argv[2]
    #experiment_name = sys.argv[3]
    
    data_location = os.path.join( HOME_DIR, "data/broad_processed_post_recomb/20160128/%s/data.h5"%(data_file) )
    fill_location = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out/medium/leave_out_%s/full_vae_fill.h5"%(disease) )
    survival_location = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out/medium/leave_out_%s/full_vae_survival.h5"%(disease) )
    savename = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out/medium/leave_out_%s/survival_xval.png"%(disease))
  else:
    data_location = os.path.join( HOME_DIR, "data/broad_processed_post_recomb/20160128/%s/data.h5"%(data_file) )
    fill_location = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out_sandbox/tiny/leave_out_%s/full_vae_fill.h5"%(disease) )
    survival_location = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out_sandbox/tiny/leave_out_%s/full_vae_survival.h5"%(disease) )
    savename = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out/tiny/leave_out_%s/survival_xval.png"%(disease))
    
  s=pd.HDFStore( survival_location, "r" )
  d=pd.HDFStore( data_location, "r" )
  f=pd.HDFStore( fill_location, "r" ) 
  
  projections, probabilties, weights, averages, X, y, E_train, T_train = run_survival_analysis( [disease], f, d, k_fold = 20, n_bootstraps = 10, epsilon= 0.1 )  
  
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
  ax2.plot( mn_w, 'o-')
  
  #I = np.argsort( mn_prob )
  I1 = pp.find( mn_prob > np.median(mn_prob) )
  I0 = pp.find( mn_prob <= np.median(mn_prob) )
  #I1 = pp.find( avg_prob > np.median(avg_prob) )
  #I0 = pp.find( avg_prob <= np.median(avg_prob) )
  
  f = pp.figure()
  ax3 = f.add_subplot(111)
  
  kmf = KaplanMeierFitter()
  if len(I1) > 0:
    kmf.fit(T_train[I1], event_observed=E_train[I1], label =  "lda_1 E=%d C=%d"%(E_train[I1].sum(),len(I1)-E_train[I1].sum()))
    ax3=kmf.plot(ax=ax3,at_risk_counts=False,show_censors=True, color='red')
  if len(I0) > 0:
    kmf.fit(T_train[I0], event_observed=E_train[I0], label = "lda_0 E=%d C=%d"%(E_train[I0].sum(),len(I0)-E_train[I0].sum()))
    ax3=kmf.plot(ax=ax3,at_risk_counts=False,show_censors=True, color='blue')
    
  
  
  
  pp.savefig(savename, dpi=300, format='png')
  print "ROC mn_prob ", roc_auc_score(y,mn_prob)
  print "ROC avg_prob ", roc_auc_score(y,avg_prob)
  pp.show()
   