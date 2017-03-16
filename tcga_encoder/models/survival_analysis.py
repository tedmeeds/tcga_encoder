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
import torch
from torch.autograd import Variable
from tcga_encoder.models.pytorch.weibull_survival import WeibullSurvivalModel,WeibullSurvivalModelNeuralNetwork
from tcga_encoder.models.pytorch.lasso_regression import PytorchLasso
from tcga_encoder.models.pytorch.bootstrap_linear_regression import BootstrapLinearRegression, BootstrapLassoRegression
from tcga_encoder.models.pytorch.dropout_linear_regression import DropoutLinearRegression
#from tcga_encoder.models.pytorch.lasso_regression_gprior_ard import PytorchLasso
#import autograd.numpy as np
#from autograd import grad

#def cost_lineat_reg_ard( X, y, w, b, h );
def linear_reg_gprior_ard( X, y, alpha, lr = 1e-4, iters = 10, verbose = False, eps = 1e-6, w_init = None ):
  n,d = X.shape
  
  XTX = np.dot( X.T, X )
  XXT = np.dot( X, X.T )
  
  XX = X*X
  sX = XX.sum(0)
  if w_init is not None:
    w = w_init
  else:
    w = np.zeros(d,dtype=float)
  h = np.zeros(d,dtype=float)
  z = np.exp(h)
  b = np.mean(y)
  old_cost = np.inf
  for i in range(iters):
    y_hat = np.dot( X, w ) + b
    cost = np.sum( np.square( y - y_hat ) )
    
    # if np.abs(old_cost - cost) < 0.001:
    #   print "stopping at ", i, old_cost, cost
    #   break
    old_cost = cost
    if verbose:
      print "Error = ", np.sum( np.square( y - y_hat ) ), w[:5], b, z[:5]
    g_w = - np.dot( X.T, (y-y_hat) )/n + (z+eps)*np.sign(w)/d
    g_b = np.sum(y-np.dot( X, w ))
    ##g_h = np.abs(w)*z - alpha*np.dot( X.T, np.dot( X, (z+eps)**-2 ))*z
    g_h = np.abs(w)*z/d - np.dot( sX.T, alpha*z*(z+eps)**-2 )/d
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
    old_w = cost
    w = w - lr*g_w
    b = b - lr*g_b
    z = np.exp(h)
    
    # dif_w = np.linalg.norm( w-old_w )
    #
    # if i > 10 and dif_w < 1e-3:
    #   print "stopping at ", i, dif_w
    #   break
    if np.any(np.isnan(w)):
      pdb.set_trace()
      
  
  y_hat = np.dot( X, w ) + b
  if verbose:
   print "Final Error = ", np.sum( np.square( y - y_hat ) )
  #pdb.set_trace()
  return w, b
    
def make_bootstraps( x, m ):
  # samples from arange(n) with replacement, m times.
  #x = np.arange(n, dtype=int)
  n = len(x)
  N = np.zeros( (m,n), dtype=int)
  for i in range(m):
    N[i,:] = sklearn.utils.resample( x, replace = True )
    
  return N
  
def xval_folds( n, K, randomize = False, seed = None ):
  if randomize is True:
    print("XVAL RANDOMLY PERMUTING")
    if seed is not None:
      print( "XVAL SETTING SEED = %d"%(seed) )
      np.random.seed(seed)
      
    x = np.random.permutation(n)
  else:
    print( "XVAL JUST IN ARANGE ORDER")
    x = np.arange(n,dtype=int)
    
  kf = KFold( K )
  train = []
  test = []
  for train_ids, test_ids in kf.split( x ):
    #train_ids = np.setdiff1d( x, test_ids )
    
    train.append( x[train_ids] )
    test.append( x[test_ids] )
  #pdb.set_trace()
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

def pytorch_survival_train_val( train, val, l1=0, n_epochs=1000 ):
  Z_train = train[0]
  T_train = train[1]
  E_train = train[2]
  
  Z_val = val[0]
  T_val = val[1]
  E_val = val[2]
  
  Z_val -= Z_train.mean(0)
  Z_val /= Z_train.std(0)
  
  Z_train -= Z_train.mean(0)
  Z_train /= Z_train.std(0)
  #print "epsilon", epsilon
  n,dim = Z_val.shape
  assert len(T_val) == n, "incorrect sizes"
  assert len(E_val) == n, "incorrect sizes"
  
  avg_projection = np.zeros( n, dtype=float )
  avg_probability = np.zeros( n, dtype=float )
  
  mean_projections = np.zeros( n, dtype=float )
  var_projections  = np.zeros( n, dtype=float )
  
  mean_probabilities = np.zeros( n, dtype=float )
  var_probabilities  = np.zeros( n, dtype=float )
  K = 10
  # for each fold, compute mean and variances
  w_mean = np.zeros( dim, dtype = float )
  w_var = np.zeros( dim, dtype = float )
   
  Z_test_py = Variable( torch.FloatTensor( Z_val ) )
  T_test_py = Variable( torch.FloatTensor( T_val ) )
  E_test_py = Variable( torch.FloatTensor( E_val ) )
  
  Z_train_py = Variable( torch.FloatTensor( Z_train ) )
  E_train_py = Variable( torch.FloatTensor( E_train ) )
  T_train_py = Variable( torch.FloatTensor( T_train ) )
  
  model =  WeibullSurvivalModel( dim )
  #model =  WeibullSurvivalModelNeuralNetwork( dim, K )
  model.add_test(E_test_py,T_test_py,Z_test_py)
  #model.fit( E_train, T_train, Z_train, lr = 1e-3, logging_frequency = 2000, l1 = l1, n_epochs = n_epochs, normalize=False )
  model.fit( E_train, T_train, Z_train, lr = 1e-3, logging_frequency = 2000, l1 = l1, n_epochs = n_epochs, normalize=False )
  
  w = model.w.data.numpy().flatten() #beta.data.numpy()

  #pdb.set_trace()
  test_proj = np.squeeze( model.LogTime( Z_test_py, at_time=0.5 ).data.numpy() )
  
  time_proj = np.exp( test_proj )
  
  T_test_proj = Variable( torch.FloatTensor( time_proj ) )
  #
  # S_test_proj = np.squeeze(model.Survival( T_test_proj, Z_test ).data.numpy())
  # S_test      = np.squeeze(model.Survival( T_test, Z_test ).data.numpy())

  #
  # f = pp.figure()
  # ax1 = f.add_subplot(121)
  # kmf = KaplanMeierFitter()
  # kmf.fit(T_train.data.numpy(), event_observed=E_train.data.numpy(), label =  "train" )
  # ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='blue')
  # kmf.fit(T_test.data.numpy(), event_observed=E_test.data.numpy(), label =  "test" )
  # ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='red')
  # model.PlotSurvival( E_train, T_train, Z_train, ax=ax1, color = "b" )
  # ax=model.PlotSurvival( E_test, T_test, Z_test, ax=ax1, color = "r" )
  #
  # ax1 = f.add_subplot(122)
  # kmf = KaplanMeierFitter()
  # kmf.fit(T_val.data.numpy(), event_observed=E_val.data.numpy(), label =  "train" )
  # ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='blue')
  # kmf.fit(T_test.data.numpy(), event_observed=E_test.data.numpy(), label =  "test" )
  # ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='red')
  # model.PlotSurvival( E_train, T_train, Z_train, ax=ax1, color = "b" )
  # ax=model.PlotSurvival( E_test, T_test, Z_test, ax=ax1, color = "r" )
  #ax.vlines(time_proj,0,1)
  #ax.plot( np.vstack( (T_test.data.numpy(), time_proj) ), np.vstack( (S_test, S_test_proj) ), 'm-')
  #pp.title("TRAIN")

  #pp.show()
  #pdb.set_trace()
  #pp.close('all')
  test_prob = model.LogLikelihood( E_test_py, T_test_py, Z_test_py ).data.numpy()
  #pdb.set_trace()  
  mean_projections   += test_proj
  mean_probabilities += test_prob
  
  var_projections   += np.square( test_proj )
  var_probabilities += np.square( test_prob )
  
  w_mean += w
  w_var += np.square(w)
  w_mn = w_mean 

  #I=pp.find( np.isinf(avg_probability) )
  #avg_probability[I] = 1 
    
  w_var   -= np.square( w_mean )
  
  var_projections   -= np.square( mean_projections )
  var_probabilities -= np.square( mean_probabilities )
  
  return (mean_projections,var_projections),(mean_probabilities,var_probabilities),(w_mean,w_var),(avg_projection,avg_probability)
  
  
def pytorch_survival_xval( E, T, Z_orig, k_fold = 10, n_bootstraps = 10, randomize = True, seed = 0, l1 = 0.0, n_epochs = 1000, normalize = False ):
  
  #print "epsilon", epsilon
  n,dim = Z_orig.shape
  assert len(T) == n, "incorrect sizes"
  assert len(E) == n, "incorrect sizes"
  
  train_folds, test_folds = xval_folds( n, k_fold, randomize = True, seed=0 )
  
  avg_projection = np.zeros( n, dtype=float )
  avg_probability = np.zeros( n, dtype=float )
  
  mean_projections = np.zeros( n, dtype=float )
  var_projections  = np.zeros( n, dtype=float )
  
  mean_probabilities = np.zeros( n, dtype=float )
  var_probabilities  = np.zeros( n, dtype=float )
  K = 10
  # for each fold, compute mean and variances
  w_mean = np.zeros( (k_fold,dim), dtype = float )
  w_var = np.zeros( (k_fold,dim), dtype = float )
  
  for k, train_ids, test_ids in zip( range(k_fold), train_folds, test_folds ):
    Z = Z_orig.copy()
    
    mn_z = Z[train_ids,:].mean(0)
    std_z = Z[train_ids,:].std(0)
    if normalize is True:
      print( "normalizing" )
      Z -= mn_z
      Z /= std_z
      
    Z_test = Variable( torch.FloatTensor( Z[test_ids,:] ) )
    T_test = Variable( torch.FloatTensor( T[test_ids] ) )
    E_test = Variable( torch.FloatTensor( E[test_ids] ) )
    
    Z_train = Variable( torch.FloatTensor( Z[train_ids,:] ) )
    E_train = Variable( torch.FloatTensor( E[train_ids] ) )
    T_train = Variable( torch.FloatTensor( T[train_ids] ) )
    
    #pdb.set_trace()
    
    Z_train_val = Z[train_ids,:]
    T_train_val = T[train_ids]
    E_train_val = E[train_ids]
    
    mean_E_train = E_train_val.sum()
    mean_E_test  = E[test_ids].sum()
    print("events train %d  events test %d"%(mean_E_train,mean_E_test))
    
    #pdb.set_trace()
    model =  WeibullSurvivalModel( dim )
    #model =  WeibullSurvivalModelNeuralNetwork( dim, K )
    model.add_test(E_test,T_test,Z_test)
    #model.fit( E_train, T_train, Z_train, lr = 1e-3, logging_frequency = 2000, l1 = l1, n_epochs = n_epochs, normalize=False )
    model.fit( E_train_val, T_train_val, Z_train_val, lr = 1e-3, logging_frequency = 2000, l1 = l1, n_epochs = n_epochs, normalize=False )
    
    w = model.w.data.numpy().flatten() #beta.data.numpy()

    #pdb.set_trace()
    test_proj = np.squeeze( model.LogTime( Z_test, at_time=0.5 ).data.numpy() )
    
    time_proj = np.exp( test_proj )
    
    T_test_proj = Variable( torch.FloatTensor( time_proj ) )

    S_test_proj = np.squeeze(model.Survival( T_test_proj, Z_test ).data.numpy())
    S_test      = np.squeeze(model.Survival( T_test, Z_test ).data.numpy())

    #test_proj /= 365.0
    #test_proj = np.log(test_proj)
    #test_proj -= np.median( test_proj )
    # pp.figure()
    #
    f = pp.figure()
    ax1 = f.add_subplot(111)
    kmf = KaplanMeierFitter()
    kmf.fit(T_train.data.numpy(), event_observed=E_train.data.numpy(), label =  "train" )
    ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='blue')
    kmf.fit(T_test.data.numpy(), event_observed=E_test.data.numpy(), label =  "test" )
    ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='red')
    model.PlotSurvival( E_train, T_train, Z_train, ax=ax1, color = "b" )
    ax=model.PlotSurvival( E_test, T_test, Z_test, ax=ax1, color = "r" )
    #ax.vlines(time_proj,0,1)
    #ax.plot( np.vstack( (T_test.data.numpy(), time_proj) ), np.vstack( (S_test, S_test_proj) ), 'm-')
    pp.title("TRAIN")

    #pp.show()
    #pdb.set_trace()
    #pp.close('all')
    test_prob = model.LogLikelihood( E_test, T_test, Z_test ).data.numpy()
    #pdb.set_trace()  
    mean_projections[ test_ids ]   += test_proj
    mean_probabilities[ test_ids ] += test_prob
    
    var_projections[ test_ids ]   += np.square( test_proj )
    var_probabilities[ test_ids ] += np.square( test_prob )
    
    w_mean[k] += w
    w_var[k] += np.square(w)
    w_mn = w_mean[k] / n_bootstraps

  #I=pp.find( np.isinf(avg_probability) )
  #avg_probability[I] = 1 
    
  w_var   -= np.square( w_mean )
  
  var_projections   -= np.square( mean_projections )
  var_probabilities -= np.square( mean_probabilities )
  
  return (mean_projections,var_projections),(mean_probabilities,var_probabilities),(w_mean,w_var),(avg_projection,avg_probability)

def predict_groups_with_xval_with_regression( X_orig, y_orig, l1, k_fold=10, randomize = True, seed = 0, use_cuda=False ):
  
  
  #print "epsilon", epsilon
  n,d = X_orig.shape
  assert len(y_orig) == n, "incorrect sizes"
  
  train_folds, test_folds = xval_folds( n, k_fold, randomize = randomize, seed = seed )
  
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
  
  Ws = np.zeros( (n,d) )
  bs = np.zeros( (n,) )
  for k, train_ids, test_ids in zip( range(k_fold), train_folds, test_folds ):
    X = X_orig.copy()
    y = y_orig.copy()
    #y -= y.mean()
    #X -= X.mean(0)
    #X /= X.std(0)
    X_test_val = X[test_ids,:]
    y_test_val = y[test_ids]
    
    X_train = X[train_ids,:]
    y_train = y[train_ids]
    
    #y_train -= np.median(y[train_ids])
    #y_test_val -= np.median(y[train_ids])
    # X_train = Variable( torch.FloatTensor( X_train ) )
    if use_cuda is True:
      X_test = Variable( torch.FloatTensor( X_test_val ) ).cuda()
      y_test = Variable( torch.FloatTensor( y_test_val ) ).cuda()
    else:
      X_test = Variable( torch.FloatTensor( X_test_val ) )
      y_test = Variable( torch.FloatTensor( y_test_val ) )
    
    
    #penalty="l2"
    #model = BootstrapLinearRegression( d, l1 )
    #model = BootstrapLassoRegression( d, l1 )
    model = DropoutLinearRegression( d, use_cuda )
    #pdb.set_trace()
    model.add_test( X_test, y_test )
    model.fit( X_train, y_train, \
               n_epochs=10000, \
               min_epochs = 2000, \
               logging_frequency = 500, \
               testing_frequency = 100, \
               lr=0.005, l1=l1 ,l2=0.00 ) #n_epochs=2000, lr = 0.01, logging_frequency = 500 )
    #sk_lda = sklearn.linear_model.Lasso(alpha=l1, fit_intercept=True, normalize=True)
    #sk_lda.fit( X_train, y_train )
    
    #w_ard, b_ard = linear_reg_gprior_ard( X_train, y_train, C, lr = 0.001, iters=1500, verbose = False )
    
    
    #sk_test_proj2 = model.predict( X_test ) #np.squeeze(model.predict( X_test ).data.numpy())
    #sk_test_proj = np.dot( X_test, w_ard ) + b_ard
    test_proj = np.squeeze( model.predict( X_test ) )
    #pdb.set_trace()

   
    
    #w_est_linear = np.dot( np.linalg.inv( np.dot(X_train.T,X_train) ), np.dot( X_train.T, y_train ) )
    
    #pdb.set_trace()
    #y_est_linear = np.dot( X_test_val, w_est_linear)
    w = np.squeeze( model.get_w() ) #.data.numpy() )
    #y_est_model = np.dot( X_test_val, w)+model.get_b()
    #test_proj = y_est_model
    #pdb.set_trace()
    mean_projections[ test_ids ]   += test_proj
    var_projections[ test_ids ]   += np.square( test_proj )
    
    #w = w_ard #
    
    #pdb.set_trace()
    Ws[test_ids,:] = w
    bs[test_ids] = model.get_b() #sk_lda.intercept_
    #bs[test_ids] = model.bias.data.numpy()
    w_mean[test_ids] += w
    w_var[test_ids] += np.square(w)
  
  #Ws = np.array(Ws)
  #bs = np.array(bs)
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
    
    w_ard, b_ard = linear_reg_gprior_ard( X_train, y_train, C, lr = 0.001, iters=1500, verbose = False, w_init = np.squeeze( sk_lda.coef_ ) )
    
    
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

def run_pytorch_survival_train_val( train_survival, val_survival, l1= 0.0, n_epochs = 1000 ):
  # projections, \
  # probabilties, \
  # weights, averages, X, y, E_train, T_train =
  
  X_columns = val_survival.columns[2:]
  
  X_train = train_survival[X_columns].values.astype(float)
  X_val   = val_survival[X_columns].values.astype(float)
  
  i_event = pp.find(train_survival["E"].values)
  E_train = train_survival["E"].values.astype(int)
  
  i_event = pp.find(val_survival["E"].values)
  E_val = val_survival["E"].values.astype(int)
  
  #E = predict_survival_train["E"].values
  T_train = np.maximum( 1, train_survival["T"].values )
  T_val   = np.maximum( 1, val_survival["T"].values )
  
  train = [X_train, T_train, E_train]
  val = [X_val, T_val, E_val]
  projections, probabilties, weights, averages = pytorch_survival_train_val( train, val, l1=l1, n_epochs=n_epochs )
  
  return projections, probabilties, weights, averages, X_val, E_val, E_train, T_train

def run_pytorch_survival_folds( disease_list, fill_store, data_store, \
                                k_fold = 10, \
                                n_bootstraps = 10, \
                                l1 = 0.0, \
                                n_epochs=1000, \
                                normalize = False, seed = 0):
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
  
  
  E = predict_survival_train["E"].values
  T = np.maximum( 1, predict_survival_train["T"].values )
  Z = X
  projections, probabilties, weights, averages = pytorch_survival_xval( E, T, Z, k_fold, l1=l1, n_epochs=n_epochs, normalize=normalize, seed=seed )
  
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

def run_survival_prediction_xval_regression( disease_list, \
               fill_store, data_store, targets, data_keys, \
               data_names, l1 = 0.0, k_fold = 10, seed=0, use_cuda = False ):
  fill_store.open()
  data_store.open()
  ALL_SURVIVAL = data_store["/CLINICAL/data"][["patient.days_to_last_followup","patient.days_to_death"]]
  tissue_barcodes = np.array( ALL_SURVIVAL.index.tolist(), dtype=str )
  surv_barcodes = np.array([ x+"_"+y for x,y in tissue_barcodes])
  NEW_SURVIVAL = pd.DataFrame( ALL_SURVIVAL.values, index =surv_barcodes, columns = ALL_SURVIVAL.columns ) 
  val_survival  = pd.concat( [NEW_SURVIVAL, fill_store["/Z/VAL/Z/mu"]], axis=1, join = 'inner' )
  fill_type = "VAL"
  na_datas = []
  datas = []
  for data_key, data_name in zip( data_keys, data_names):
    all_bad = False
    try:
      na_datas.append( data_store[data_key].loc[val_survival.index])
      datas.append( data_store[data_key].loc[val_survival.index].fillna(0) )
    
      bad_ids = pp.find( pp.isnan(na_datas[-1].values.sum(1)))
      
    except:
      
      all_bad = True
      bad_ids=[x,x,x,x]

    
    if len(bad_ids) > 0 or all_bad is True:
      if all_bad is True:
        bad_bcs = val_survival.index
      else:
        bad_bcs = na_datas[-1].index.values[bad_ids]
      
      data_type = data_key.split("/")[1]
      key = "/Fill/%s/%s"%(fill_type,data_type)
      #pdb.set_trace()
      if key in fill_store:
        x_fill = fill_store[key].loc[bad_bcs]
        if all_bad is False:
          XX = na_datas[-1].values
          XX[bad_ids,:] = x_fill.values
          datas[-1] = pd.DataFrame( XX, columns = na_datas[-1].columns, index=na_datas[-1].index )
        else:
          XX = x_fill.values
          datas.append( pd.DataFrame( XX, columns = x_fill.columns, index=x_fill.index ) )
        
        
        
        
      else:
        print "skipping filling in %s for ids "%(data_key), bad_bcs
        #pdb.set_trace()
    
      
      #pdb.set_trace()
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
  predictions, weights, averages = predict_groups_with_xval_with_regression( X, y, l1, k_fold=k_fold, use_cuda=use_cuda  )
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
   