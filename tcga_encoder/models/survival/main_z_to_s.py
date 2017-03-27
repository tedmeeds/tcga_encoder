from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
from tcga_encoder.models.survival_analysis import *
#from tcga_encoder.algorithms import *
import seaborn as sns
import pdb
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from lifelines.statistics import logrank_test
sns.set_style("whitegrid")
sns.set_context("talk")
from tcga_encoder.models.pytorch.bootstrap_linear_regression import BootstrapLinearRegression, BootstrapLassoRegression
from scipy import stats
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# events_plot( val_proj, val, ax=ax1, name = "validation" )
def events_plot( projection, data, logging_dict, disease, ax = None, name = "", rgs=None ):
  if ax is None:
    f = pp.figure()
    ax = f.add_subplot(111)
    
  X = data[0]
  T = data[1]
  E = data[2]
  I = np.argsort(-projection)
  y = E
  #I = np.argsort(-mn_prob)
  third = int(len(I)/3.0)
  half = int(len(I)/2.0)
  
  if rgs is None:
    I0 = I[:half]
    I1 = []#I[third:2*third]
    I2 = I[half:]
  else:
    I0 = pp.find( projection > rgs[0][0])
    I2 = pp.find( projection < rgs[2][1])
    I1 = []#pp.find( (projection > rgs[2][1])  * (projection < rgs[0][0]))
    #pdb.set_trace()
  # I0 = I[:half]
  # I1 = [] #I[third:2*third]
  # I2 = I[half:]
  kmf = KaplanMeierFitter()
  if len(I2) > 0:
    kmf.fit(T[I2], event_observed=E[I2], label =  "E=%d C=%d"%(E[I2].sum(),len(I2)-E[I2].sum()))
    ax=kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color='red')
  if len(I1) > 0:
    kmf.fit(T[I1], event_observed=E[I1], label =  "E=%d C=%d"%(E[I1].sum(),len(I1)-E[I1].sum()))
    ax=kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color='green')
  if len(I0) > 0:
    kmf.fit(T[I0], event_observed=E[I0], label = "E=%d C=%d"%(E[I0].sum(),len(I0)-E[I0].sum()))
    ax=kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color='blue')
  if len(I0) == 0:
    results = logrank_test(T[I1], T[I2], event_observed_A=E[I1], event_observed_B=E[I2])
  elif len(I2) == 0:
    results = logrank_test(T[I0], T[I1], event_observed_A=E[I0], event_observed_B=E[I1])
  else:
    results = logrank_test(T[I0], T[I2], event_observed_A=E[I0], event_observed_B=E[I2])
  pp.title("%s %s Log-rank Test: %0.1f"%(name, disease, results.test_statistic))
  save_location = os.path.join( logging_dict[SAVEDIR], "survival_pytorch_%s.png"%(name) )
  pp.savefig(save_location, dpi=300, format='png')
  #print "ROC mn_prob ", roc_auc_score(y,mn_prob)
  
  print "ROC mn_proj ", roc_auc_score(y,projection)


  print "LOG RANK TEST: ", results.test_statistic
  
  #pdb.set_trace()
  return roc_auc_score(y,projection), results, (I0,I1,I2)
  
  
def symmetric_kl( m1,s1,m2,s2, sum_axis = None):
  return 0.5*(kl_divergence( m1,s1,m2,s2,sum_axis=sum_axis)+kl_divergence( m2,s2,m1,s1,sum_axis=sum_axis))

#def symmetric_kl( m1,s1,m2,s2, sum_axis = None):
#  return 0.5*(kl_divergence( m1,s1,m2,s2,sum_axis=sum_axis)+kl_divergence( m2,s2,m1,s1,sum_axis=sum_axis))

  
def kl_divergence( m1,s1,m2,s2, sum_axis = None):
  # p: m1,s1
  # q: m2, s2
  # returns KL(p,q)
  v1 = s1*s1+1e-6
  v2 = s2*s2+1e-6
  
  log_s1 = np.log(s1+1e-6)
  log_s2 = np.log(s2+1e-6)
  
  kl_p_q = log_s2-log_s1 - 0.5 + (v1+np.square(m1-m2))/(2*v2)   
      
  if sum_axis is not None:
    kl_p_q = kl_p_q.sum(axis=sum_axis)
  return kl_p_q
  
def plot_data_with_importance( w, importance, i_importance, name, save_location, n = 20 ):
  ff=pp.figure(figsize=(16,6)); ax1 = ff.add_subplot(111); 
  n=len(w.T)
  sns.heatmap( w.T, linewidths=0.2, xticklabels=False, cbar=False  ); pp.yticks( rotation=0, size=10); pp.title(name)
  ax2 = ff.add_subplot(1,10,10); ax2.plot( importance[i_importance[:n]], n-0.5-np.arange(n), 'k-o',lw=2,alpha=0.5 ); 
  #ax2 = ff.add_subplot(1,10,10); ax2.plot( importance[i_importance[], 'k-o',lw=2,alpha=0.5 ); 
  
  pp.savefig( save_location + "%s.png"%name, fmt='png', dpi=300)
 
def importance_calc( w, b, X_orig, y_true, normalize = False ):
   X = X_orig.copy();
   mn_x = X_orig.mean(0)
   std_x = X_orig.std(0)
   if normalize is True:
     X -= mn_x; X /= std_x
   y_pred_all = np.sum( w*X, 1) + b 
   E_all = np.mean( np.square( y_true - y_pred_all ) )
   #E_all = np.corrcoef(y_true, y_pred_all)[0][1]
   
   n,d = w.shape
   E_rest = np.zeros(d)
   for i in range(d):
     w_i = w.copy(); w_i[:,i] =  0 #w[:,i].mean()
     
     y_pred = np.sum( w_i*X, 1) + b + mn_x[i]*w[:,i].mean()
     E_rest[i] = np.mean( np.square( y_true - y_pred) )
     
     #if i == 28:
     #   pdb.set_trace()
     #E_rest[i] = np.corrcoef(y_true, y_pred)[0][1]
   
   #pdb.set_trace()
   return (E_rest - E_all) #/mn_x #np.maximum(0,E_rest-E_all)
   
   
def importance_calc_old( w, X, E ):
  ww = w*w
  XX = X*X
  total_E = np.sum( np.square(E) )
  importance = []
  n,nw = w.shape
  F = np.zeros( (n,nw) )
  for j in range(nw):
    f = 0
    for ii in range(n):
      e_i_j = w[ii,j]*X[ii,j]
      E_i = E[ii]
      F[ii,j] = np.abs(-e_i_j*(2*E[ii] - e_i_j))
      #F[ii,j]= ww[ii,j]*XX[ii,j] - 2*w[ii,j]*E[ii]*X[ii,j]
    #importance.append(f)
  importance = np.array(F)
  #pdb.set_trace()
  #i = XX*ww
  #importance = np.mean( i , 0 )
  #importance = np.abs( np.mean( w,0 ) ) / np.var( w,0 )
  return importance

######################################################################################################
def filter_diseases( train_survival, val_survival, filters ):
  if len(filters) == 0:
    return train_survival, val_survival
    
  diseases_a = np.array( [d_bc.split('_')[0] for d_bc in train_survival.index.values], dtype=str) 
  diseases_b = np.array( [d_bc.split('_')[0] for d_bc in val_survival.index.values], dtype=str) 

  query_a = np.zeros( len(train_survival), dtype=bool )
  query_b = np.zeros( len(val_survival), dtype=bool )
  for f in filters:
    I_a = diseases_a == f
    I_b = diseases_b == f
    
    query_a |= I_a
    query_b |= I_b
    
  
  train_survival = train_survival[query_a]
  val_survival = val_survival[query_b]
  
  return train_survival, val_survival
    
def add_diseases( train_survival, val_survival, tissue_store ):
  
  bcs_a = train_survival.index
  bcs_b = val_survival.index
  
  tissues_a = tissue_store.loc[ bcs_a ].fillna(0)
  tissues_b = tissue_store.loc[ bcs_b ].fillna(0)
  
  #pdb.set_trace()
  return pd.concat( [train_survival, tissues_a], axis=1), pd.concat( [val_survival, tissues_b], axis=1)
      
def get_data( fill_store, data_store, filters = [], add_tissue = False ):
  fill_store.open()
  data_store.open()
  ALL_SURVIVAL = data_store["/CLINICAL/data"][["patient.days_to_last_followup","patient.days_to_death"]]
  tissue_barcodes = np.array( ALL_SURVIVAL.index.tolist(), dtype=str )
  surv_barcodes = np.array([ x+"_"+y for x,y in tissue_barcodes])
  NEW_SURVIVAL = pd.DataFrame( ALL_SURVIVAL.values, index =surv_barcodes, columns = ALL_SURVIVAL.columns ) 
  
  val_survival  = NEW_SURVIVAL.join( fill_store["/Z/VAL/Z/mu"], how='inner' ).rename(columns={"patient.days_to_last_followup":"T","patient.days_to_death":"E"} )
  
  #pdb.set_trace()
  train_survival  = NEW_SURVIVAL.join( fill_store["/Z/TRAIN/Z/mu"], how='inner' ).rename(columns={"patient.days_to_last_followup":"T","patient.days_to_death":"E"} ) #pd.concat( [NEW_SURVIVAL, fill_store["/Z/BATCH/Z/mu"]], axis=1, join = 'inner' )
  
  
  Times_train = train_survival[ "T" ].fillna(0).values.astype(int)+train_survival[ "E" ].fillna(0).values.astype(int)
  Times_val = val_survival[ "T" ].fillna(0).values.astype(int)+val_survival[ "E" ].fillna(0).values.astype(int)
  
  Events_train = (1-np.isnan( train_survival[ "E" ].astype(float)) ).astype(int)
  Events_val = (1-np.isnan( val_survival[ "E" ].astype(float)) ).astype(int)
  
  
  val_survival["E"] = Events_val
  val_survival["T"] = Times_val
  
  train_survival["E"] = Events_train
  train_survival["T"] = Times_train
  
  train_survival, val_survival = filter_diseases( train_survival, val_survival, filters )
  
  if add_tissue is True:
    train_survival, val_survival = add_diseases( train_survival, val_survival, data_store['/CLINICAL/TISSUE'] )
  
  all_tissues = data_store['/CLINICAL/TISSUE']
  
  tissue_train = all_tissues.loc[train_survival.index]
  tissue_val = all_tissues.loc[val_survival.index]
  fill_store.close()
  data_store.close()
  
  return train_survival, tissue_train, val_survival, tissue_val 


if __name__ == "__main__":
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_file = sys.argv[1]
  print "Running: ",yaml_file  
  weights_matrix = []
  y = load_yaml( yaml_file)
  load_data_from_dict( y[DATA] )
  data_dict      = y[DATA] #{N_TRAIN:4000}
  survival_dict  = y["survival"]
  logging_dict   = y[LOGGING]
  
  logging_dict[SAVEDIR] = os.path.join( HOME_DIR, os.path.join( logging_dict[LOCATION], logging_dict[EXPERIMENT] ) )

  
  #data_location = os.path.join( HOME_DIR, "data/broad_processed_post_recomb/20160128/%s/data.h5"%(data_file) )
  fill_location = os.path.join( logging_dict[SAVEDIR], "full_vae_fill.h5" )
  survival_location = os.path.join( logging_dict[SAVEDIR], "full_vae_survival.h5" )
  #savename = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/leave_out/tiny/leave_out_%s/survival_xval.png"%(disease))
  
  print "FILL: ", fill_location
  print "SURV: ", survival_location
  s=pd.HDFStore( survival_location, "r" )
  d=data_dict['store'] #pd.HDFStore( data_location, "r" )
  f=pd.HDFStore( fill_location, "r" ) 
  
  #pdb.set_trace()
  
  
  for survival_spec in survival_dict:
    name = survival_spec["name"]
    print "running run_pytorch_survival_folds ,", data_dict['validation_tissues']
    
    #folds = survival_spec["folds"]
    bootstraps = survival_spec["bootstraps"]
    epsilon =  survival_spec["epsilon"]
    if survival_spec.has_key("l1_survival"):
      l1_survival = survival_spec["l1_survival"]
    else:
      l1_survival = 0.0
    if survival_spec.has_key("n_epochs"):
      n_epochs = survival_spec["n_epochs"]
    else:
      n_epochs = 1000
    

    if survival_spec.has_key("l1_regression"):
      l1_regression = survival_spec["l1_regression"]
    else:
      l1_regression = 0.0
    
    folds_survival =  survival_spec["folds_survival"]
    folds_regression =  survival_spec["folds_regression"]
    if survival_spec.has_key("add_tissue"):
      add_tissue = survival_spec["add_tissue"]
    else:
      add_tissue = False

    if survival_spec.has_key("filters"):
      filters = survival_spec["filters"]
    else:
      filters = []
    
    
    save_weights_template = os.path.join( logging_dict[SAVEDIR], "survival_weights_" ) 
    
    #train, validation = get_data(data_dict['validation_tissues'], f, d)
    train_survival, train_tissue, val_survival, val_tissue = get_data(f, d, filters =filters, add_tissue = add_tissue)

    
    #pdb.set_trace()

    projections, \
    probabilties, \
    times, \
    weights, train, val = run_tensorflow_survival_train_val( train_survival, train_tissue, \
                                                          val_survival, val_tissue, \
                                                          spec = survival_spec )
    model = weights[1]
    weights = weights[0]
    Z_val = val[0]
    T_val = val[1]
    E_val = val[2]

    Z_train = train[0]
    T_train = train[1]
    E_train = train[2]
        
    #model.PlotSurvival( E_val, T_val, Z_val )
                                                                                
    train_proj  = projections[0]
    val_proj    = projections[1]
    train_prob  = probabilties[0]
    val_prob    = probabilties[1]
    train_times = times[0]
    val_times   = times[1]
    #projections, probabilties, times, w, train, val
    
    disease = data_dict['validation_tissues'][0]


    #avg_proj = averages[0]
    #avg_prob = averages[1]

    mn_w = weights

    fig = pp.figure()
    ax2 = fig.add_subplot(111)
    roc_train, results_train, I_train = events_plot( train_proj, train, logging_dict,  data_dict['validation_tissues'][0], ax=ax2, name = "train" )

    rg0 = min( train_proj[I_train[0]] ),max( train_proj[I_train[0]] )   
    rg1 = []#min( train_proj[I_train[1]] ),max( train_proj[I_train[1]] )   
    rg2 = min( train_proj[I_train[2]] ),max( train_proj[I_train[2]] )    
    
    fig = pp.figure()
    ax1 = fig.add_subplot(111)
    roc_val, results_val, I_val = events_plot( val_proj, val, logging_dict,  data_dict['validation_tissues'][0], ax=ax1, name = "validation", rgs = [rg0,rg1,rg2] )
    
    fig = pp.figure()
    ax1 = fig.add_subplot(111)
    roc_val, results_val, I_val = events_plot( val_proj, val, logging_dict,  data_dict['validation_tissues'][0], ax=ax1, name = "validation2")
    
    
    
    #f.open()
    #d.open()
    
    # Z_train = f["/Z/TRAIN/Z/mu"].values
    # Z_val = f["/Z/VAL/Z/mu"].values
    # train_bcs = train_survival.index #f["/Z/TRAIN/Z/mu"].index.values.astype(str)
    #
    # disease_list = np.array( [st.split('_')[0] for st in train_bcs], dtype=str)
    # diseases = np.unique( disease_list)
    # n_diseases = len(diseases)
    #
    # z_min = np.min( Z_train )
    # z_max = np.max( Z_train )
    # z_plot = np.linspace( 1.1*z_min, 1.1*z_max, 500 )
    #
    # n_z = Z_train.shape[1]
    # train_means = np.zeros( (n_diseases,n_z) )
    # train_stds  = np.zeros( (n_diseases,n_z) )
    # val_means   = np.zeros(n_z)
    # val_stds    = np.zeros(n_z)
    #
    # for idx,disease in zip(range(n_diseases),diseases):
    #   I_disease = pp.find( disease_list == disease )
    #   train_means[idx,:] = Z_train[I_disease,:].mean(0)
    #   train_stds[idx,:] = Z_train[I_disease,:].std(0)
    #
    # val_means = Z_val.mean(0)
    # val_stds = Z_val.std(0)
    
    s.close()
    d.close()
    f.close()  
  pp.show()


  
  
  