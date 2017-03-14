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

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def plot_data_with_importance( w, importance, i_importance, name, save_location, n = 20 ):
  ff=pp.figure(figsize=(16,6)); ax1 = ff.add_subplot(111); 
  n=len(w.T)
  sns.heatmap( w.T, linewidths=0.001, xticklabels=False, cbar=False  ); pp.yticks( rotation=0, size=10); pp.title(name)
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
      
def main(yaml_file, weights_matrix):
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
  survival=pd.HDFStore( survival_location, "r" )
  d=data_dict['store'] #pd.HDFStore( data_location, "r" )
  f=pd.HDFStore( fill_location, "r" ) 
  
  #pdb.set_trace()
  
  
  for survival_spec in survival_dict:
    name = survival_spec["name"]
    if survival_spec.has_key("use_cuda"):
      use_cuda = survival_spec["use_cuda"]
    else:
      use_cuda = False
    print "usign cuda: ", use_cuda
    print "running xval x to z ,", data_dict['validation_tissues']
    
    if survival_spec.has_key("l1_survival"):
      l1_survival = survival_spec["l1_survival"]
    else:
      l1_survival = 0.0
    if survival_spec.has_key("n_epochs"):
      n_epochs = survival_spec["n_epochs"]
    else:
      n_epochs = 1000
    

    if survival_spec.has_key("l1_regression_list"):
      l1_regression_list = survival_spec["l1_regression_list"]
    else:
      l1_regression_list = [0.0]
    
    folds_regression =  survival_spec["folds_regression"]
    disease = data_dict['validation_tissues'][0]
    
    survival_selection = "rank"
    survival.open()
    if survival_selection == "rank":
      survival_target = np.squeeze( survival["/survival_rank"].values )
    elif survival_selection == "loglike":
      survival_target =  np.squeeze( survival["/survival_log_like"].values )
    survival.close()
    
    f.open()
    d.open()
    ALL_SURVIVAL = d["/CLINICAL/data"][["patient.days_to_last_followup","patient.days_to_death"]]
    tissue_barcodes = np.array( ALL_SURVIVAL.index.tolist(), dtype=str )
    surv_barcodes = np.array([ x+"_"+y for x,y in tissue_barcodes])
    NEW_SURVIVAL = pd.DataFrame( ALL_SURVIVAL.values, index =surv_barcodes, columns = ALL_SURVIVAL.columns ) 
    val_survival  = pd.concat( [NEW_SURVIVAL, f["/Z/VAL/Z/mu"]], axis=1, join = 'inner' )
  
    f.close()
    d.close()
  
    #-------
    predict_survival_train = val_survival #pd.concat( [test_survival, val_survival], axis=0, join = 'outer' )
    predict_barcodes_train = predict_survival_train.index
    splt = np.array( [ [s.split("_")[0], s.split("_")[1]] for s in predict_barcodes_train ] )
    predict_survival_train = pd.DataFrame( predict_survival_train.values, index = splt[:,1], columns = predict_survival_train.columns )
    predict_survival_train["disease"] = splt[:,0]

    T_train = predict_survival_train[ "patient.days_to_last_followup" ].fillna(0).values.astype(int)+predict_survival_train[ "patient.days_to_death" ].fillna(0).values.astype(int)
    predict_survival_train["T"] = T_train
    E_train = (1-np.isnan( predict_survival_train[ "patient.days_to_death" ].astype(float)) ).astype(int)
    predict_survival_train["E"] = E_train
    
    I = np.argsort(survival_target)
    third = int(len(I)/3.0)
    half = int(len(I)/2.0)
    #data = {"miRNA":[ "miRNA", ["/miRNA/FAIR"] ],"RNA":[ "RNA", ["/RNA/FAIR"] ], "DNA":["DNA",["/DNA/channel/0"]], "METH":["METH",["/METH/FAIR"]]}
    data = {"miRNA":[ "miRNA", ["/miRNA/FAIR"] ], "METH":["METH",["/METH/FAIR"]],"RNA":[ "RNA", ["/RNA/FAIR"] ]}
    #data = {"METH":["METH",["/METH/FAIR"]]}
    #data = {"RNA":["RNA",["/RNA/FAIR"]]}
    results = {}
    #survival_target=mn_proj #- mn_proj.mean()
    n2show = 20
    for data_name, data_info in data.iteritems():
      print("WORKING %s"%(data_name))
      best_mse = np.inf
      mses = []
      for l1_regression in l1_regression_list:
        data_names = data_info[0]
        data_keys  = data_info[1]
        reg_projections, \
        reg_weights, \
        reg_averages, \
        reg_X, \
        reg_y = run_survival_prediction_xval_regression( data_dict['validation_tissues'], f, d, \
                                                        targets=survival_target, \
                                                        data_keys = data_keys, \
                                                        data_names = data_names, \
                                                        l1 = l1_regression, \
                                                        k_fold = folds_regression, seed = 2, use_cuda=use_cuda  )

        reg_Ws = reg_weights[2]
        reg_bs = reg_weights[3]
        #print 'biases: ', reg_rna_bs.mean()
        I_reg_parameters = np.argsort( -np.abs( reg_weights[0] ) )
        I_reg_predictions = np.argsort( reg_projections[0] )

        sorted_reg_X = pd.DataFrame( reg_X.values[I_reg_predictions,:][:,I_reg_parameters], columns = [reg_X.columns[i] for i in I_reg_parameters] )
      
        high_rank_names = [reg_X.columns[i] for i in I_reg_parameters[:n2show]]
        low_rank_names = [reg_X.columns[i] for i in I_reg_parameters[-n2show:]]
      
        y_pred = reg_projections[0];
        y_true = reg_y ;
        reg_r2 = r2_score(y_true, y_pred)
        reg_mse = mean_squared_error(y_true, y_pred)
        reg_expvar = explained_variance_score(y_true, y_pred)
        reg_corr = np.corrcoef( y_true, y_pred )[0][1]
      
        print "%s Regression corr "%data_name, reg_corr
        print "%s Regression mse  "%data_name, reg_mse

        mses.append(reg_mse)
        print "%s MSE so far: "%(data_name), mses
        if reg_mse < best_mse:
          best_mse = reg_mse
          
          importance = np.abs( reg_weights[0])
      
          I_w_reg_parameters_importance = np.argsort( -importance )
          w_sorted_reg_full = pd.DataFrame( np.dot( reg_X.values[I_reg_predictions,:][:,I_w_reg_parameters_importance], np.sign(np.diag(reg_weights[0][I_w_reg_parameters_importance]))), columns = [reg_X.columns[i] for i in I_w_reg_parameters_importance] )
          
          
          w_sorted_reg_X3 = pd.DataFrame( np.dot( reg_X.values[I_reg_predictions,:][:,I_w_reg_parameters_importance[:n2show]], np.sign(np.diag(reg_weights[0][I_w_reg_parameters_importance[:n2show]]))), columns = [reg_X.columns[i] for i in I_w_reg_parameters_importance[:n2show]] )
      
      
          save_location_data = os.path.join( logging_dict[SAVEDIR], "survival_regression_best_%s_"%(survival_selection) )

          plot_data_with_importance(w_sorted_reg_X3, importance, I_w_reg_parameters_importance, "%s %s"%(disease, data_name), save_location_data )


          save_location_reg = os.path.join( logging_dict[SAVEDIR], "survival_regression_best_W_%s_%s.png"%(survival_selection,data_name) )
          f0=pp.figure()
          ax1 = f0.add_subplot(111)
          ax1.plot(reg_Ws[:,I_w_reg_parameters_importance].T, 'k.', alpha=0.85)
          ax1.plot(reg_Ws[:,I_w_reg_parameters_importance].T.mean(1), 'ro', alpha=0.85)
          pp.savefig(save_location_reg, dpi=300, format='png')

          save_location_reg = os.path.join( logging_dict[SAVEDIR], "survival_regression_best_predictions_v_true_%s_%s.png"%(survival_selection,data_name) )
          f0=pp.figure()
          ax1 = f0.add_subplot(111)
          ax1.plot(y_true, y_pred, 'k.', alpha=0.5)
          pp.xlabel( "True"); pp.ylabel( "Pred"); pp.title( "r2 = %0.3f  mse = %0.3f  corr = %0.3f"%(reg_r2,reg_mse,reg_corr ))
          pp.savefig(save_location_reg, dpi=300, format='png')
      
          f2=pp.figure()
          ax1 = f2.add_subplot(111)
          I = np.argsort( - y_pred )
          I0 = I[:half]
          I1 = [] #I[third:2*third]
          I2 = I[half:]
          kmf = KaplanMeierFitter()
          if len(I2) > 0:
            kmf.fit(T_train[I2], event_observed=E_train[I2], label =  "lda_1 E=%d C=%d"%(E_train[I2].sum(),len(I2)-E_train[I2].sum()))
            ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='red')
          if len(I1) > 0:
            kmf.fit(T_train[I1], event_observed=E_train[I1], label =  "lda_1 E=%d C=%d"%(E_train[I1].sum(),len(I1)-E_train[I1].sum()))
            ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='green')
          if len(I0) > 0:
            kmf.fit(T_train[I0], event_observed=E_train[I0], label = "lda_0 E=%d C=%d"%(E_train[I0].sum(),len(I0)-E_train[I0].sum()))
            ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='blue')
          results = logrank_test(T_train[I0], T_train[I2], event_observed_A=E_train[I0], event_observed_B=E_train[I2])
          print "LOG RANK TEST: ", results.test_statistic
          save_location = os.path.join( logging_dict[SAVEDIR], "survival_regression_best_survival_%s_%s.png"%(survival_selection,data_name) )  
          results = logrank_test(T_train[I0], T_train[I2], event_observed_A=E_train[I0], event_observed_B=E_train[I2])
          pp.title("%s Log-rank Test: %0.1f"%(disease, results.test_statistic))
          pp.savefig(save_location, dpi=300, format='png')
          
          survival.open()
          if survival_selection == "rank":
            survival["/survival_rank/w/%s"%(data_name)] = w_sorted_reg_full
          elif survival_selection == "loglike":
            survival["/survival_log_like/w/%s"%(data_name)] = w_sorted_reg_full
          survival.close()
          pp.close('all')
            
    d.close()
    f.close()  
  
  
######################################################################################################
if __name__ == "__main__":
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_file = sys.argv[1]
  print "Running: ",yaml_file  
  weights_matrix = []
  main( yaml_file, weights_matrix )
  pp.close('all')
  #pp.show()

  
  
  