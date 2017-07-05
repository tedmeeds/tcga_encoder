from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
#from tcga_encoder.algorithms import *
import seaborn as sns
from sklearn.manifold import TSNE, locally_linear_embedding

import lifelines
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

def main( data_location, results_location ):
  data_path    = os.path.join( HOME_DIR ,data_location ) #, "data.h5" )
  results_path = os.path.join( HOME_DIR, results_location )
  
  data_filename = os.path.join( data_path, "data.h5")
  fill_filename = os.path.join( results_path, "full_vae_fill.h5" )
  
  save_dir = os.path.join( results_path, "survival_concordance" )
  check_and_mkdir(save_dir)
  survival_curves_dir = os.path.join( save_dir, "sig_curves" )
  check_and_mkdir(survival_curves_dir)
  
  print "HOME_DIR: ", HOME_DIR
  print "data_filename: ", data_filename
  print "fill_filename: ", fill_filename
  
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
  
  Z=Z.loc[barcodes]
  
  tissues = data_store["/CLINICAL/TISSUE"].loc[barcodes]
  #Overall Survival (OS) The event call is derived from "vital status" parameter. The time_to_event is in days, equals to days_to_death if patient deceased; in the case of a patient is still living, the time variable is the maximum(days_to_last_known_alive, days_to_last_followup).  This pair of clinical parameters are called _EVENT and _TIME_TO_EVENT on the cancer browser. 
  
  
  ALL_SURVIVAL = data_store["/CLINICAL/data"][["patient.days_to_last_followup","patient.days_to_death","patient.days_to_birth"]]
  tissue_barcodes = np.array( ALL_SURVIVAL.index.tolist(), dtype=str )
  surv_barcodes = np.array([ x+"_"+y for x,y in tissue_barcodes])
  NEW_SURVIVAL = pd.DataFrame( ALL_SURVIVAL.values, index =surv_barcodes, columns = ALL_SURVIVAL.columns ) 
  NEW_SURVIVAL = NEW_SURVIVAL.loc[barcodes]
  #clinical = data_store["/CLINICAL/data"].loc[barcodes]
  
  Age = NEW_SURVIVAL[ "patient.days_to_birth" ].values.astype(int)
  Times = NEW_SURVIVAL[ "patient.days_to_last_followup" ].fillna(0).values.astype(int)+NEW_SURVIVAL[ "patient.days_to_death" ].fillna(0).values.astype(int)
  Events = (1-np.isnan( NEW_SURVIVAL[ "patient.days_to_death" ].astype(float)) ).astype(int)
  
  ok_age_query = Age<-10
  ok_age = pp.find(ok_age_query )
  tissues = tissues[ ok_age_query ]
  #pdb.set_trace()
  Age=-Age[ok_age]
  Times = Times[ok_age]
  Events = Events[ok_age]
  barcodes = barcodes[ok_age]
  NEW_SURVIVAL = NEW_SURVIVAL.loc[barcodes]
  
  #ok_followup_query = NEW_SURVIVAL[ "patient.days_to_last_followup" ].fillna(0).values>=0
  #ok_followup = pp.find( ok_followup_query )
  
  bad_followup_query = NEW_SURVIVAL[ "patient.days_to_last_followup" ].fillna(0).values.astype(int)<0
  bad_followup = pp.find( bad_followup_query )
  
  ok_followup_query = 1-bad_followup_query
  ok_followup = pp.find( ok_followup_query )
  
  bad_death_query = NEW_SURVIVAL[ "patient.days_to_death" ].fillna(0).values.astype(int)<0
  bad_death = pp.find( bad_death_query )
  
  #pdb.set_trace()
  Age=Age[ok_followup]
  Times = Times[ok_followup]
  Events = Events[ok_followup]
  barcodes = barcodes[ok_followup]
  NEW_SURVIVAL = NEW_SURVIVAL.loc[barcodes]
  
  Z = Z.loc[barcodes]
  Z["E"] = Events
  Z["T"] = Times
  Z["Age"] = np.log(Age)
  
  tissues = data_store["/CLINICAL/TISSUE"].loc[barcodes]
  
  tissue_names = tissues.columns
  tissue_idx = np.argmax( tissues.values, 1 )
  
  Z["Tissue"] = tissue_idx


  n_tissues = len(tissue_names)
  n_random = 1000
  random_names = ["r_%d"%(trial_idx) for trial_idx in range(n_random)]
  
  
  alpha=0.001
  
  nbr_to_plot = 5
  concordance_values = {}
  concordance_random = {}
  
  concordance_z_values = pd.DataFrame( np.nan*np.ones((n_tissues,n_z) ), index = tissue_names, columns=z_names )
  concordance_z_random = pd.DataFrame( np.nan*np.ones((n_tissues,n_random) ), index = tissue_names, columns=random_names )
  concordance_z_values_xval = pd.DataFrame( np.nan*np.ones((n_tissues,n_z) ), index = tissue_names, columns=z_names )
  concordance_I_values = pd.DataFrame( np.nan*np.ones((n_tissues,n_z) ), index = tissue_names, columns=z_names )
  concordance_I_random = pd.DataFrame( np.nan*np.ones((n_tissues,n_random) ), index = tissue_names, columns=random_names )

  concordance_z_p_values = pd.DataFrame( np.ones( (n_tissues,n_z) ), \
                                        index = tissue_names, \
                                        columns = z_names )
  # cf = CoxPHFitter()
  # scores = k_fold_cross_validation(cf, Z, 'T', event_col='E', k=5)
  # pdb.set_trace()
  split_nbr = 3
  for t_idx in range(n_tissues):

    t_ids = tissue_idx == t_idx
    tissue_name = tissue_names[t_idx]
    
    if tissue_name == "gbm":
      print "skipping gbm"
      continue
    print "working %s"%(tissue_name)
    bcs = barcodes[t_ids]
    Z_tissue = Z.loc[ bcs ]
    
    events = Z_tissue["E"]
    times  = Z_tissue["T"]
    Z_values = Z_tissue[z_names].values
    
    
    n_tissue = len(bcs)
    print "  using z_values"
    for z_idx in range(n_z):
      z = Z_values[:,z_idx]
      z_data = Z_tissue[ ["z_%d"%(z_idx), "E","T"] ]
      I = np.argsort(z)
      z_concordance = lifelines.utils.concordance_index(times[I], z, event_observed=events[I]) 
      z_concordance = max( z_concordance, 1.0-z_concordance )
      concordance_z_values["z_%d"%(z_idx)].loc[tissue_name] = z_concordance

    print "  using random"
    for r_idx in range(n_random):
      #z = Z_values[:,z_idx]
      z = np.random.randn(n_tissue)
      I = np.argsort(z) #np.random.permutation(n_tissue)
      z_concordance = lifelines.utils.concordance_index(times[I], z, event_observed=events[I])
      z_concordance = max( z_concordance, 1.0-z_concordance )
      concordance_z_random["r_%d"%(r_idx)].loc[tissue_name] = z_concordance
      
    v = concordance_z_values.loc[tissue_name].values
    r = concordance_z_random.loc[tissue_name].values
    concordance_z_p_values.loc[tissue_name] = (1.0 + (v[:,np.newaxis]>r).sum(1))/(1.0+len(r))
    conc=concordance_z_p_values.loc[tissue_name]
    sig = (concordance_z_p_values.loc[tissue_name] < 0.05).astype(int)
    z_sig_names = sig[ sig==1 ].index.values
    for z_name in z_sig_names:
      z_idx = int( z_name.split("_")[1] )
      z = Z_values[:,z_idx]
      #z_data = Z_tissue[ ["z_%d"%(z_idx), "E","T"] ]
      I = np.argsort(z)
      I_splits = np.array_split( I, split_nbr ) 
      #groups = np.zeros(n_tissue)
      # k = 1
      # for splits in I_splits[1:]:
      #   groups[splits] = k; k+=1
        
      results = logrank_test(times[I_splits[0]], times[I_splits[-1]], events[ I_splits[0] ], events[ I_splits[-1] ] )
      p_value = results.p_value
      c = conc[ z_name ]
      f = pp.figure()
      ax= f.add_subplot(111)
      kmf = KaplanMeierFitter()
      k=0
      for splits in I_splits:
        kmf.fit(times[splits], event_observed=events[splits], label="q=%d/%d"%(k+1,split_nbr)  )
        ax=kmf.plot(ax=ax,at_risk_counts=False,show_censors=True,ci_show=False)
        k+=1
      pp.ylim(0,1)
      pp.title( "%s %s p-value = %0.4f concordance = %0.3f "%( tissue_name, z_name, p_value, c ) )
      
      pp.savefig( survival_curves_dir + "/%s_%s_p%0.5f_c%0.3f.png"%(tissue_name, z_name, p_value, c), format="png", dpi=300)
      pp.savefig( survival_curves_dir + "/%s_%s_p%0.5f_c%0.3f.png"%(z_name, tissue_name, p_value, c), format="png", dpi=300)
      
      
    #pdb.set_trace()

  concordance_z_random.drop("gbm",inplace=True)
  concordance_z_values.drop("gbm",inplace=True)
  concordance_z_p_values.drop("gbm",inplace=True)
  # concordance_z_p_values = pd.DataFrame( np.ones( concordance_z_values.values.shape), \
  #                                       index = concordance_z_values.index, \
  #                                       columns = concordance_z_values.columns )
                                        
  # for tissue in concordance_z_random.index.values:
  #   v = concordance_z_values.loc[tissue].values
  #   r = concordance_z_random.loc[tissue].values
  #   concordance_z_p_values.loc[tissue] = (1.0 + (v[:,np.newaxis]>r).sum(1))/(1.0+len(r))
  
  concordance_z_p_values.to_csv( save_dir + "/concordance_z_p_values.csv" )  
  concordance_z_random.to_csv( save_dir + "/concordance_z_random.csv" )  
  concordance_z_values.to_csv( save_dir + "/concordance_z_values.csv" )  
  #pdb.set_trace()
  
  f = pp.figure()
  ax_z = f.add_subplot(221)
  ax_log_z = f.add_subplot(223)
  ax_p = f.add_subplot(222)
  ax_log_p = f.add_subplot(224)
  
  bins_conc=np.linspace(0.5,1,21)
  bins_p=np.linspace(0.0,1,21)
  ax_z.hist( concordance_z_values.values.flatten(), bins=bins_conc, normed=True, histtype="step", lw=2, log=False) 
  ax_z.hist( concordance_z_random.values.flatten(), bins=bins_conc, normed=True, histtype="step", lw=2, log=False)
  ax_log_z.hist( concordance_z_values.values.flatten(), bins=bins_conc, normed=True, histtype="step", lw=2, log=True) 
  ax_log_z.hist( concordance_z_random.values.flatten(), bins=bins_conc, normed=True, histtype="step", lw=2, log=True)  
  
  ax_p.hist( concordance_z_p_values.values.flatten(), bins=bins_p, normed=True, histtype="step", lw=2, log=False)
  ax_log_p.hist( concordance_z_p_values.values.flatten(), bins=bins_p, normed=True, histtype="step", lw=2, log=True) 
  pp.savefig( save_dir + "/p_values.png", format="png", dpi=300)
  
  
  
  return concordance_z_random, concordance_z_values, concordance_z_p_values
  #, concordance_z_p_values_xval
  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  concordance_z_random, concordance_z_values, concordance_z_p_values = main( data_location, results_location )
  
  