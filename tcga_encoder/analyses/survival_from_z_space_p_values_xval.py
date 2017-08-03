from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
from tcga_encoder.analyses.survival_functions import *
from tcga_encoder.analyses.everything_functions import *
from tcga_encoder.analyses.everything_long import *

#from tcga_encoder.algorithms import *
import seaborn as sns
from sklearn.manifold import TSNE, locally_linear_embedding

from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
  
  
def main( data_location, results_location ):
  K = 5
  
  data_path    = os.path.join( HOME_DIR ,data_location ) #, "data.h5" )
  results_path = os.path.join( HOME_DIR, results_location )
  
  data_filename = os.path.join( data_path, "data.h5")
  fill_filename = os.path.join( results_path, "full_vae_fill.h5" )
  
  save_dir = os.path.join( results_path, "survival_p_values_xval" )
  check_and_mkdir(save_dir)
  #survival_curves_dir = os.path.join( survival_dir, "sig_curves" )
  #check_and_mkdir(survival_curves_dir)
  
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

  data_store.close()
  fill_store.close()
  
  n_tissues = len(tissue_names)
  n_random = 100
  random_names = ["r_%d"%(trial_idx) for trial_idx in range(n_random)]
  
  
  alpha=0.001
  
  split_nbr = 2 #[2,4]
  nbr_to_plot = 5
  

  split_p_values = pd.DataFrame( np.nan*np.ones((n_tissues,n_z) ), index = tissue_names, columns=z_names )
  for t_idx in range(n_tissues):

    t_ids = tissue_idx == t_idx
    tissue_name = tissue_names[t_idx]
    
    if tissue_name == "gbm":
      print "skipping gbm"
      continue
      
    tissue_dir = os.path.join( save_dir, tissue_name  )
    check_and_mkdir(tissue_dir)
    
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
      I = np.argsort(z)
      
      xval_groups = np.zeros(len(I))
      xval_splits = [[],[]]
      z_scores = []
      folds = StratifiedKFold(n_splits=K, shuffle = True, random_state=0)
      for train_ids, test_ids in folds.split( Z_values, events ): #[:,np.newaxis].astype(int) ):
    
        train_bcs = bcs[train_ids]
        test_bcs  = bcs[test_ids]
        k_z       = Z_values[:,z_idx][test_ids]
        k_events  = events[test_ids]
        k_times   = times[test_ids]
        #survival_curves_dir_split = os.path.join( survival_curves_dir, "q%d"%(split_nbr), "z%d"%(z_idx) )
      
        k_I = np.argsort(k_z)
        
        n_tissue = len(test_bcs)
        I_splits = survival_splits( k_events, k_I, split_nbr )
        groups = groups_by_splits( n_tissue, I_splits )
      
        results = multivariate_logrank_test(k_times, groups=groups, event_observed=k_events )
        z_scores.append( np.log( results.p_value ) )
        xval_groups[ test_ids ] = groups
        for idx in I_splits[0]:
          xval_splits[0].append(idx)
        for idx in I_splits[1]:
          xval_splits[1].append(idx)
        
      z_scores = -2*np.array(z_scores)
      mean_z_score = np.mean(z_scores)
      #pdb.set_trace()
      print tissue_name, z_idx, z_scores, mean_z_score
      split_p_values["z_%d"%(z_idx)].loc[tissue_name] = mean_z_score
        
      if mean_z_score >5:
        f=pp.figure()
        ax = plot_survival_by_splits( times, events, xval_splits, at_risk_counts=False,show_censors=True,ci_show=False, cmap = "rainbow")
        pp.title( "%s z%d z-score = %g"%( tissue_name, z_idx, mean_z_score ) )
        pp.savefig( tissue_dir + "/%s_z%d.png"%(tissue_name, z_idx), format="png" ) #, dpi=300)
        pp.close('all')
    
  split_p_values.to_csv( save_dir + "/z_scores.csv" )
  #  pdb.set_trace() #split_p_values

  
  # f=pp.figure()
  # for idx,split_nbr in zip( range(len(split_nbrs)), split_nbrs ):
  #   #split_p_values_random[ split_nbr ].drop("gbm",inplace=True)
  #   split_p_values[ split_nbr ].drop("gbm",inplace=True)
  #
  #   #split_p_values_random[ split_nbr ].to_csv( survival_dir + "/p_values_q%d_random.csv"%(split_nbr) )
  #   split_p_values[ split_nbr ].to_csv( survival_dir + "/p_values_q%d.csv"%(split_nbr) )
  #
  #   #pdb.set_trace()
  #   ax = f.add_subplot( 1,len(split_nbrs),idx+1 )
  #   #ax.hist( split_p_values_random[split_nbr].values.flatten(), bins=np.linspace(0,1,11), histtype="step", normed=True, color="red", lw=2 )
  #   ax.hist( split_p_values[split_nbr].values.flatten(), bins=np.linspace(0,1,11), histtype="step", normed=True, color="blue", lw=2 )
  #   #pdb.set_trace()
  #   pp.title( "%d splits"%(split_nbr) )
  #   #pp.legend(["random","z-space"])
  #   pp.savefig( survival_dir + "/p_values.png", format="png")#, dpi=300)
  #   pp.close('all')
  #
  #
  # pp.close('all')
  
  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )