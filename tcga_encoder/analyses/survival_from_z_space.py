from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
#from tcga_encoder.algorithms import *
import seaborn as sns
from sklearn.manifold import TSNE, locally_linear_embedding

def main( data_location, results_location ):
  data_path    = os.path.join( HOME_DIR ,data_location ) #, "data.h5" )
  results_path = os.path.join( HOME_DIR, results_location )
  
  data_filename = os.path.join( data_path, "data.h5")
  fill_filename = os.path.join( results_path, "full_vae_fill.h5" )
  
  survival_dir = os.path.join( results_path, "survival" )
  check_and_mkdir(survival_dir)
  
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
  
  from lifelines import CoxPHFitter
  from lifelines.datasets import load_regression_dataset
  from lifelines.utils import k_fold_cross_validation
  from lifelines import KaplanMeierFitter
  from lifelines.statistics import logrank_test
  # cf = CoxPHFitter()
  # cf.fit( Z, 'T', event_col='E', strata=['Tissue'])
  #
  # cf.print_summary()
  # S = np.hstack( [])
  # survival_data = pd.DataFrame( X)
  n_tissues = len(tissue_names)
  n_trials = 2*n_z
  trial_names = ["r_%d"%(trial_idx) for trial_idx in range(n_trials)]
  
  p_values_half  = np.ones( (n_tissues,n_z), dtype=float)
  p_values_third_random  = np.ones( (n_tissues,n_trials), dtype=float)
  p_values_third = np.ones( (n_tissues,n_z), dtype=float)
  p_values_fifth = np.ones( (n_tissues,n_z), dtype=float)
  p_values_tenth = np.ones( (n_tissues,n_z), dtype=float)
  
  for t_idx in range(n_tissues):
    t_ids = tissue_idx == t_idx
    tissue_name = tissue_names[t_idx]
    
    print "working %s"%(tissue_name)
    bcs = barcodes[t_ids]
    Z_tissue = Z.loc[ bcs ]
    
    events = Z_tissue["E"]
    times  = Z_tissue["T"]
    Z_values = Z_tissue[z_names].values
    
    n_tissue = len(bcs)
    if n_tissue < 40:
      continue
      
    half  = int(n_tissue/2.0)
    third = int(n_tissue/3.0)
    third = int(n_tissue/3.0)
    fifth = int(n_tissue/5.0)
    tenth = int(n_tissue/10.0)
    for z_idx in range(n_z):
      z = Z_values[:,z_idx]
      I = np.argsort(z)
      z1_half = I[:half]
      z2_half = I[half:]
      z1_third = I[:third]
      z2_third = I[-third:]
      z1_fifth = I[:fifth]
      z2_fifth = I[-fifth:]
      z1_tenth = I[:tenth]
      z2_tenth = I[-tenth:]
      
      #kmf = KaplanMeierFitter()
      #kmf.fit(times[z1_half], event_observed=z1_half[I] )
      
      results_half = logrank_test(times[z1_half], times[z2_half], events[z1_half], events[z2_half], alpha=.99 )
      p_values_half[t_idx,z_idx] = results_half.p_value
      results_third = logrank_test(times[z1_third], times[z2_third], events[z1_third], events[z2_third], alpha=.99 )
      p_values_third[t_idx,z_idx] = results_third.p_value
      
      results_fifth = logrank_test(times[z1_fifth], times[z2_fifth], events[z1_fifth], events[z2_fifth], alpha=.99 )
      p_values_fifth[t_idx,z_idx] = results_fifth.p_value
      results_tenth = logrank_test(times[z1_tenth], times[z2_tenth], events[z1_tenth], events[z2_tenth], alpha=.99 )
      p_values_tenth[t_idx,z_idx] = results_tenth.p_value
      #pdb.set_trace()
    
    # for trial_idx in range(n_trials):
    #   I = np.random.permutation(n_tissue)
    #   z1_third = I[:third]
    #   z2_third = I[-third:]
    #   results_third = logrank_test(times[z1_third], times[z2_third], events[z1_third], events[z2_third], alpha=.99 )
    #   p_values_third_random[t_idx,trial_idx] = results_third.p_value
    #events = Z["E"].loc[]
  
  #
  p_values_half  = pd.DataFrame( p_values_half, index = tissue_names, columns=z_names )
  p_values_third = pd.DataFrame( p_values_third, index = tissue_names, columns=z_names )
  p_values_fifth = pd.DataFrame( p_values_fifth, index = tissue_names, columns=z_names )
  p_values_tenth = pd.DataFrame( p_values_tenth, index = tissue_names, columns=z_names )
  #p_values_third_random = pd.DataFrame( p_values_third_random, index = tissue_names, columns=trial_names )
  #pdb.set_trace()
  #
  #
  # tissue_names = tissues.columns
  # tissue_idx = np.argmax( tissues.values, 1 )
  #
  # pdb.set_trace()
  # pp.savefig( tsne_dir + "/tsne_perplexity_%d.png"%(perplexity), format='png', dpi=300 )
  p_values_half.to_csv( survival_dir + "/p_values_half.csv" )
  p_values_third.to_csv( survival_dir + "/p_values_third.csv" )
  p_values_fifth.to_csv( survival_dir + "/p_values_fifth.csv" )
  p_values_tenth.to_csv( survival_dir + "/p_values_tenth.csv" )
  #p_values_third_random.to_csv( survival_dir + "/p_values_third_random.csv" )
  
  pv = p_values_third
  splits = "1/3"
  split_word="third"
  binses = [20,50,100]
  for pv,splits,split_word in zip( [p_values_half,p_values_third,p_values_fifth,p_values_tenth],["1/2","1/3","1/5","1/10"],["half","third","fifth","tenth"]):
    for bins in binses:
      pp.figure()
      pp.hist( pv.values.flatten(), bins, range=(0,1), normed=True, histtype="step", lw=3 )
      pp.plot( [0,1],[1,1], 'r-', lw=3)
      pp.legend( ["Z","random"])
      pp.xlabel("p-value logrank test")
      pp.ylabel("Pr(p_value)")
      pp.title("Comparison between p-values using latent space (%s splits)"%splits)
      pp.savefig( survival_dir + "/p_values_comparison_%s_%dbins.png"%(split_word,bins), format='png', dpi=300 )

  pp.close('all')
  #pdb.set_trace()
  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )