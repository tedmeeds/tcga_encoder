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
  
  tsne_dir = os.path.join( results_path, "survival" )
  check_and_mkdir(tsne_dir)
  
  print "HOME_DIR: ", HOME_DIR
  print "data_filename: ", data_filename
  print "fill_filename: ", fill_filename
  
  print "LOADING stores"
  data_store = pd.HDFStore( data_filename, "r" )
  fill_store = pd.HDFStore( fill_filename, "r" )
  
  Z_train = fill_store["/Z/TRAIN/Z/mu"]
  Z_val = fill_store["/Z/VAL/Z/mu"]
  
  Z = np.vstack( (Z_train.values, Z_val.values) )
  #pdb.set_trace()
  Z = pd.DataFrame( Z, index = np.hstack( (Z_train.index.values, Z_val.index.values)), columns = ["z_%d"%z_idx for z_idx in range(Z.shape[1])])
  
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
  cf = CoxPHFitter()
  cf.fit( Z, 'T', event_col='E', strata=['Tissue'])

  cf.print_summary()
  # S = np.hstack( [])
  # survival_data = pd.DataFrame( X)
  
  pdb.set_trace()
  #
  #
  # tissue_names = tissues.columns
  # tissue_idx = np.argmax( tissues.values, 1 )
  #
  # pdb.set_trace()
  # pp.savefig( tsne_dir + "/tsne_perplexity_%d.png"%(perplexity), format='png', dpi=300 )
  
  #pdb.set_trace()
  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )