from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
from tcga_encoder.algorithms import *
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("talk")

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


import tensorflow as tf

#load_data_from_dict

#from tensorflow import *



#from models.layers import *
#from models.regularizers import *
#from models.algorithms import *

#from models.vae.tcga_models import *
#from utils.utils import *
#from data.load_datasets_from_broad import load_sources

#from utils.image_utils import *


def add_variables( var_dict, data_dict ):
  # add very specific numbers:
  var_dict["dna_dim"]    = data_dict['dataset'].GetDimension("DNA")
  var_dict["meth_dim"]   = data_dict['dataset'].GetDimension("METH")
  var_dict["rna_dim"]    = data_dict['dataset'].GetDimension("RNA")
  var_dict["tissue_dim"] = data_dict['dataset'].GetDimension("TISSUE")
  
def load_architecture( arch_dict, data_dict ):
  add_variables( arch_dict[VARIABLES], data_dict )
  return arch_dict[NETWORK]( arch_dict, data_dict)
  
# def load_architectures( arches, data ):
#   networks = OrderedDict()
#   for arch in arches:
#     networks[ arch[NAME] ] = load_architecture( arch, data )
#   return networks

  
######################################################################################################
if __name__ == "__main__":
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_file = sys.argv[1]
  print "Running: ",yaml_file
  
  y = load_yaml( yaml_file)
    
  #print y
  
  
  logging_dict = {}
  #print "Loading data"
  load_data_from_dict( y[DATA] )
  algo_dict = y[ALGORITHM]
  arch_dict = y[ARCHITECTURE]
  data_dict = y[DATA] #{N_TRAIN:4000}
  logging_dict = y[LOGGING]
  logging_dict[SAVEDIR] = os.path.join( HOME_DIR, os.path.join( logging_dict[LOCATION], logging_dict[EXPERIMENT] ) )
  # #networks = load_architectures( y[ARCHITECTURES], y[DATA] )
  #add_variables( arch_dict[VARIABLES], data_dict )
  network = load_architecture( arch_dict, data_dict )
  network_name = arch_dict[NAME]

  #
  
  # make BATCHER and reassign it to dict
  algo_dict[BATCHER] = algo_dict[BATCHER]( network_name, None, data_dict, algo_dict, arch_dict, logging_dict )


  algo_dict[BATCHER].network_name   = network_name
  algo_dict[BATCHER].network        = network
  
  sess = tf.InteractiveSession()
  
  results_dict = {}
  train( sess, network, algo_dict, data_dict, logging_dict, results_dict )

  batcher = algo_dict[BATCHER]
  model_store   = algo_dict[BATCHER].model_store
  latent_store  = algo_dict[BATCHER].latent_store
  epoch_store   = algo_dict[BATCHER].epoch_store
  data_store    = algo_dict[BATCHER].data_store
  fill_store    = algo_dict[BATCHER].fill_store
  
  model_store.open()
  data_store.open()
  latent_store.open()
  epoch_store.open()
  fill_store.open()
  
  # TEST FILL for all TARGETS
  rna_test    = data_store["/RNA/FAIR"].loc[ batcher.test_barcodes ]
  dna_0_test  = data_store["/DNA/channel/0"].loc[ batcher.test_barcodes ]
  #dna_1_test  = data_store["/DNA/channel/1"].loc[ batcher.test_barcodes ]
  #dna_2_test  = data_store["/DNA/channel/2"].loc[ batcher.test_barcodes ]
  #dna_3_test  = data_store["/DNA/channel/3"].loc[ batcher.test_barcodes ]
  meth_test   = data_store["/METH/FAIR"].loc[ batcher.test_barcodes ]
  tissue_test = data_store["/CLINICAL/TISSUE"].loc[ batcher.test_barcodes ]

  rna_train    = data_store["/RNA/FAIR"].loc[ batcher.train_barcodes ]
  dna_0_train  = data_store["/DNA/channel/0"].loc[ batcher.train_barcodes ]
  #dna_1_train  = data_store["/DNA/channel/1"].loc[ batcher.train_barcodes ]
  #dna_2_train  = data_store["/DNA/channel/2"].loc[ batcher.train_barcodes ]
  #dna_3_train  = data_store["/DNA/channel/3"].loc[ batcher.train_barcodes ]
  meth_train   = data_store["/METH/FAIR"].loc[ batcher.train_barcodes ]
  tissue_train = data_store["/CLINICAL/TISSUE"].loc[ batcher.train_barcodes ]
  
  
  ALL_SURVIVAL = data_store["/CLINICAL/data"][["patient.days_to_last_followup","patient.days_to_death"]]
  tissue_barcodes = np.array( ALL_SURVIVAL.index.tolist(), dtype=str )
  surv_barcodes = np.array([ x+"_"+y for x,y in tissue_barcodes])
  NEW_SURVIVAL = pd.DataFrame( ALL_SURVIVAL.values, index =surv_barcodes, columns = ALL_SURVIVAL.columns ) 
  test_survival = pd.concat( [NEW_SURVIVAL, fill_store["/Z/TEST/Z/mu"]], axis=1, join = 'inner' )
  val_survival = pd.concat( [NEW_SURVIVAL, fill_store["/Z/VAL/Z/mu"]], axis=1, join = 'inner' )
  
  predict_survival = pd.concat( [test_survival, val_survival], axis=0, join = 'outer' )
  predict_barcodes = predict_survival.index
  splt = np.array( [ [s.split("_")[0], s.split("_")[1]] for s in predict_barcodes ] )
  predict_survival = pd.DataFrame( predict_survival.values, index = splt[:,1], columns = predict_survival.columns )
  predict_survival["disease"] = splt[:,0]
  
  Times = predict_survival[ "patient.days_to_last_followup" ].fillna(0).values.astype(int)+predict_survival[ "patient.days_to_death" ].fillna(0).values.astype(int)
  predict_survival["T"] = Times
  Events = (1-np.isnan( predict_survival[ "patient.days_to_death" ].astype(float)) ).astype(int)
  predict_survival["E"] = Events
  
  z_columns = []
  columns = ["T","E"]
  for zidx in range(20):
    columns.append("z%d"%(zidx))
    z_columns.append("z%d"%(zidx))
    
  reg_data = pd.DataFrame( predict_survival[columns].values.astype(int), columns=columns)
  
  from lifelines import KaplanMeierFitter
  from sklearn.cluster import KMeans
  
  def kmf_split( predict_survival, K = 3, diseases = ["lgg"], Zs = [0,1,2,3,4,5,6,7,8,9,10] ):

    z_columns = []
    for z in Zs:
      z_columns.append( "z%d"%z) 
    #diseases = ["kirc","kirp","kich"]
    #diseases = ["brca"]
    #K = 6
    for disease in diseases: #batcher.tissue_names:
      f = pp.figure()
      kmf = KaplanMeierFitter()
      ax = f.add_subplot(111)
      
      disease_query = predict_survival["disease"].values == disease
      disease_survival = predict_survival[ disease_query ]

      
      T = disease_survival["T"].values
      E = disease_survival["E"].values
      Z = disease_survival[z_columns].values
      if len(T)>0:
        kmf.fit(T, event_observed=E, label = disease)
        ax=kmf.plot(ax=ax, ci_force_lines=True)
      else:
        continue
      
      kmeans = KMeans(n_clusters=K ).fit(Z)
      for k in range(K):
        I = pp.find( kmeans.labels_==k)
        Ti=T[I]
        Ei=E[I]
      
        if len(Ti)>0:
          kmf.fit(Ti, event_observed=Ei, label = disease + "_k=%d"%k)
          ax=kmf.plot(ax=ax)
          
  def kmf_quantiles( predict_survival, diseases = ["lgg"], z=0, quants = [0.0,0.1,0.9,1.0] ):

    z_columns = ["z%d"%z]

    for disease in diseases: #batcher.tissue_names:
      f = pp.figure()
      kmf = KaplanMeierFitter()
      ax = f.add_subplot(111)
      
      disease_query = predict_survival["disease"].values == disease
      disease_survival = predict_survival[ disease_query ]

      
      T = disease_survival["T"].values
      E = disease_survival["E"].values
      Z = np.squeeze( disease_survival[z_columns].values )
      print Z
      
      iZ = np.argsort(Z)
      nz = len(iZ)
      Is = []
      for a,b in zip(quants[:-1],quants[1:]):
        Is.append( iZ[ a*nz : b*nz ] )
        
      if len(T)>0:
        kmf.fit(T, event_observed=E, label = disease)
        ax=kmf.plot(ax=ax, ci_force_lines=True)
      else:
        continue
      
      #kmeans = KMeans(n_clusters=K ).fit(Z)
      for k,I in zip(range(len(Is)),Is):
        #I = pp.find( kmeans.labels_==k)
        Ti=T[I]
        Ei=E[I]
      
        if len(Ti)>0:
          kmf.fit(Ti, event_observed=Ei, label = disease + "_q=%d"%k)
          ax=kmf.plot(ax=ax)
  for disease in batcher.tissue_names:
    kmf_split( predict_survival, K=3, diseases = [disease], Zs = [2] )
  pp.show()
     
  # other_barcodes = np.setdiff1d( data_store["/RNA/FAIR"].index, np.union1d(batcher.train_barcodes,batcher.test_barcodes))
  # rna_other    = data_store["/RNA/FAIR"].loc[ other_barcodes ]
  # dna_0_other  = data_store["/DNA/channel/0"].loc[ other_barcodes ]
  # dna_1_other  = data_store["/DNA/channel/1"].loc[ other_barcodes ]
  # dna_2_other  = data_store["/DNA/channel/2"].loc[ other_barcodes ]
  # dna_3_other  = data_store["/DNA/channel/3"].loc[ other_barcodes ]
  # meth_other   = data_store["/METH/FAIR"].loc[ other_barcodes ]
  # tissue_other = data_store["/CLINICAL/TISSUE"].loc[ other_barcodes ]
  #
  # inputs_combos = ["RNA","DNA","METH","RNA+DNA","RNA+METH","DNA+METH","RNA+DNA+METH"]
  # targets = OrderedDict()
  # targets[RNA] = {"observed":rna_test, "error":"mse"}
  # targets[METH] = {"observed":meth_test, "error":"mse"}
  # targets[DNA+"/0"] = {"observed":dna_0_test, "error":"auc"}
  # targets[DNA+"/1"] = {"observed":dna_1_test, "error":"auc"}
  # targets[DNA+"/2"] = {"observed":dna_2_test, "error":"auc"}
  # targets[DNA+"/3"] = {"observed":dna_3_test, "error":"auc"}
  #
  # print "==================================="
  # print "   ERROR                           "
  # print "==================================="
  # for target_source, values in targets.iteritems():
  #   print "++++ %s"%target_source
  #   observed  = values["observed"].values
  #   for inputs in inputs_combos:
  #     predicted = fill_store["/Fill/%s/%s"%(target_source,inputs)].values
  #
  #     if values["error"] == "mse":
  #       error = np.mean( np.square( observed-predicted ) )
  #       print "%s\t%10s\t%0.6f"%(target_source, inputs, error  )
  #     elif values["error"] == "auc":
  #
  #       p_flattened = predicted.flatten()
  #       o_flattened = observed.flatten()
  #
  #       error = roc_auc_score(o_flattened,p_flattened)
  #       print "%s\t%10s\t%0.6f"%(target_source, inputs, error  )
  #
  #
  # print "==================================="
  # print "   LOGLIK                          "
  # print "==================================="
  # for target_source, values in targets.iteritems():
  #   print "++++ %s"%target_source
  #   #observed  = values["observed"].values
  #   for inputs in inputs_combos:
  #     predicted = fill_store["/Loglik/%s/%s"%(target_source,inputs)].values
  #
  #     error = np.mean( np.sum( predicted, 1 ) )
  #     print "%s\t%10s\t%0.6f"%(target_source, inputs, error  ) 
  
  #print model_store
  # print data_store
  # print latent_store
  # print epoch_store
  # def violinplot( data, order_list, prior_mean, prior_std, x=None, y=None, orient = "v", sort = True, lims = (-15,15), width=0.95,lw=0.5 ):
  #   n = len(prior_mean)
  #   if sort is True:
  #     i_order = np.argsort( prior_mean )
  #   else:
  #     i_order = np.arange( n, dtype=int )
  #
  #   if orientation == "v":
  #     pp.fill_between(np.arange(n), prior_mean[i_order]+2*prior_std[i_order], prior_mean[i_order]-2*prior_std[i_order], color='black', alpha=0.25)
  #     pp.plot(prior_mean[i_order], 'k-')
  #     sns.violinplot(  x=x, y=y, data = data, width=width, linewidth=lw, order=order_list[i_order], orient=orient )
  #     #sns.swarmplot(  x=x, y=y, data = data, linewidth=lw, order=order_list[i_order], orient=orient )
  #     pp.ylim(lims)
  #   else:
  #
  #     pp.fill_betweenx(np.arange(n), prior_mean[i_order]+2*prior_std[i_order], prior_mean[i_order]-2*prior_std[i_order], color='black', alpha=0.25)
  #     pp.plot(prior_mean[i_order], np.arange( n, dtype=int ), 'k-')
  #     sns.violinplot( x=y, y=x, data = data, width=width, linewidth=lw, order=order_list[i_order], orient=orient )
  #     pp.xlim(lims)
  #
  #
  #
  #   pp.xlabel("")
  #   pp.ylabel("")
  #   #pp.title(tissue)
  #   pp.grid('on')
    
  # def plot_z( z, test_tissue, use_columns, fill_store ):
  #   pp.figure()
  #
  #   pp.title( "Z%d"%(z))
  #
  #   rec_means = []
  #   rec_stds  = []
  #   gen_means = []
  #   gen_stds  = []
  #   z_rec_df = fill_store["/Z/rec/mu"]["z%d"%z]
  #   z_rec_df["Tissue"] = pd.Series( [], index=z_rec_df.index)
  #   for tissue in use_columns:
  #     query = test_tissue[tissue].values==1.0
  #     barcodes = test_tissue[tissue][query].index
  #
  #     # rec_means.append( fill_store["/Z/rec/mu"]["z%d"%z].loc[barcodes].values )
  #     # rec_stds.append( np.sqrt(fill_store["/Z/rec/var"]["z%d"%z].loc[barcodes].values) )
  #     # gen_means.append( fill_store["/Z/gen/mu"]["z%d"%z].loc[barcodes].values )
  #     # gen_stds.append( np.sqrt(fill_store["/Z/gen/var"]["z%d"%z].loc[barcodes].values) )
  #
  #   rec_means = pd.DataFrame( np.array(rec_means), columns = use_columns )
  #   rec_stds  = pd.DataFrame( np.array(rec_stds), columns = use_columns )
  #   gen_means = pd.DataFrame( np.array(gen_means), columns = use_columns )
  #   gen_stds   = pd.DataFrame( np.array(gen_stds), columns = use_columns )
  #
  #   #violinplot(rec_means, z_list, prior_mu_z, prior_std_z, orient=orientation, sort=False, lims=lims)
  #   #sns.violinplot( x=None, y=None, data = rec_means, width=width, linewidth=lw, order=order_list[i_order], orient=orient )
  #   sns.violinplot( x=None, y=None, data = rec_means, width=width, linewidth=lw, order=order_list[i_order], orient=orient )
  #
  # test_tissue = data_store["/CLINICAL/TISSUE"].loc[batcher.test_barcodes]
  # most_common_order = np.argsort(-test_tissue.values.sum(0))
  # use_ids = pp.find( test_tissue.values.sum(0) > 1 )
  # use_columns = test_tissue.columns[use_ids]
  #
  # n_z = batcher.n_z
  # for z in range(2):
  #   plot_z( z, test_tissue, use_columns, fill_store )
  
  model_store.close()
  data_store.close()
  latent_store.close()
  epoch_store.close()
  
  
  