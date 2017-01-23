from tcga_encoder.utils.helpers import *
from lifelines import KaplanMeierFitter
from sklearn.cluster import KMeans
import pdb
def kmf_split( predict_survival_train, predict_survival_test, K, disease, Zs ):

  z_columns = []
  for z in Zs:
    z_columns.append( "z%d"%z) 

  disease_query_train    = predict_survival_train["disease"].values == disease
  disease_survival_train = predict_survival_train[ disease_query_train ]
  T_train = disease_survival_train["T"].values
  E_train = disease_survival_train["E"].values
  Z_train = disease_survival_train[z_columns].values

  disease_query_test    = predict_survival_test["disease"].values == disease
  disease_survival_test = predict_survival_test[ disease_query_test ]
  T_test = disease_survival_test["T"].values
  E_test = disease_survival_test["E"].values
  Z_test = disease_survival_test[z_columns].values

  
  if len(T_train)==0:
    return None, None, None
  #   kmf.fit(T, event_observed=E, label = disease)
  #   ax=kmf.plot(ax=ax, ci_force_lines=True)
  # else:
  #   continue
  
  kmeans = KMeans(n_clusters=K ).fit(Z_train.astype(float))
  
  f = pp.figure()
  kmf = KaplanMeierFitter()
  ax1 = f.add_subplot(311)
  ax2 = f.add_subplot(312)
  ax3 = f.add_subplot(313)
  
  test_labels = []
  if len(Z_test) > 0:
    test_labels = kmeans.predict( Z_test.astype(float) )
    pdb.set_trace()
    
  for k in range(K):
    I = pp.find( kmeans.labels_==k)
    Ti=T_train[I]
    Ei=E_train[I]
  
    if len(Ti)>0:
      kmf.fit(Ti, event_observed=Ei, label = "train_k=%d"%k)
      ax1=kmf.plot(ax=ax1)
      
    if len(test_labels) > 0:
      I_test = pp.find( test_labels==k)
      Ti_test=T_test[I_test]
      Ei_test=E_test[I_test]
  
      if len(Ti_test)>0:
        kmf.fit(Ti_test, event_observed=Ei_test, label = "test_k=%d"%k)
        ax2=kmf.plot(ax=ax2)
    
      T = np.hstack( (Ti,Ti_test))
      E = np.hstack( (Ei,Ei_test))
      if len(T)>0:
        kmf.fit(T, event_observed=E, label = "all_k=%d"%k)
        ax3=kmf.plot(ax=ax3)
    #pdb.set_trace()
  pp.title("%s"%(disease))
  
  return f, kmf, kmeans
          

  
def kmeans_then_survival( batcher, sess, info ):
  fill_store = batcher.fill_store
  data_store = batcher.data_store
  fill_store.open()
  
  ALL_SURVIVAL = data_store["/CLINICAL/data"][["patient.days_to_last_followup","patient.days_to_death"]]
  tissue_barcodes = np.array( ALL_SURVIVAL.index.tolist(), dtype=str )
  surv_barcodes = np.array([ x+"_"+y for x,y in tissue_barcodes])
  NEW_SURVIVAL = pd.DataFrame( ALL_SURVIVAL.values, index =surv_barcodes, columns = ALL_SURVIVAL.columns ) 
  train_survival = pd.concat( [NEW_SURVIVAL, fill_store["/Z/TRAIN/Z/mu"]], axis=1, join = 'inner' )
  test_survival  = pd.concat( [NEW_SURVIVAL, fill_store["/Z/TEST/Z/mu"]], axis=1, join = 'inner' )
  val_survival  = pd.concat( [NEW_SURVIVAL, fill_store["/Z/VAL/Z/mu"]], axis=1, join = 'inner' )
  
  fill_store.close()
  #-------
  predict_survival_train = train_survival #pd.concat( [test_survival, val_survival], axis=0, join = 'outer' )
  predict_barcodes_train = predict_survival_train.index
  splt = np.array( [ [s.split("_")[0], s.split("_")[1]] for s in predict_barcodes_train ] )
  predict_survival_train = pd.DataFrame( predict_survival_train.values, index = splt[:,1], columns = predict_survival_train.columns )
  predict_survival_train["disease"] = splt[:,0]
  
  Times_train = predict_survival_train[ "patient.days_to_last_followup" ].fillna(0).values.astype(int)+predict_survival_train[ "patient.days_to_death" ].fillna(0).values.astype(int)
  predict_survival_train["T"] = Times_train
  Events_train = (1-np.isnan( predict_survival_train[ "patient.days_to_death" ].astype(float)) ).astype(int)
  predict_survival_train["E"] = Events_train
  #-------
  predict_survival_test = pd.concat( [test_survival, val_survival], axis=0, join = 'outer' )
  predict_barcodes_test = predict_survival_test.index
  splt = np.array( [ [s.split("_")[0], s.split("_")[1]] for s in predict_barcodes_test ] )
  predict_survival_test = pd.DataFrame( predict_survival_test.values, index = splt[:,1], columns = predict_survival_test.columns )
  predict_survival_test["disease"] = splt[:,0]
  
  Times_test = predict_survival_test[ "patient.days_to_last_followup" ].fillna(0).values.astype(int)+predict_survival_test[ "patient.days_to_death" ].fillna(0).values.astype(int)
  predict_survival_test["T"] = Times_test
  Events_test = (1-np.isnan( predict_survival_test[ "patient.days_to_death" ].astype(float)) ).astype(int)
  predict_survival_test["E"] = Events_test  
  z_columns = []
  columns = ["T","E"]
  for zidx in range(20):
    columns.append("z%d"%(zidx))
    z_columns.append("z%d"%(zidx))
    
  #reg_data = pd.DataFrame( predict_survival[columns].values.astype(int), columns=columns)
  
  for disease in batcher.tissue_names:
    f_disease, kmf, kmeans = kmf_split( predict_survival_train, predict_survival_test, K=3, disease = disease, Zs = np.arange(batcher.n_z) )
    if f_disease is not None:
      pp.savefig( batcher.viz_filename_survival + "_%s.png"%(disease), fmt='png', bbox_inches='tight')
    
# def kmf_quantiles( predict_survival, diseases = ["lgg"], z=0, quants = [0.0,0.1,0.9,1.0] ):
#
#   z_columns = ["z%d"%z]
#
#   for disease in diseases: #batcher.tissue_names:
#     f = pp.figure()
#     kmf = KaplanMeierFitter()
#     ax = f.add_subplot(111)
#
#     disease_query = predict_survival["disease"].values == disease
#     disease_survival = predict_survival[ disease_query ]
#
#
#     T = disease_survival["T"].values
#     E = disease_survival["E"].values
#     Z = np.squeeze( disease_survival[z_columns].values )
#     print Z
#
#     iZ = np.argsort(Z)
#     nz = len(iZ)
#     Is = []
#     for a,b in zip(quants[:-1],quants[1:]):
#       Is.append( iZ[ a*nz : b*nz ] )
#
#     if len(T)>0:
#       kmf.fit(T, event_observed=E, label = disease)
#       ax=kmf.plot(ax=ax, ci_force_lines=True)
#     else:
#       continue
#
#     #kmeans = KMeans(n_clusters=K ).fit(Z)
#     for k,I in zip(range(len(Is)),Is):
#       #I = pp.find( kmeans.labels_==k)
#       Ti=T[I]
#       Ei=E[I]
#
#       if len(Ti)>0:
#         kmf.fit(Ti, event_observed=Ei, label = disease + "_q=%d"%k)
#         ax=kmf.plot(ax=ax)
#
# pp.show()
    