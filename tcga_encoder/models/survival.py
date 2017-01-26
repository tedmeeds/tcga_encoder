from tcga_encoder.utils.helpers import *
from lifelines import KaplanMeierFitter
from sklearn.cluster import KMeans, SpectralClustering
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tcga_encoder.models.lda import LinearDiscriminantAnalysis
from sklearn.neighbors import KernelDensity
import pdb

def kmf_kmeans( predict_survival_train, predict_survival_test, K, disease, Zs ):

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
    #pdb.set_trace()
    
  colours = "brgkmcbrgkmcbrgkmcbrgkmcbrgkmcbrgkmcbrgkmc"
  for k in range(K):
    I = pp.find( kmeans.labels_==k)
    Ti=T_train[I]
    Ei=E_train[I]
  
    if len(Ti)>0:
      kmf.fit(Ti, event_observed=Ei, label = "train_k=%d"%k)
      ax1=kmf.plot(ax=ax1, color=colours[k])
      
    if len(test_labels) > 0:
      I_test = pp.find( test_labels==k)
      Ti_test=T_test[I_test]
      Ei_test=E_test[I_test]
  
      if len(Ti_test)>0:
        kmf.fit(Ti_test, event_observed=Ei_test, label = "test_k=%d"%k)
        ax2=kmf.plot(ax=ax2, color=colours[k])
    
      T = np.hstack( (Ti,Ti_test))
      E = np.hstack( (Ei,Ei_test))
      if len(T)>0:
        kmf.fit(T, event_observed=E, label = "all_k=%d"%k)
        ax3=kmf.plot(ax=ax3, color=colours[k])
    #pdb.set_trace()
  pp.suptitle("%s"%(disease))
  
  return f, kmf, kmeans
          

def kmf_spectral( predict_survival_train, predict_survival_test, K, disease, Zs ):

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
  
  kmeans = SpectralClustering(n_clusters=K, n_neighbors=5, affinity='nearest_neighbors' ).fit(Z_train.astype(float))
  
  f = pp.figure()
  kmf = KaplanMeierFitter()
  ax1 = f.add_subplot(311)
  ax2 = f.add_subplot(312)
  ax3 = f.add_subplot(313)
  
  test_labels = []
  if len(Z_test) > 0:
    test_labels = kmeans.predict( Z_test.astype(float) )
    #pdb.set_trace()
    
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
  pp.suptitle("%s"%(disease))
  
  return f, kmf, kmeans
 
def kmf_lda( predict_survival_train, predict_survival_test, K, disease, Zs ):

  z_columns = []
  for z in Zs:
    z_columns.append( "z%d"%z) 

  disease_query_train    = predict_survival_train["disease"].values == disease
  disease_survival_train = predict_survival_train[ disease_query_train ]
  T_train = disease_survival_train["T"].values
  E_train = disease_survival_train["E"].values
  Z_train = disease_survival_train[z_columns].values
  n_train = len(Z_train)
  
  disease_query_test    = predict_survival_test["disease"].values == disease
  disease_survival_test = predict_survival_test[ disease_query_test ]
  T_test = disease_survival_test["T"].values
  E_test = disease_survival_test["E"].values
  Z_test = disease_survival_test[z_columns].values

  group_split1 = np.zeros( n_train, dtype=int )
  group_split2 = np.zeros( n_train, dtype=int )
  
  if len(T_train)==0:
    return None, None, None, None, None
  
  X = Z_train
  y = E_train.astype(int)
  I1 = pp.find(y==1)
  I0 = pp.find(y==0)
  lda = LinearDiscriminantAnalysis()
  lda.fit(X, y)

  f = pp.figure()
  ax1 = f.add_subplot(131)
  predict_train = lda.predict( Z_train, ignore_pi=True )
  group_split1 = np.squeeze( predict_train ).astype(int)
  project_train = lda.transform( Z_train )
  log_joint_train = lda.log_joint( project_train )
  order_train = np.argsort( log_joint_train[1]-log_joint_train[0] )
  
  I1 = pp.find(predict_train==1)
  I0 = pp.find(predict_train==0)
  #pdb.set_trace()
  kmf = KaplanMeierFitter()
  if len(I1) > 0:
    kmf.fit(T_train[I1], event_observed=E_train[I1], label =  "lda_1 E=%d C=%d"%(E_train[I1].sum(),len(I1)-E_train[I1].sum()))
    ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='red')
  if len(I0) > 0:
    kmf.fit(T_train[I0], event_observed=E_train[I0], label = "lda_0 E=%d C=%d"%(E_train[I0].sum(),len(I0)-E_train[I0].sum()))
    ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='blue')
   
  colors = ["blue", "green", "orange", "red"] #"bgor"
  n = len(Z_train)
  n_chunks = 4
  ax2 = f.add_subplot(132)
  kmf2 = KaplanMeierFitter()
  q_idx=0
  for ids in chunks( np.arange(n,dtype=int), int(1+float(n)/4) ):
    these_ids = order_train[ids]
    kmf2.fit(T_train[these_ids], event_observed=E_train[these_ids], label =  "lda q %d E=%d C=%d"%(q_idx,E_train[these_ids].sum(),len(these_ids)-E_train[these_ids].sum()))
    ax2=kmf2.plot(ax=ax2,at_risk_counts=False,show_censors=True, color= colors[q_idx])
    group_split2[these_ids] = q_idx
    q_idx+=1
    
    
    
  ax3 = f.add_subplot(133)
  x_plot = np.linspace( min(np.min(lda.x_proj1),np.min(lda.x_proj0)), max(np.max(lda.x_proj1),np.max(lda.x_proj0)), 500) 
  lda.plot_joint_density( x_plot, ax=ax3, ignore_pi=True )
  ax3.legend()
    
  return f, kmf, lda, group_split1, group_split2
           
          
  
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
    f_disease, kmf, kmeans = kmf_kmeans( predict_survival_train, predict_survival_test, K=2, disease = disease, Zs = np.arange(batcher.n_z) )
    #f_disease, kmf, kmeans = kmf_spectral( predict_survival_train, predict_survival_test, K=3, disease = disease, Zs = np.arange(batcher.n_z) )
    if f_disease is not None:
      pp.savefig( batcher.viz_filename_survival + "_%s.png"%(disease), fmt='png')

def lda_then_survival( batcher, sess, info ):
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
    f_disease, kmf, kmeans, g1, g2 = kmf_lda( predict_survival_train, predict_survival_test, K=3, disease = disease, Zs = np.arange(batcher.n_z) )

    
    if f_disease is not None:
      batcher.SaveSurvival( disease, predict_survival_train, g1, g2 )
      pp.savefig( batcher.viz_filename_survival_lda + "_%s.png"%(disease), fmt='png')
      pp.close('all')
          
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

def analyze_survival_store( store, disease, split ):
  s = store["%s/%s"%(disease, split)]
  
  genes = s.columns
  vals  = s.values
  
  some_mutations = pp.find( vals.sum(0) )
  
  vals  = vals[:,some_mutations]
  genes = genes[some_mutations]
  
  order = np.argsort( -vals.sum(0) )
  vals = vals[:,order]
  genes = genes[order]
  s2 = pd.DataFrame( vals, columns = genes, index=s.index)
  
  grouped = s2.groupby(level=1)
  means = []
  for name, group in grouped:
    print name, group.sum().T[:10]
    means.append( group.values.mean(0) )
  
  dif = means[-1] - means[0]
  
  order2 = np.argsort( -np.abs(dif) )
  for g,d in zip( genes[order2][:10], dif[order2][:10] ):
    print g, d
    
    
  grouped.sum().T.plot()
  
  

  
  
    