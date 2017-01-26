from tcga_encoder.utils.helpers import *
from lifelines import KaplanMeierFitter
from sklearn.cluster import KMeans, SpectralClustering
from tcga_encoder.models.lda import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KernelDensity
from tcga_encoder.utils.math_funcs import kl

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
          
          
  
def lda_on_mutations( batcher, sess, info ):
  train_barcodes = batcher.train_barcodes
  test_barcodes  = batcher.test_barcodes
  val_barcodes   = batcher.validation_barcodes
  
  fill_store = batcher.fill_store
  data_store = batcher.data_store
  #fill_store.open()
  #dna_genes = ["APC","TP53"]
  #dna_genes = data_store[batcher.DNA_keys[0]].columns[:40]
  dna_genes = ["APC", "RB1", "PIK3CA"]
  #pdb.set_trace()
  n_dna = len(dna_genes)
  train_dna_data = np.zeros( (len(train_barcodes),n_dna) )
  test_dna_data  = np.zeros( (len(test_barcodes),n_dna) )
  val_dna_data   = np.zeros( (len(val_barcodes),n_dna) )
  for idx,DNA_key in zip(range(len(batcher.DNA_keys)),batcher.DNA_keys):
    batch_data = data_store[DNA_key][dna_genes].loc[ train_barcodes ].fillna( 0 ).values
    train_dna_data += batch_data
    
    batch_data = data_store[DNA_key][dna_genes].loc[ test_barcodes ].fillna( 0 ).values
    test_dna_data += batch_data
    
    batch_data = data_store[DNA_key][dna_genes].loc[ val_barcodes ].fillna( 0 ).values
    val_dna_data += batch_data
  
  fill_store.open()  
  train_Z = fill_store["/Z/TRAIN/Z/mu"].values.astype(float)
  test_Z  = fill_store["/Z/TEST/Z/mu"].values.astype(float)
  val_Z   = fill_store["/Z/VAL/Z/mu"].values.astype(float)
  #pdb.set_trace()
  fill_store.close()
  
  #-------
  aucs = OrderedDict()
  for gene, gene_idx in zip( dna_genes, range(n_dna) ):
    #pdb.set_trace()
    I = pp.find( train_dna_data[:,gene_idx] > 0 )
    J = pp.find( train_dna_data[:,gene_idx] == 0 )
    if len(I) == 0:
      continue
    Z_mutation = train_Z[I,:]
    Z_free     = train_Z[J,:]
    
    X = train_Z
    y = train_dna_data[:,gene_idx].astype(int)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    predict_train = lda.predict( X, ignore_pi=True )
    proj_train = lda.transform( X  )
    log_prob_train = lda.log_prob_1( X[I,:], ignore_pi=True)
    #pdb.set_trace()
    f = pp.figure()
    ax = f.add_subplot(111)
    x_plot = np.linspace( min(np.min(lda.x_proj1),np.min(lda.x_proj0)), max(np.max(lda.x_proj1),np.max(lda.x_proj0)), 500) 
    lda.plot_joint_density( x_plot, ax=ax, ignore_pi=True )
    #pdb.set_trace()
    ax.legend()
    #pdb.set_trace()
    #pp.hist( x_proj_train_0, alpha=0.5 )
    pp.savefig( batcher.viz_filename_z_to_dna + "_%s.png"%(gene), fmt='png', bbox_inches='tight')
    pp.close('all')
    aucs[gene] = OrderedDict()
    
    m1 = proj_train[I].mean()
    v1 = proj_train[I].var()
    for predict_gene in data_store[batcher.DNA_keys[0]].columns:
      y = data_store[batcher.DNA_keys[0]][predict_gene].loc[ train_barcodes ].fillna( 0 ).values.astype(int)
      if len(np.unique(y))==2 and y.sum()>10:
        I = pp.find(y)
        m2 = proj_train[I].mean()
        v2 = proj_train[I].var()
        #auc = roc_auc_score( y, prob_train)
        #proj_train
        #kl( m1, v1, m2, v2 )
        #log_prob = lda.log_prob_1( X[I,:], ignore_pi=True).mean()
        aucs[gene][ predict_gene ] = -kl( m1, v1, m2, v2 ) #log_prob
    
    genes = np.array( aucs[gene].keys(), dtype=str )
    vals  = np.array( aucs[gene].values(), dtype=float )
    closest = np.argsort( - vals )
    print "-------------------------"
    print "-------------------------"
    print "%s projection for %10s has log prob = %0.3f"%(gene, gene, aucs[gene][gene])
    print "-------------------------"
    for idx in range(20):
      print "%s projection for %10s has log prob = %0.3f"%(gene, genes[closest[idx]], vals[closest[idx]])
  
  print "========================="
  #pdb.set_trace()
      

    