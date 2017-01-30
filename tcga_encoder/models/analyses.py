from tcga_encoder.utils.helpers import *
from tcga_encoder.definitions.locations import *
from lifelines import KaplanMeierFitter
import sklearn
from sklearn.cluster import KMeans, SpectralClustering
from tcga_encoder.models.lda import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA2
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
    #Xn = X - X.mean(0)
    #Xn /= Xn.std(0)
    y = train_dna_data[:,gene_idx].astype(int)
    lda = LinearDiscriminantAnalysis(epsilon=1.0)
    lda.fit(X, y)
    
    lda2 = LDA2()
    lda2.fit(X,y)
    
    predict_train = lda.predict( X, ignore_pi=True )
    proj_train2 = lda.transform( X  )
    proj_train = lda2.transform( X  )
    
    prob_train = lda.prob( X, ignore_pi=True)
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
        auc = roc_auc_score( y, prob_train)
        #proj_train
        #kl( m1, v1, m2, v2 )
        log_prob = lda.log_prob_1( X[I,:], ignore_pi=True).mean()
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
      
# def old():
#   -    x_proj_train_1 = lda.transform( Z_mutation )
#    -    x_proj_train_0 = lda.transform( Z_free )
#    -
#    -    h1 = 0.001+np.std(x_proj_train_1)*(4.0/3.0/len(I))**(1.0/5.0)
#    -    h0 =  0.001+np.std(x_proj_train_0)*(4.0/3.0/len(J))**(1.0/5.0)
#    -
#    -    x_left  = -1.0 + min( min(x_proj_train_1), min( x_proj_train_0) )
#    -    x_right = 1.0 + max( max(x_proj_train_1), max( x_proj_train_0) )
#    -    x_plot = np.linspace( x_left, x_right, 100 ).reshape( (100,1) )
#    -
#    -    kde1 = KernelDensity(kernel='gaussian', bandwidth=h1).fit(x_proj_train_1)
#    -    kde2 = KernelDensity(kernel='gaussian', bandwidth=h0).fit(x_proj_train_0)
#    -    log_dens1 = kde1.score_samples(x_plot)
#    -    log_dens2 = kde2.score_samples(x_plot)

def compress_survival_prediction( disease, data_h5, survival_h5, K = 10, penalty = "l2", C = 1.0 ):
  d = pd.HDFStore( data_h5, "r" )
  s = pd.HDFStore( survival_h5, "r" )
  S1 = s["/%s/split1"%(disease)]
  S2 = s["/%s/split2"%(disease)]
  
  # need to add disease in front of s barcodes
  d_bcs = [ "%s_%s"%(disease,bc) for bc in S1.index.levels[0]]
  bcs   = S1.index.levels[0]
  d_bc2_bc = OrderedDict()
  for d_bc, bc in zip( d_bcs, bcs ):
    d_bc2_bc[ d_bc ] = bc
  
  observed = d["/CLINICAL/observed"].loc[d_bcs]
  dna  = d["/DNA/channel/0"].loc[d_bcs]
  rna  = d["/RNA/FAIR"].loc[d_bcs]
  meth = d["/METH/FAIR"].loc[d_bcs]
  
  dna_observed = observed["DNA"].values > 0
  dna_observed_d_bcs = observed[ dna_observed ].index

  rna_observed = observed["RNA"].values > 0
  rna_observed_d_bcs = observed[ rna_observed ].index
  
  meth_observed = observed["METH"].values > 0
  meth_observed_d_bcs = observed[ meth_observed ].index
  
  dna_for_training  = dna.loc[ dna_observed_d_bcs ]
  rna_for_training  = rna.loc[ rna_observed_d_bcs ]
  meth_for_training = meth.loc[ meth_observed_d_bcs ]
  
  
  # K-Fold xval
  data_for_training = dna_for_training
  d_bcs_for_training = np.array( dna_observed_d_bcs, dtype=str)
  #data_for_training = rna_for_training
  #d_bcs_for_training = np.array( rna_observed_d_bcs, dtype=str)
  #data_for_training = meth_for_training
  #d_bcs_for_training = np.array( meth_observed_d_bcs, dtype=str)
  
  
  feature_names = data_for_training.columns
  split_index = S1.index
  #pdb.set_trace()
  S = pd.DataFrame( split_index.get_level_values(1).values.astype(int), columns=["group"], index = split_index.get_level_values(0).values.astype(str) )
  
  splits_for_training = S
  bcs_for_training = np.array( [d_bc2_bc[d_bc] for d_bc in d_bcs_for_training], dtype=str )
  n   = len(bcs_for_training)
  #pdb.set_trace()
  y = np.squeeze( splits_for_training.loc[bcs_for_training].values )
  test_prob = np.zeros( n, dtype = float )
  test_predictions = np.zeros( n, dtype = int )
  models = []
  for subset_ids in chunks( np.arange(n,dtype=int), int(1+float(n)/K) ):
    d_bc_subset = d_bcs_for_training[subset_ids]
    bc_subset   = bcs_for_training[subset_ids]
    
    #pdb.set_trace()
    train_d_bcs = np.setdiff1d( d_bcs_for_training, d_bc_subset )
    test_d_bcs  = d_bc_subset
    
    #pdb.set_trace()
    train_bcs = np.setdiff1d( bcs_for_training, bc_subset )
    test_bcs  = bc_subset
    
    train_x = data_for_training.loc[train_d_bcs].values
    test_x  = data_for_training.loc[test_d_bcs].values
    
    mn = train_x.mean(0)
    st = train_x.std(0)
    
    #train_x -= mn; #train_x /= st
    #test_x -= mn; #test_x /= st
    
    train_y = np.squeeze( splits_for_training.loc[train_bcs].values )
    test_y  = np.squeeze( splits_for_training.loc[test_bcs].values )
    #from sklearn.svm import LinearSVC
    #model = sklearn.svm.LinearSVC(penalty=penalty,C=C, intercept_scaling=2.0, fit_intercept=True)
    model = sklearn.linear_model.LogisticRegression(penalty=penalty,C=C, intercept_scaling=2.0, fit_intercept=False)
    model.fit( train_x, train_y )
    
    predict_train = model.predict( train_x )
    predict_test  = model.predict( test_x )
    predict_proba  = model.predict_proba( test_x )
    
    test_prob[subset_ids]          = predict_proba
    test_predictions[ subset_ids ] = predict_test
    y[ subset_ids ] = test_y
    models.append( model )
    
  
  test_log_prob = np.mean( y*np.log( test_prob +1e-12) + (1-y)*np.log(1.0-test_prob+1e-12) ) 
  test_accuracy = np.mean( test_predictions == y )
  test_auc = roc_auc_score( y, test_prob)
  #print "%s %d-fold accuracy = %0.3f"%( disease, K, test_accuracy )
  
  #pdb.set_trace()
  
  return test_accuracy, models, y, test_predictions, test_prob, test_log_prob, test_auc
    
    
if __name__ == "__main__":
  
  disease = "sarc"
  data_file = "pan_tiny_multi_set"
  experiment_name = "tiny_leave_blca_out"
  
  if len(sys.argv) == 4:
    disease   = sys.argv[1]
    data_file = sys.argv[2]
    experiment_name = sys.argv[3]
  
  data_location = os.path.join( HOME_DIR, "data/broad_processed_post_recomb/20160128/%s/data.h5"%(data_file) )
  survival_location = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/%s/full_vae_survival.h5"%(experiment_name) )
  
  #s = pd.HDFStore( survival_file, "r" )
  #S1 = s["/%s/split1"%(disease)]
  f = pp.figure()
  ax1 = f.add_subplot(211)
  ax2 = f.add_subplot(212)
  K=20
  penalties = ["l2","l1"]
  Css = [[ 1.0,0.9,0.75,0.5,0.1,0.01, 0.001, 0.0001],[5.0,2.0,1.0]]
  best_values = OrderedDict()
  mn_models = OrderedDict() 
  axs = [ax1,ax2]
  for penalty_idx, penalty,Cs in zip( range(2), penalties,Css ):
    best_values[ penalty ] = []
    for C  in Cs:
      test_accuracy, models, y, test_predictions, test_prob, test_log_prob, test_auc  = compress_survival_prediction( disease, data_location, survival_location, K, penalty, C )
      print "%s %d-fold auc = %0.3f accuracy = %0.3f, log prob = %0.3f (C = %f, reg = %s)"%( disease, K, test_auc, test_accuracy, test_log_prob, C, penalty )
      best_values[ penalty ].append([test_accuracy,test_log_prob,test_auc])
        
    best_values[ penalty ] = np.array(best_values[ penalty ], dtype=float )
    
    best_idx = np.argmin( best_values[ penalty ][:,0] )
    
    best_C = Cs[ best_idx ]
    
    test_accuracy, models, y, test_predictions, test_prob, test_log_prob, test_auc  = compress_survival_prediction( disease, data_location, survival_location, K, penalty, best_C )
    
    mn_models[ penalty ] = np.zeros(models[0].coef_.shape[1])  
    for m in models:
      axs[penalty_idx].plot( m.coef_.T, 'o-' )
      mn_models[ penalty ] += np.squeeze(m.coef_.T)
  pp.show()
          
    
      
  




   
    