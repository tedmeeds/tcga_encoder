from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.analyses.everything_functions import *
from tcga_encoder.analyses.survival_functions import *
import networkx as nx
#try:
from networkx.drawing.nx_agraph import graphviz_layout as g_layout

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import squareform
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
#except:


def deeper_meaning_dna_and_z_correct( data, K=10, min_p_value=1e-3, threshold = 0.01, Cs = [0.00001, 0.001,0.1,10.0,1000.0] ):
  save_dir   = os.path.join( data.save_dir, "correct_deeper_meaning_dna_and_z_p_tissue_%0.2f_p_spear_%g_logreg"%(threshold,min_p_value) )
  check_and_mkdir(save_dir) 
  n_C = len(Cs)
  dna_auc_dir   = os.path.join( data.save_dir, "dna_auc_latent" )
  spearmanr_dir = os.path.join( data.save_dir, "spearmans_latent_tissue" )
  
  # spear_dna_z_rho = pd.read_csv( spearmanr_dir + "/dna_z_rho.csv", index_col="gene" )
  # spear_dna_z_p   = pd.read_csv( spearmanr_dir + "/dna_z_p.csv", index_col="gene" )
  # dna_z_auc     = pd.read_csv( dna_auc_dir + "/dna_z_auc.csv", index_col="gene" )
  # dna_z_auc_p   = pd.read_csv( dna_auc_dir + "/dna_z_auc_p.csv", index_col="gene" )
  # dna_z_p = dna_z_auc_p
  #
  RNA_scale   = 2.0 / (1+np.exp(-data.RNA_scale)) -1   
  miRNA_scale = 2.0 / (1+np.exp(-data.miRNA_scale))-1 
  METH_scale  = 2.0 / (1+np.exp(-data.METH_scale ))-1  
  
  n_rna   = data.n_rna
  n_mirna = data.n_mirna
  n_meth  = data.n_meth
  
  rna_z_rho = pd.read_csv( spearmanr_dir + "/rna_z_rho.csv", index_col="gene" )
  # rna_z_p   = pd.read_csv( spearmanr_dir + "/rna_z_p.csv", index_col="gene" )
  #
  mirna_z_rho = pd.read_csv( spearmanr_dir + "/mirna_z_rho.csv", index_col="gene" )
  # mirna_z_p   = pd.read_csv( spearmanr_dir + "/mirna_z_p.csv", index_col="gene" )
  #
  meth_z_rho = pd.read_csv( spearmanr_dir + "/meth_z_rho.csv", index_col="gene" )
  # meth_z_p   = pd.read_csv( spearmanr_dir + "/meth_z_p.csv", index_col="gene" )
  
  Z           = data.Z
  T           = data.T
  barcodes    = Z.index.values
  z_names     = data.z_names.values
  n_z         = len(z_names)
  
  dna_names = data.dna.sum(0).sort_values(ascending=False).index.values
  n_dna = len(dna_names)
  dna = data.dna[ dna_names ]
  
  #z_scores = -np.sum( np.log2(dna_z_p),1)
  
  #f=pp.figure( figsize=(24,12) )
  #genes = z_scores.sort_values().index.values #dna_names[:nbr_genes]
  #pdb.set_trace()
  #min_p_value = 1e-3
  k_idx = 1
  results = []
  random_state=0
  Z_W = []
  performances = []
  results = []
  for gene in dna_names:
    bad_gene = False
    dna_values = dna[gene].values
    ids_with_n, relevant_tissues = ids_with_at_least_p_mutations( dna_values, T, p = threshold )
    
    if len(ids_with_n)==0:
      print "skipping ",gene
      continue    
    
    gene_bcs = barcodes[ ids_with_n ]
    
    
    
    k = 0
    
    y_true = dna_values[ids_with_n]
    y_est = np.zeros( ( len(y_true),n_C) )
    X = Z[ ids_with_n ].values
    assert len(y_true) == len(X), "should be same"
    c=Counter()
    #best_ridge = 10.001
    Z_weights = np.zeros( (n_C,K,n_z) )
    Z_counts  = np.zeros( (K,n_z) )
    
    K_use = min( y_true.sum(),K)
    folds = StratifiedKFold(n_splits=K_use, shuffle = True, random_state=random_state)
    for train_split, test_split in folds.split( X, y_true ):
      y_train = y_true[train_split]; X_train = X[train_split,:]
      y_test = y_true[test_split];   
      
      # find top z by spearmanr p-values
      rho_dna_z = pearsonr( 2*y_train-1, X_train ) 
      #rho_dna_z = stats.spearmanr(y_train, X_train )
      dna_z_rho = np.squeeze(rho_dna_z[0])
      dna_z_p   = np.squeeze(rho_dna_z[1])
      #dna_z_rho = np.squeeze(rho_dna_z[0][:1,:][:,1:])
      #dna_z_p   = np.squeeze(rho_dna_z[1][:1,:][:,1:])
      
      #ok_dna_z_p = pp.find( dna_z_p<min_p_value )
      ok_dna_z_p = pp.find( dna_z_p < min_p_value )
      
      #pdb.set_trace()
      I=np.argsort( ok_dna_z_p )
      if len(I) == 0:
        bad_gene = True
        print "SKIPPING ", gene
        continue
      if len(I)>40:
        I=I[:40]
      print "  using %d zs"%(len(I))
      #pdb.set_trace()
      c.update(z_names[np.sort(ok_dna_z_p[I])])
      #print gene, k, z_names[np.sort(ok_dna_z_p[I])]
      
      z_ids_k = ok_dna_z_p[I]
      X_train = X[train_split,:][:,z_ids_k]
      X_test = X[test_split,:][:,z_ids_k]
      Z_counts[k,z_ids_k] = 1.0
      M=LogisticBinaryClassifier()
      
      for C_idx, C in zip(range(n_C),Cs):
        M.fit( y_train, X_train, C=C )
        y_est[test_split,C_idx] = M.prob(X_test)
        
        weights = np.zeros(n_z)
        #pdb.set_trace()
         
        Z_weights[C_idx,k,z_ids_k] = M.coef_
      k+=1
    if bad_gene is True:
      continue
    aucs = np.zeros(n_C)
    aucs_p = np.zeros(n_C)
    for C_idx, C in zip(range(n_C),Cs):
      auc_y_est, p_value_y_est = auc_and_pvalue( y_true, y_est[:,C_idx] )
      aucs[C_idx] = auc_y_est; aucs_p[C_idx] = p_value_y_est
    best_c_idx = np.argmax(aucs)
    best_c = Cs[best_c_idx]
    y_est_best = y_est[:,best_c_idx]
    best_auc = aucs[best_c_idx]
    best_auc_p = aucs_p[best_c_idx]
    best_Z_weights = Z_weights[best_c_idx]
    
    z_w = pd.Series( best_Z_weights.sum(0) / Z_counts.sum(0), index = z_names, name = gene )
    
    
    bad = np.isnan(z_w); z_w[bad]=0
    #pdb.set_trace()
    order_y_est = np.argsort(y_est_best)
    y_est_best_pd = pd.DataFrame( np.vstack((y_true[order_y_est],y_est_best[order_y_est])).T, index = gene_bcs[order_y_est], columns=["true","est"] )
    performance = pd.Series( [best_auc,best_auc_p], index = ["AUC","p-value"], name = gene )
    z_w = pd.Series( z_w, index = z_names, name = gene )
    
    performances.append(performance)
    Z_W.append(z_w)
    
    
    print gene, "AUC/p", best_auc,  best_auc_p, best_c
    #print gene, "Z", [cj[0] for cj in c.most_common()]
    #print gene, "C", [cj[1] for cj in c.most_common()]
    
    gene_dir =  os.path.join( save_dir, "%s"%(gene) )
    check_and_mkdir(gene_dir)
    
    y_est_best_pd.to_csv( gene_dir +"/y_est_best.csv",index_label="barcode" )
    performance.to_csv( gene_dir +"/performance.csv" )
    z_w.to_csv( gene_dir +"/z_w.csv" )
    
    w_rna   = pd.Series( np.dot( rna_z_rho, z_w ), index = rna_z_rho.index, name = gene ).sort_values(ascending=False)
    w_mirna = pd.Series( np.dot( mirna_z_rho, z_w ), index = mirna_z_rho.index, name = gene ).sort_values(ascending=False)
    w_meth  = pd.Series( np.dot( meth_z_rho, z_w ), index = meth_z_rho.index, name = gene ).sort_values(ascending=False)
    
    w_rna.to_csv( gene_dir +"/w_rna.csv" )
    w_mirna.to_csv( gene_dir +"/w_mirna.csv" )
    w_meth.to_csv( gene_dir +"/w_meth.csv" )
    
    print "  computing spearmans"
    #rho_rna_z = stats.spearmanr( RNA_scale.values[ids_with_n,:], y_est_best )
    #rho_mirna_z = stats.spearmanr( miRNA_scale.values[ids_with_n,:],y_est_best)
    #rho_meth_z = stats.spearmanr( METH_scale.values[ids_with_n,:], y_est_best)
    
    rho_rna_z = pearsonr( RNA_scale.values[ids_with_n,:], y_est_best )
    rho_mirna_z = pearsonr( miRNA_scale.values[ids_with_n,:],y_est_best)
    rho_meth_z = pearsonr( METH_scale.values[ids_with_n,:], y_est_best)
      
    print "  organizing"
    #pdb.set_trace()
    # rna_z_rho_gene = pd.Series( np.squeeze( rho_rna_z[0][:n_rna,:][:,n_rna:] ), index = data.rna_names, name = gene)
    # rna_z_p_gene   = pd.Series( np.squeeze( rho_rna_z[1][:n_rna,:][:,n_rna:] ), index = data.rna_names, name = gene)
    #
    # mirna_z_rho_gene = pd.Series( np.squeeze( rho_mirna_z[0][:n_mirna,:][:,n_mirna:] ), index = data.mirna_names, name = gene)
    # mirna_z_p_gene   = pd.Series( np.squeeze( rho_mirna_z[1][:n_mirna,:][:,n_mirna:] ), index = data.mirna_names, name = gene)
    #
    # meth_z_rho_gene = pd.Series( np.squeeze( rho_meth_z[0][:n_meth,:][:,n_meth:] ), index = data.meth_names, name = gene)
    # meth_z_p_gene   = pd.Series( np.squeeze( rho_meth_z[1][:n_meth,:][:,n_meth:] ), index = data.meth_names, name = gene)

    rna_z_rho_gene = pd.Series( np.squeeze( rho_rna_z[0] ), index = data.rna_names, name = gene)
    rna_z_p_gene   = pd.Series( np.squeeze( rho_rna_z[1] ), index = data.rna_names, name = gene)
  
    mirna_z_rho_gene = pd.Series( np.squeeze( rho_mirna_z[0] ), index = data.mirna_names, name = gene)
    mirna_z_p_gene   = pd.Series( np.squeeze( rho_mirna_z[1] ), index = data.mirna_names, name = gene)
   
    meth_z_rho_gene = pd.Series( np.squeeze( rho_meth_z[0] ), index = data.meth_names, name = gene)
    meth_z_p_gene   = pd.Series( np.squeeze( rho_meth_z[1] ), index = data.meth_names, name = gene)

    
    rna_z_p_gene = rna_z_p_gene.sort_values()
    mirna_z_p_gene = mirna_z_p_gene.sort_values()
    meth_z_p_gene = meth_z_p_gene.sort_values()
    
    rna_z_rho_gene   = rna_z_rho_gene.loc[ rna_z_p_gene.index ]
    mirna_z_rho_gene = mirna_z_rho_gene.loc[ mirna_z_p_gene.index ]
    meth_z_rho_gene  = meth_z_rho_gene.loc[ meth_z_p_gene.index ]
    
    rna_z_rho_gene.to_csv( gene_dir +"/y_rna_rho.csv" )
    mirna_z_rho_gene.to_csv( gene_dir +"/y_mirna_rho.csv" )
    meth_z_rho_gene.to_csv( gene_dir +"/y_meth_rho.csv" )
    rna_z_p_gene.to_csv( gene_dir +"/y_rna_p.csv" )
    mirna_z_p_gene.to_csv( gene_dir +"/y_mirna_p.csv" )
    meth_z_p_gene.to_csv( gene_dir +"/y_meth_p.csv" )
    
    results.append( {"a_dna":gene,"cv": [float(best_auc),  float(best_auc_p)],\
                    "c_wildtype":len(y_true),  "c_mutations":int(np.sum(y_true)),\
                    "tissues":relevant_tissues,\
                    "rna_y":list(rna_z_p_gene[:20].index.values) ,\
                    "rna_z":list( np.abs(w_rna).sort_values(ascending=False).index.values[:20]),\
                    "mirna_y":list(mirna_z_p_gene[:20].index.values) ,\
                    "mirna_z":list( np.abs(w_mirna).sort_values(ascending=False).index.values[:20]),\
                    "meth_y":list(meth_z_p_gene[:20].index.values) ,\
                    "meth_z":list( np.abs(w_meth).sort_values(ascending=False).index.values[:20]),\
                    "top_z":list(np.abs(z_w).sort_values(ascending=False).index.values[:20]) } )
    #pdb.set_trace()
   
  Z_W_concat = pd.concat( Z_W, axis=1 ).T
  performances_concat = pd.concat( performances, axis=1 ).T 
  
  order_ = performances_concat["AUC"].sort_values(ascending=False).index.values
  
  Z_W_concat.loc[order_].to_csv( save_dir +"/z_w.csv",index_label="gene" )
  performances_concat.loc[order_].to_csv( save_dir +"/performances.csv",index_label="gene" )
  fptr = open( save_dir + "/pan_cancer_dna.yaml","w+" )
  fptr.write( yaml.dump(results))
  fptr.close()
  #pdb.set_trace()
  #
  #
  #   p_values = dna_z_p.loc[gene][ dna_z_p.loc[gene] < min_p_value ]
  #
  #
  #
  #   nbr_cancer_types = len(relevant_tissues)
  #   mutations = pp.find( dna_values[ids_with_n] == 1)
  #   wildtype = pp.find( dna_values[ids_with_n]==0)
  #
  #   if len(mutations)==0:
  #     continue
  #
  #   best_z_names = p_values.sort_values().index.values
  #   if len(best_z_names) < 5:
  #     best_z_names = dna_z_p.loc[gene].sort_values()[:5].index.values
  #   elif len(best_z_names) > 20:
  #     best_z_names = best_z_names[:20]
  #   best_z_rna_z_p = rna_z_p[ best_z_names ]
  #   best_z_rna_z_rho = rna_z_rho[ best_z_names ]
  #   print "================"
  #   best_z_score_rna = -np.sum(np.log2(best_z_rna_z_p+1e-200),1)
  #   print best_z_score_rna.sort_values(ascending=False)[:20].index.values
  #   # for z_name in best_z_names:
  #   #   print best_z_rna_z_p[ z_name ].sort_values()[:10]
  #   #   print (-np.abs(best_z_rna_z_rho[ z_name ])).sort_values()[:10]
  #     #pdb.set_trace()
  #   print gene, p_values.sort_values().index.values
  #
  #   y_true = dna_values[ids_with_n]
  #   X = Z[ids_with_n][best_z_names].values
  #
  #   #MCV = GenerativeBinaryClassifierKFold( K = 10 )
  #   MCV = LogisticBinaryClassifierKFold( K = 10 )
  #
  #   best_auc = -np.inf
  #   best_ridge = 0.0
  #   best_y_est = None
  #   best_auc_p = -np.inf
  #   for ridge in ridges:
  #
  #     #y_est_cv = MCV.fit_and_prob( y_true, X, ridge=ridge, cov_type="shared" )
  #     y_est_cv = MCV.fit_and_prob( y_true, X, C=ridge )
  #     #if gene == "BRAF":
  #     #  pdb.set_trace()
  #     auc_y_est_cv, p_value_y_est_cv = auc_and_pvalue( y_true, y_est_cv )
  #     print "for ridge in ridges ",ridge, auc_y_est_cv
  #     if auc_y_est_cv > best_auc:
  #       best_auc = auc_y_est_cv
  #       best_auc_p = p_value_y_est_cv
  #       best_ridge = ridge
  #       best_y_est = y_est_cv
  #
  #   y_est_cv = best_y_est
  #   auc_y_est_cv, p_value_y_est_cv =   best_auc, best_auc_p
  #
  #   # M=LogisticBinaryClassifier()
  #   # #M = GenerativeBinaryClassifier()
  #   # M.fit( y_true, X, C=best_ridge )
  #   # #M.fit( y_true, X, ridge=best_ridge )
  #   # y_est = M.prob(X)
  #   # auc_y_est, p_value_y_est = auc_and_pvalue( y_true, y_est )
  #
  #
  #   #print "learned auc (tr) = %0.3f  p-value: %0.f"%(auc_y_est, p_value_y_est)
  #   print "learned auc (cv) = %0.3f  p-value: %0.f"%(auc_y_est_cv, p_value_y_est_cv)
  #   print "compare to:"
  #   other_aucs = []
  #   signs = []
  #   for z_name in best_z_names:
  #     print z_name, dna_z_auc[z_name].loc[gene], dna_z_auc_p[z_name].loc[gene]
  #     other_aucs.append( max( float(dna_z_auc[z_name].loc[gene]), float(1-dna_z_auc[z_name].loc[gene])) )
  #     if dna_z_auc[z_name].loc[gene]<0.5:
  #       signs.append(-1)
  #     else:
  #       signs.append(1)
  #   signs = np.array(signs)[np.newaxis,:]
  #   #pdb.set_trace()
  #   print "================"
  #   results.append( [gene, {"tissues":relevant_tissues, "nbr_tissues":nbr_cancer_types,"wildtype":len(wildtype),\
  #                   "mutations":len(mutations)},list(p_values.sort_values().index.values),\
  #                   list(best_z_score_rna.sort_values(ascending=False)[:20].index.values),\
  #                    #["train", float(auc_y_est), float(p_value_y_est)], \
  #                    ["cv", float(auc_y_est_cv), float(p_value_y_est_cv)],\
  #                    other_aucs ] )
  #
  #
  #   if len(best_z_names)>5:
  #     gene_dir =  os.path.join( save_dir, "%s"%(gene) )
  #     check_and_mkdir(gene_dir)
  #     y_order = np.argsort(y_est_cv)
  #
  #     #signs = []
  #     X_order = X[y_order,:]
  #     X_order -= X_order.mean(0)
  #     X_order /= X_order.std(0)
  #     X_order *=signs
  #     Z_order = pd.DataFrame( X_order, index = Z[ids_with_n].index.values[y_order], columns = best_z_names )
  #
  #     #k_colors = np.array([k_pallette[kmeans_patients_labels[i]] for i in order_labels] )
  #
  #     print "MAKNING ", gene
  #     size1=12
  #     size2=4
  #     r = size1/size2
  #     f = pp.figure( figsize=(size1,size2))
  #     ax = f.add_subplot(111)
  #     ax.imshow(Z_order.T, cmap='rainbow', interpolation='nearest', aspect=float(len(y_true))/(len(best_z_names)*r))
  #     #ax.imshow(differ)
  #     ax.autoscale(False)
  #
  #     #h = sns.heatmap( Z_order, row_colors=None, row_cluster=False, col_cluster=False, figsize=(12,12) )
  #     #pdb.set_trace()
  #     #pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  #     #pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  #     #pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  #     #pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  #     #h.ax_row_dendrogram.set_visible(False)
  #     #h.ax_col_dendrogram.set_visible(False)
  #     #h.cax.set_visible(False)
  #   #h.ax_heatmap.hlines(len(kmeans_patients_labels)-pp.find(np.diff(np.array(kmeans_patients_labels)[order_labels]))-1, *h.ax_heatmap.get_xlim(), color="black", lw=5)
  #   #h.ax_heatmap.vlines(pp.find(np.diff(np.array(kmeans_z_labels)[order_labels_z]))+1, *h.ax_heatmap.get_ylim(), color="black", lw=5)
  #
  #     f.savefig( gene_dir + "/sorted_by_%s.png"%(gene), fmt="png", bbox_inches='tight')
  #     pp.close('all')
  #
  #
  #
  # check_and_mkdir(save_dir)
  #
  # fptr = open( save_dir + "/pan_cancer_dna.yaml","w+" )
  # fptr.write( yaml.dump(results))
  # fptr.close()