from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.analyses.everything_functions import *
from tcga_encoder.analyses.survival_functions import *
import networkx as nx
#try:
from networkx.drawing.nx_agraph import graphviz_layout as g_layout
from sklearn.metrics import average_precision_score, precision_recall_curve 
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

def tanh(x):
  return (1.0-np.exp(-2*x))/(1.0+np.exp(-2*x))

def deeper_meaning_dna_and_z_correct( data, nbr_dna_genes2process = 100,K=10, min_p_value=1e-3, threshold = 0.01, Cs = [0.00001, 0.001,0.1,10.0,1000.0] ):
  save_dir   = os.path.join( data.save_dir, "correct_deeper_meaning_dna_and_z_p_tissue_%0.2f_p_spear_%g_logreg"%(threshold,min_p_value) )
  check_and_mkdir(save_dir) 
  n_C = len(Cs)
  dna_auc_dir   = os.path.join( data.save_dir, "dna_auc_latent" )
  spearmanr_dir = os.path.join( data.save_dir, "spearmans_latent_tissue" )

  min_features = 5
  max_features = 20
  # spear_dna_z_rho = pd.read_csv( spearmanr_dir + "/dna_z_rho.csv", index_col="gene" )
  # spear_dna_z_p   = pd.read_csv( spearmanr_dir + "/dna_z_p.csv", index_col="gene" )
  # dna_z_auc     = pd.read_csv( dna_auc_dir + "/dna_z_auc.csv", index_col="gene" )
  # dna_z_auc_p   = pd.read_csv( dna_auc_dir + "/dna_z_auc_p.csv", index_col="gene" )
  # dna_z_p = dna_z_auc_p
  #
  #RNA_scale   = 2.0 / (1+np.exp(-data.RNA_scale)) -1   
  #miRNA_scale = 2.0 / (1+np.exp(-data.miRNA_scale))-1 
  #METH_scale  = 2.0 / (1+np.exp(-data.METH_scale ))-1  
  RNA_scale   =  tanh(data.RNA_scale)  
  miRNA_scale = tanh(data.miRNA_scale)
  METH_scale  = tanh(data.METH_scale )  
  
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
  X_ALL       = Z.values
  n = len(X_ALL)
  T           = data.T
  
  tissue_names = T.columns
  n_tissues    = len(tissue_names)
  tissue2idx = OrderedDict()
  for tissue,t_idx in zip( tissue_names, range(n_tissues)):
    tissue2idx[ tissue ] = t_idx
  
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
  Y_full_best=[]
  performances = []
  results = []
  tissues_used_all = []
  bcs_used_all=[]
  tissue_performances = {}
  for gene in dna_names[:nbr_dna_genes2process]:
    bad_gene = False
    dna_values = dna[gene].values
    ids_with_n, relevant_tissues = ids_with_at_least_p_mutations( dna_values, T, p = threshold )
    ids2use = pp.find(ids_with_n)
    tissues_used = np.zeros( n_tissues, dtype=int)
    for tissue in relevant_tissues:
      tissues_used[ tissue2idx[tissue] ] = 1
      
    tissue_series = pd.Series( tissues_used, index = tissue_names, name=gene)
    if len(ids_with_n)==0:
      print "1 skipping ",gene
      continue    
    
    gene_bcs = barcodes[ ids_with_n ]
    
    bcs_used = np.zeros( n, dtype = bool )
    bcs_used[ids_with_n] = True
    bcs_used = pd.Series( bcs_used, index = barcodes, name = gene )
    k = 0
    
    y_true = dna_values[ids_with_n]
    y_est = np.zeros( ( len(y_true),n_C) )
    X = Z[ ids_with_n ].values
    assert len(y_true) == len(X), "should be same"
    c=Counter()
    #best_ridge = 10.001
    Z_weights = np.zeros( (n_C,K,n_z) )
    Z_counts  = np.zeros( (K,n_z) )
    
    K_use = min( int(y_true.sum()),K)
    
    if K_use <K:
      bad_gene=True
      print "2 skipping ",gene
      continue
    y_full = np.zeros( (len(X_ALL),n_C) )
    folds = StratifiedKFold(n_splits=K_use, shuffle = True, random_state=random_state)
    for train_split, test_split in folds.split( X, y_true ):
      
      if bad_gene is True:
        continue 
      y_train = y_true[train_split]; X_train = X[train_split,:]
      y_test = y_true[test_split]; 
       
      
      # find top z by spearmanr p-values
      rho_dna_z = stats.spearmanr(y_train, X_train )
      dna_z_rho = np.squeeze(rho_dna_z[0][:1,:][:,1:])
      dna_z_p   = np.squeeze(rho_dna_z[1][:1,:][:,1:])
      
      ok_dna_z_p = pp.find( dna_z_p < min_p_value )
      J = np.argsort(dna_z_p)
      I=np.argsort( ok_dna_z_p )
      #pdb.set_trace()
      if len(I) < min_features:
        I = J[:min_features]
        #bad_gene = True
      elif len(I)>max_features:
        I=J[:max_features]
      else:
        I = ok_dna_z_p
      print "  using %d zs"%(len(I))
      #pdb.set_trace()
      c.update(z_names[np.sort(I)])
      #print gene, k, z_names[np.sort(ok_dna_z_p[I])]
      
      z_ids_k = I #ok_dna_z_p[I]
      #X_all  = Z_values
      X_ALL_gene = X_ALL[:,z_ids_k]
      #pdb.set_trace()
      X_train = X[train_split,:][:,z_ids_k]
      X_test = X[test_split,:][:,z_ids_k]
      Z_counts[k,z_ids_k] = 1.0
      M=LogisticBinaryClassifier()
      
      for C_idx, C in zip(range(n_C),Cs):
        M.fit( y_train, X_train, C=C )
        y_est[test_split,C_idx] = M.prob(X_test)
        y_full[:,C_idx] +=  M.prob(X_ALL_gene)
        weights = np.zeros(n_z)
        #pdb.set_trace()
         
        Z_weights[C_idx,k,z_ids_k] = M.coef_
      k+=1
    if bad_gene is True:
      continue
    y_full /= K
    aucs = np.zeros(n_C)
    aucs_p = np.zeros(n_C)
    mean_precisions = np.zeros(n_C)
    for C_idx, C in zip(range(n_C),Cs):
      auc_y_est, p_value_y_est = auc_and_pvalue( y_true, y_est[:,C_idx] )
      mean_precisions[C_idx] = average_precision_score(y_true, y_est[:,C_idx])
      aucs[C_idx] = auc_y_est; aucs_p[C_idx] = p_value_y_est
    best_c_idx = np.argmax(aucs)
    best_mean_precision=mean_precisions[best_c_idx]
    best_c = Cs[best_c_idx]
    y_est_best = y_est[:,best_c_idx]
    
    best_auc = aucs[best_c_idx]
    best_auc_p = aucs_p[best_c_idx]
    best_Z_weights = Z_weights[best_c_idx]
    
    z_w = pd.Series( best_Z_weights.sum(0) / Z_counts.sum(0), index = z_names, name = gene )
    y_full_best = pd.Series( y_full[:,best_c_idx], index = barcodes, name = gene )
    
    #
    tissue_performance = tissue_level_performance( ids2use, dna_values, y_full_best, relevant_tissues, data.T )
    #pdb.set_trace()
    
    bad = np.isnan(z_w); z_w[bad]=0
    #pdb.set_trace()
    order_y_est = np.argsort(y_est_best)
    y_est_best_pd = pd.DataFrame( np.vstack((y_true[order_y_est],y_est_best[order_y_est])).T, index = gene_bcs[order_y_est], columns=["true","est"] )
    performance = pd.Series( [best_mean_precision,best_auc,best_auc_p, len(y_true), int(np.sum(y_true))], index = ["AUPRC","AUROC","p-value","wildtype","mutations"], name = gene )
    z_w = pd.Series( z_w, index = z_names, name = gene )
    
    performances.append(performance)
    Z_W.append(z_w)
    Y_full_best.append( y_full_best )
    tissues_used_all.append(tissue_series)
    bcs_used_all.append(bcs_used)
    
    print gene, "AUC/p/pr/c", best_auc,  best_auc_p, best_mean_precision, best_c
    #print gene, "Z", [cj[0] for cj in c.most_common()]
    #print gene, "C", [cj[1] for cj in c.most_common()]
    
    gene_dir =  os.path.join( save_dir, "%s"%(gene) )
    check_and_mkdir(gene_dir)
    
    y_est_best_pd.to_csv( gene_dir +"/y_est_best.csv",index_label="barcode" )
    performance.to_csv( gene_dir +"/performance.csv" )
    z_w.to_csv( gene_dir +"/z_w.csv" )
    tissue_series.to_csv( gene_dir + "/tissue_series.csv")
    bcs_used.to_csv( gene_dir +"/bcs_used.csv")
    tissue_performance.to_csv( gene_dir+ "/tissue_performance.csv", index_label="tissue")
  
    tissue_performances[gene] = pd.DataFrame( tissue_performance.values, index=tissue_performance.index, columns=tissue_performance.columns)
    pp.close('all')
  print "FINALIZING"
  
  Y_full_best_concat  = pd.concat( Y_full_best, axis = 1 ) 
  Z_W_concat          = pd.concat( Z_W, axis=1 )
  performances_concat = pd.concat( performances, axis=1 )
  tissues_used_all    = pd.concat( tissues_used_all, axis = 1 )
  bcs_used_all        = pd.concat(bcs_used_all, axis=1)
  #tissue_performances_concat = pd.concat( tissue_performances, axis=1, names = Y_full_best_concat.columns )
  order_ = performances_concat.loc["AUROC"].sort_values(ascending=False).index.values
  #pdb.set_trace()
  bcs_used_all = bcs_used_all[order_]
  tissues_used_all=tissues_used_all[order_]
  Y_full_best_concat = Y_full_best_concat[order_]
  Z_W_concat = Z_W_concat[order_]
  performances_concat = performances_concat[order_]
  
  bcs_used_all.to_csv( save_dir +"/barcodes.csv", index_label="barcode" )
  tissues_used_all.to_csv( save_dir +"/tissues.csv", index_label="tissue")
  Y_full_best_concat.to_csv( save_dir +"/Y_full_best.csv", index_label="barcode")
  Z_W_concat.to_csv( save_dir +"/z_w.csv",index_label="z" )
  performances_concat.to_csv( save_dir +"/performances.csv",index_label="measure" )
  
  print "spearman with inputs"
  print "computing RNA-Z spearman rho's"
  rho_rna_y = stats.spearmanr( RNA_scale.values, Y_full_best_concat.values )
  #pdb.set_trace()
  print "computing miRNA-Z spearman rho's"
  rho_mirna_y = stats.spearmanr( miRNA_scale.values, Y_full_best_concat.values )
  print "computing METH-Z spearman rho's"
  rho_meth_y  = stats.spearmanr( METH_scale.values, Y_full_best_concat.values )
  print "computing DNA-Z spearman rho's"
  rho_dna_y   = stats.spearmanr(2*dna.values-1, Y_full_best_concat.values )

  rna_y_rho = pd.DataFrame( rho_rna_y[0][:n_rna,:][:,n_rna:], index = data.rna_names, columns= Y_full_best_concat.columns )
  rna_y_p   = pd.DataFrame( rho_rna_y[1][:n_rna,:][:,n_rna:], index = data.rna_names, columns=Y_full_best_concat.columns)

  mirna_y_rho = pd.DataFrame( rho_mirna_y[0][:n_mirna,:][:,n_mirna:], index = data.mirna_names, columns=Y_full_best_concat.columns)
  mirna_y_p   = pd.DataFrame( rho_mirna_y[1][:n_mirna,:][:,n_mirna:], index = data.mirna_names, columns=Y_full_best_concat.columns)
 
  meth_y_rho = pd.DataFrame( rho_meth_y[0][:n_meth,:][:,n_meth:], index = data.meth_names, columns=Y_full_best_concat.columns)
  meth_y_p   = pd.DataFrame( rho_meth_y[1][:n_meth,:][:,n_meth:], index = data.meth_names, columns=Y_full_best_concat.columns)

  #dna_y_rho = pd.DataFrame( rho_dna_y[0][:n_dna,:][:,n_dna:], index = data.dna_names, columns=Y_full_best_concat.columns)
  #dna_y_p   = pd.DataFrame( rho_dna_y[1][:n_dna,:][:,n_dna:], index = data.dna_names, columns=Y_full_best_concat.columns)

  
  rna_y_rho.to_csv( save_dir + "/rna_y_rho.csv", index_label="gene" )
  rna_y_p.to_csv( save_dir + "/rna_y_p.csv", index_label="gene" )
  
  mirna_y_rho.to_csv( save_dir + "/mirna_y_rho.csv", index_label="gene" )
  mirna_y_p.to_csv( save_dir + "/mirna_y_p.csv", index_label="gene" )
  
  meth_y_rho.to_csv( save_dir + "/meth_y_rho.csv", index_label="gene" )
  meth_y_p.to_csv( save_dir + "/meth_y_p.csv", index_label="gene" )
  
  nbr_genes_to_keep=20
  nbr_z_to_keep = 20
  results = []
  for gene in order_:
    gene_dir =  os.path.join( save_dir, "%s"%(gene) )
    print "finalizing ",gene
    dna_gene     = dna[gene]
    performance    = performances_concat[ gene ]
    tissues_used   = tissues_used_all[gene][tissues_used_all[gene]==1].index.values
    y_est_full     = Y_full_best_concat[gene] #.sort_values()
    z_w            = np.abs(Z_W_concat[gene]).sort_values(ascending=False)[:nbr_z_to_keep]
    bcs            = bcs_used_all[gene][bcs_used_all[gene]].index.values
    
    rna_y_p_gene   = rna_y_p[gene].sort_values()[:nbr_genes_to_keep]
    mirna_y_p_gene = mirna_y_p[gene].sort_values()[:nbr_genes_to_keep]
    meth_y_p_gene  = meth_y_p[gene].sort_values()[:nbr_genes_to_keep]
    
    
    rna_y_rho_gene   = rna_y_rho[gene].loc[rna_y_p_gene.index]
    mirna_y_rho_gene = mirna_y_rho[gene].loc[mirna_y_p_gene.index]
    meth_y_rho_gene  = meth_y_rho[gene].loc[meth_y_p_gene.index]
    
    plot_roc( gene_dir, gene, y_est_full.loc[bcs], dna_gene.loc[bcs], T.loc[bcs], performance.loc["AUROC"] )
    plot_pr( gene_dir, gene, y_est_full.loc[bcs], dna_gene.loc[bcs], T.loc[bcs], performance.loc["AUPRC"] )
    plot_predictions( gene_dir, gene, y_est_full.loc[bcs], dna_gene.loc[bcs], T.loc[bcs], performance.loc["AUROC"], performance.loc["AUPRC"] )
    
    plot_pr_tissues( gene_dir, gene, bcs, y_est_full, dna_gene, tissues_used, data.T, tissue_performances[gene] )
    plot_roc_tissues( gene_dir, gene, bcs, y_est_full, dna_gene, tissues_used, data.T, tissue_performances[gene] )
    
    plot_heatmap( gene_dir, gene, Z[z_w.index.values].loc[bcs], y_est_full.loc[bcs], z_w, "Z", normalize=True )
    plot_heatmap( gene_dir, gene, RNA_scale[rna_y_p_gene.index.values].loc[bcs], y_est_full.loc[bcs], rna_y_rho_gene, "RNA", normalize=True )
    plot_heatmap( gene_dir, gene, miRNA_scale[mirna_y_p_gene.index.values].loc[bcs], y_est_full.loc[bcs], mirna_y_rho_gene, "miRNA", normalize=True )
    plot_heatmap( gene_dir, gene, METH_scale[meth_y_p_gene.index.values].loc[bcs], y_est_full.loc[bcs], meth_y_rho_gene, "METH", normalize=True )
    plot_heatmap( gene_dir, gene, T.loc[bcs], y_est_full.loc[bcs], None, "TISSUE" )
    #pdb.set_trace()
    
    pp.close('all')

def deeper_meaning_dna_and_rna_fair_correct( data, nbr_dna_genes2process = 100, K=10, min_p_value=1e-3, threshold = 0.01, Cs = [0.00001, 0.001,0.1,10.0,1000.0] ):
  save_dir   = os.path.join( data.save_dir, "correct_deeper_meaning_dna_and_rna_fair_tissue_%0.2f_p_spear_%g_logreg"%(threshold,min_p_value) )
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
  #RNA_scale   = 2.0 / (1+np.exp(-data.RNA_scale)) -1   
  #miRNA_scale = 2.0 / (1+np.exp(-data.miRNA_scale))-1 
  #METH_scale  = 2.0 / (1+np.exp(-data.METH_scale ))-1  
  RNA_fair    = data.RNA_fair
  RNA_scale   =  tanh(data.RNA_scale)  
  miRNA_scale = tanh(data.miRNA_scale)
  METH_scale  = tanh(data.METH_scale )  
  
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

  min_features = 5
  max_features = 20
  Z           = data.Z
  barcodes    = Z.index.values
  
  DATA        = RNA_fair.loc[barcodes]
  X_ALL       = DATA.values
  n           = len(X_ALL)

  z_names     = DATA.columns
  n_z         = len(z_names)
  
  T           = data.T
  
  tissue_names = T.columns
  n_tissues    = len(tissue_names)
  tissue2idx = OrderedDict()
  for tissue,t_idx in zip( tissue_names, range(n_tissues)):
    tissue2idx[ tissue ] = t_idx
  
  dna_names = data.dna.sum(0).sort_values(ascending=False).index.values
  n_dna = len(dna_names)
  dna = data.dna[ dna_names ]
  
  k_idx = 1
  results = []
  random_state=0
  Z_W = []
  Y_full_best=[]
  performances = []
  results = []
  tissues_used_all = []
  bcs_used_all=[]
  tissue_performances = {}
  for gene in dna_names[:nbr_dna_genes2process]:
    bad_gene = False
    dna_values = dna[gene].values
    ids_with_n, relevant_tissues = ids_with_at_least_p_mutations( dna_values, T, p = threshold )
    ids2use = pp.find(ids_with_n)
    tissues_used = np.zeros( n_tissues, dtype=int)
    for tissue in relevant_tissues:
      tissues_used[ tissue2idx[tissue] ] = 1
      
    tissue_series = pd.Series( tissues_used, index = tissue_names, name=gene)
    if len(ids_with_n)==0:
      print "1 skipping ",gene
      continue    
    
    gene_bcs = barcodes[ ids_with_n ]
    
    bcs_used = np.zeros( n, dtype = bool )
    bcs_used[ids_with_n] = True
    bcs_used = pd.Series( bcs_used, index = barcodes, name = gene )
    k = 0
    
    y_true = dna_values[ids_with_n]
    y_est = np.zeros( ( len(y_true),n_C) )
    X = DATA[ ids_with_n ].values
    assert len(y_true) == len(X), "should be same"
    c=Counter()
    #best_ridge = 10.001
    Z_weights = np.zeros( (n_C,K,n_z) )
    Z_counts  = np.zeros( (K,n_z) )
    K_use = min( int(y_true.sum()),K)
    
    if K_use <K:
      bad_gene=True
      print "2 skipping ",gene
      continue
    y_full = np.zeros( (len(X_ALL),n_C) )
    folds = StratifiedKFold(n_splits=K_use, shuffle = True, random_state=random_state)
    for train_split, test_split in folds.split( X, y_true ):
      
      if bad_gene is True:
        continue 
      y_train = y_true[train_split]; X_train = X[train_split,:]
      y_test = y_true[test_split]; 
       
      
      # find top z by spearmanr p-values
      rho_dna_z = stats.spearmanr(y_train, X_train )
      dna_z_rho = np.squeeze(rho_dna_z[0][:1,:][:,1:])
      dna_z_p   = np.squeeze(rho_dna_z[1][:1,:][:,1:])
      
      ok_dna_z_p = pp.find( dna_z_p < min_p_value )
      J = np.argsort(dna_z_p)
      I=np.argsort( ok_dna_z_p )
      #pdb.set_trace()
      if len(I) < min_features:
        I = J[:min_features]
        #bad_gene = True
      elif len(I)>max_features:
        I=J[:max_features]
      else:
        I = ok_dna_z_p
      print "  using %d zs"%(len(I))
      #pdb.set_trace()
      c.update(z_names[np.sort(I)])
      #print gene, k, z_names[np.sort(ok_dna_z_p[I])]
      
      z_ids_k = I #ok_dna_z_p[I]
      #X_all  = Z_values
      X_ALL_gene = X_ALL[:,z_ids_k]
      #pdb.set_trace()
      X_train = X[train_split,:][:,z_ids_k]
      X_test = X[test_split,:][:,z_ids_k]
      Z_counts[k,z_ids_k] = 1.0
      M=LogisticBinaryClassifier()
      
      for C_idx, C in zip(range(n_C),Cs):
        M.fit( y_train, X_train, C=C )
        y_est[test_split,C_idx] = M.prob(X_test)
        y_full[:,C_idx] +=  M.prob(X_ALL_gene)
        weights = np.zeros(n_z)
        #pdb.set_trace()
         
        Z_weights[C_idx,k,z_ids_k] = M.coef_
      k+=1
    if bad_gene is True:
      continue
    y_full /= K
    aucs = np.zeros(n_C)
    aucs_p = np.zeros(n_C)
    mean_precisions = np.zeros(n_C)
    for C_idx, C in zip(range(n_C),Cs):
      auc_y_est, p_value_y_est = auc_and_pvalue( y_true, y_est[:,C_idx] )
      mean_precisions[C_idx] = average_precision_score(y_true, y_est[:,C_idx])
      aucs[C_idx] = auc_y_est; aucs_p[C_idx] = p_value_y_est
    best_c_idx = np.argmax(aucs)
    best_mean_precision=mean_precisions[best_c_idx]
    best_c = Cs[best_c_idx]
    y_est_best = y_est[:,best_c_idx]
    
    best_auc = aucs[best_c_idx]
    best_auc_p = aucs_p[best_c_idx]
    best_Z_weights = Z_weights[best_c_idx]
    
    z_w = pd.Series( best_Z_weights.sum(0) / Z_counts.sum(0), index = z_names, name = gene )
    y_full_best = pd.Series( y_full[:,best_c_idx], index = barcodes, name = gene )
    
    #
    tissue_performance = tissue_level_performance( ids2use, dna_values, y_full_best, relevant_tissues, data.T )
    #pdb.set_trace()
    
    bad = np.isnan(z_w); z_w[bad]=0
    #pdb.set_trace()
    order_y_est = np.argsort(y_est_best)
    y_est_best_pd = pd.DataFrame( np.vstack((y_true[order_y_est],y_est_best[order_y_est])).T, index = gene_bcs[order_y_est], columns=["true","est"] )
    performance = pd.Series( [best_mean_precision,best_auc,best_auc_p, len(y_true), int(np.sum(y_true))], index = ["AUPRC","AUROC","p-value","wildtype","mutations"], name = gene )
    z_w = pd.Series( z_w, index = z_names, name = gene )
    
    performances.append(performance)
    Z_W.append(z_w)
    Y_full_best.append( y_full_best )
    tissues_used_all.append(tissue_series)
    bcs_used_all.append(bcs_used)
    
    print gene, "AUC/p/pr/c", best_auc,  best_auc_p, best_mean_precision, best_c
    #print gene, "Z", [cj[0] for cj in c.most_common()]
    #print gene, "C", [cj[1] for cj in c.most_common()]
    
    gene_dir =  os.path.join( save_dir, "%s"%(gene) )
    check_and_mkdir(gene_dir)
    
    y_est_best_pd.to_csv( gene_dir +"/y_est_best.csv",index_label="barcode" )
    performance.to_csv( gene_dir +"/performance.csv" )
    z_w.to_csv( gene_dir +"/rna_fair_w.csv" )
    tissue_series.to_csv( gene_dir + "/tissue_series.csv")
    bcs_used.to_csv( gene_dir +"/bcs_used.csv")
    tissue_performance.to_csv( gene_dir+ "/tissue_performance.csv", index_label="tissue")
  
    tissue_performances[gene] = pd.DataFrame( tissue_performance.values, index=tissue_performance.index, columns=tissue_performance.columns)
    pp.close('all')
  print "FINALIZING"
  
  Y_full_best_concat  = pd.concat( Y_full_best, axis = 1 ) 
  Z_W_concat          = pd.concat( Z_W, axis=1 )
  performances_concat = pd.concat( performances, axis=1 )
  tissues_used_all    = pd.concat( tissues_used_all, axis = 1 )
  bcs_used_all        = pd.concat(bcs_used_all, axis=1)
  #tissue_performances_concat = pd.concat( tissue_performances, axis=1, names = Y_full_best_concat.columns )
  order_ = performances_concat.loc["AUROC"].sort_values(ascending=False).index.values
  #pdb.set_trace()
  bcs_used_all = bcs_used_all[order_]
  tissues_used_all=tissues_used_all[order_]
  Y_full_best_concat = Y_full_best_concat[order_]
  Z_W_concat = Z_W_concat[order_]
  performances_concat = performances_concat[order_]
  
  bcs_used_all.to_csv( save_dir +"/barcodes.csv", index_label="barcode" )
  tissues_used_all.to_csv( save_dir +"/tissues.csv", index_label="tissue")
  Y_full_best_concat.to_csv( save_dir +"/Y_full_best.csv", index_label="barcode")
  Z_W_concat.to_csv( save_dir +"/rna_fair_w.csv",index_label="z" )
  performances_concat.to_csv( save_dir +"/performances.csv",index_label="measure" )
  
  print "spearman with inputs"
  print "computing RNA-Z spearman rho's"
  rho_rna_y = stats.spearmanr( RNA_scale.values, Y_full_best_concat.values )
  #pdb.set_trace()
  print "computing miRNA-Z spearman rho's"
  rho_mirna_y = stats.spearmanr( miRNA_scale.values, Y_full_best_concat.values )
  print "computing METH-Z spearman rho's"
  rho_meth_y  = stats.spearmanr( METH_scale.values, Y_full_best_concat.values )
  print "computing DNA-Z spearman rho's"
  rho_dna_y   = stats.spearmanr(2*dna.values-1, Y_full_best_concat.values )

  rna_y_rho = pd.DataFrame( rho_rna_y[0][:n_rna,:][:,n_rna:], index = data.rna_names, columns= Y_full_best_concat.columns )
  rna_y_p   = pd.DataFrame( rho_rna_y[1][:n_rna,:][:,n_rna:], index = data.rna_names, columns=Y_full_best_concat.columns)

  mirna_y_rho = pd.DataFrame( rho_mirna_y[0][:n_mirna,:][:,n_mirna:], index = data.mirna_names, columns=Y_full_best_concat.columns)
  mirna_y_p   = pd.DataFrame( rho_mirna_y[1][:n_mirna,:][:,n_mirna:], index = data.mirna_names, columns=Y_full_best_concat.columns)
 
  meth_y_rho = pd.DataFrame( rho_meth_y[0][:n_meth,:][:,n_meth:], index = data.meth_names, columns=Y_full_best_concat.columns)
  meth_y_p   = pd.DataFrame( rho_meth_y[1][:n_meth,:][:,n_meth:], index = data.meth_names, columns=Y_full_best_concat.columns)

  #dna_y_rho = pd.DataFrame( rho_dna_y[0][:n_dna,:][:,n_dna:], index = data.dna_names, columns=Y_full_best_concat.columns)
  #dna_y_p   = pd.DataFrame( rho_dna_y[1][:n_dna,:][:,n_dna:], index = data.dna_names, columns=Y_full_best_concat.columns)

  
  rna_y_rho.to_csv( save_dir + "/rna_y_rho.csv", index_label="gene" )
  rna_y_p.to_csv( save_dir + "/rna_y_p.csv", index_label="gene" )
  
  mirna_y_rho.to_csv( save_dir + "/mirna_y_rho.csv", index_label="gene" )
  mirna_y_p.to_csv( save_dir + "/mirna_y_p.csv", index_label="gene" )
  
  meth_y_rho.to_csv( save_dir + "/meth_y_rho.csv", index_label="gene" )
  meth_y_p.to_csv( save_dir + "/meth_y_p.csv", index_label="gene" )
  
  nbr_genes_to_keep=20
  nbr_z_to_keep = 20
  results = []
  for gene in order_:
    gene_dir =  os.path.join( save_dir, "%s"%(gene) )
    print "finalizing ",gene
    dna_gene     = dna[gene]
    performance    = performances_concat[ gene ]
    tissues_used   = tissues_used_all[gene][tissues_used_all[gene]==1].index.values
    y_est_full     = Y_full_best_concat[gene] #.sort_values()
    z_w            = np.abs(Z_W_concat[gene]).sort_values(ascending=False)[:nbr_z_to_keep]
    bcs            = bcs_used_all[gene][bcs_used_all[gene]].index.values
    
    rna_y_p_gene   = rna_y_p[gene].sort_values()[:nbr_genes_to_keep]
    mirna_y_p_gene = mirna_y_p[gene].sort_values()[:nbr_genes_to_keep]
    meth_y_p_gene  = meth_y_p[gene].sort_values()[:nbr_genes_to_keep]
    
    
    rna_y_rho_gene   = rna_y_rho[gene].loc[rna_y_p_gene.index]
    mirna_y_rho_gene = mirna_y_rho[gene].loc[mirna_y_p_gene.index]
    meth_y_rho_gene  = meth_y_rho[gene].loc[meth_y_p_gene.index]
    
    plot_roc( gene_dir, gene, y_est_full.loc[bcs], dna_gene.loc[bcs], T.loc[bcs], performance.loc["AUROC"] )
    plot_pr( gene_dir, gene, y_est_full.loc[bcs], dna_gene.loc[bcs], T.loc[bcs], performance.loc["AUPRC"] )
    plot_predictions( gene_dir, gene, y_est_full.loc[bcs], dna_gene.loc[bcs], T.loc[bcs], performance.loc["AUROC"], performance.loc["AUPRC"] )
    
    plot_pr_tissues( gene_dir, gene, bcs, y_est_full, dna_gene, tissues_used, data.T, tissue_performances[gene] )
    plot_roc_tissues( gene_dir, gene, bcs, y_est_full, dna_gene, tissues_used, data.T, tissue_performances[gene] )
    
    plot_heatmap( gene_dir, gene, DATA[z_w.index.values].loc[bcs], y_est_full.loc[bcs], z_w, "RNA", normalize=True )
    plot_heatmap( gene_dir, gene, RNA_scale[rna_y_p_gene.index.values].loc[bcs], y_est_full.loc[bcs], rna_y_rho_gene, "RNA", normalize=True )
    plot_heatmap( gene_dir, gene, miRNA_scale[mirna_y_p_gene.index.values].loc[bcs], y_est_full.loc[bcs], mirna_y_rho_gene, "miRNA", normalize=True )
    plot_heatmap( gene_dir, gene, METH_scale[meth_y_p_gene.index.values].loc[bcs], y_est_full.loc[bcs], meth_y_rho_gene, "METH", normalize=True )
    plot_heatmap( gene_dir, gene, T.loc[bcs], y_est_full.loc[bcs], None, "TISSUE" )
    #pdb.set_trace()
    
    pp.close('all')

def deeper_meaning_dna_and_rna_fair_correct_by_tissue( data, nbr_dna_genes2process = 100, K=10, min_p_value=1e-3, threshold = 0.01, Cs = [0.00001, 0.001,0.1,10.0,1000.0] ):
  save_dir   = os.path.join( data.save_dir, "correct_by_tissue_deeper_meaning_dna_and_rna_fair_tissue_%0.2f_p_spear_%g_logreg"%(threshold,min_p_value) )
  check_and_mkdir(save_dir) 
  n_C = len(Cs)
  dna_auc_dir   = os.path.join( data.save_dir, "dna_auc_latent" )
  spearmanr_dir = os.path.join( data.save_dir, "spearmans_latent_tissue" )

  RNA_fair    = data.RNA_fair
  RNA_scale   = tanh(data.RNA_scale)  
  miRNA_scale = tanh(data.miRNA_scale)
  METH_scale  = tanh(data.METH_scale )  
  
  n_rna   = data.n_rna
  n_mirna = data.n_mirna
  n_meth  = data.n_meth
  
  rna_z_rho = pd.read_csv( spearmanr_dir + "/rna_z_rho.csv", index_col="gene" )
  mirna_z_rho = pd.read_csv( spearmanr_dir + "/mirna_z_rho.csv", index_col="gene" )
  meth_z_rho = pd.read_csv( spearmanr_dir + "/meth_z_rho.csv", index_col="gene" )
  
  data_name   = "Z"
  Z           = data.Z
  barcodes    = Z.index.values
  
  DATA        = Z #RNA_fair.loc[barcodes]
  X_ALL       = DATA.values
  n           = len(X_ALL)

  z_names     = DATA.columns
  n_z         = len(z_names)
  
  T           = data.T.loc[barcodes]
  
  tissue_names = T.columns
  n_tissues    = len(tissue_names)
  tissue2idx = OrderedDict()
  for tissue,t_idx in zip( tissue_names, range(n_tissues)):
    tissue2idx[ tissue ] = t_idx
  
  dna_names = data.dna.sum(0).sort_values(ascending=False).index.values
  n_dna = len(dna_names)
  dna = data.dna[ dna_names ]
  
  min_features = 5
  max_features = 20
  k_idx = 1
  results = []
  random_state=0
  nbr_z_to_keep=10
  tissue_performances = {}

  for tissue in tissue_names[:2]:
    tissue_dir =  os.path.join( save_dir, "%s"%(tissue) )
    check_and_mkdir(tissue_dir)
    
    print "working ", tissue
    tissue_bcs = T[ T[ tissue ] == 1 ].index.values
    tissue_dna = data.dna.loc[tissue_bcs]
    
    dna_names = tissue_dna.sum(0).sort_values(ascending=False).index.values
    n_dna = len(dna_names)
    dna = tissue_dna[ dna_names ]
    
    Z_W = []
    performances = []
    tissue_series=[]
    for gene in dna_names[:nbr_dna_genes2process]:
      bad_gene = False
      dna_values = dna[gene].values    
      gene_bcs = tissue_bcs
      bcs_used = tissue_bcs
      k = 0
      y_true = dna_values
      y_est = np.zeros( ( len(y_true),n_C) )
      X = DATA.loc[gene_bcs].values
      assert len(y_true) == len(X), "should be same"
      c=Counter()
      #best_ridge = 10.001
      Z_weights = np.zeros( (n_C,K,n_z) )
      Z_counts  = np.zeros( (K,n_z) )
    
      K_use = min( int(y_true.sum()),K)
    
      if K_use <K:
        bad_gene=True
        print "2 skipping ",gene
        continue
      #y_full = np.zeros( (len(X_ALL),n_C) )
      folds = StratifiedKFold(n_splits=K_use, shuffle = True, random_state=random_state)
      for train_split, test_split in folds.split( X, y_true ):
      
        if bad_gene is True:
          continue 
        y_train = y_true[train_split]; X_train = X[train_split,:]
        y_test = y_true[test_split]; 
       
      
        # find top z by spearmanr p-values
        rho_dna_z = stats.spearmanr(y_train, X_train )
        dna_z_rho = np.squeeze(rho_dna_z[0][:1,:][:,1:])
        dna_z_p   = np.squeeze(rho_dna_z[1][:1,:][:,1:])
      
        ok_dna_z_p = pp.find( dna_z_p < min_p_value )
        J = np.argsort(dna_z_p)
        I=np.argsort( ok_dna_z_p )
        #pdb.set_trace()
        if len(I) < min_features:
          I = J[:min_features]
          #bad_gene = True
        elif len(I)>max_features:
          I=I[:max_features]
          
        print "  using %d zs"%(len(I))
        #c.update(z_names[np.sort(ok_dna_z_p[I])])
        c.update(z_names[I])
        z_ids_k  = I
        X_train  = X[train_split,:][:,z_ids_k]
        X_test   = X[test_split,:][:,z_ids_k]
        Z_counts[k,z_ids_k] = 1.0
        M=LogisticBinaryClassifier()
      
        for C_idx, C in zip(range(n_C),Cs):
          M.fit( y_train, X_train, C=C )
          y_est[test_split,C_idx] = M.prob(X_test)
          Z_weights[C_idx,k,z_ids_k] = M.coef_
          
        k+=1
      # if bad_gene is True:
      #   continue

      aucs = np.zeros(n_C)
      aucs_p = np.zeros(n_C)
      mean_precisions = np.zeros(n_C)
      for C_idx, C in zip(range(n_C),Cs):
        auc_y_est, p_value_y_est = auc_and_pvalue( y_true, y_est[:,C_idx] )
        mean_precisions[C_idx] = average_precision_score(y_true, y_est[:,C_idx])
        aucs[C_idx] = auc_y_est; aucs_p[C_idx] = p_value_y_est
      best_c_idx          = np.argmax(aucs)
      best_mean_precision = mean_precisions[best_c_idx]
      best_c              = Cs[best_c_idx]
      y_est_best          = y_est[:,best_c_idx]
    
      best_auc       = aucs[best_c_idx]
      best_auc_p     = aucs_p[best_c_idx]
      best_Z_weights = Z_weights[best_c_idx]
    
      z_w = pd.Series( best_Z_weights.sum(0) / Z_counts.sum(0), index = z_names, name = gene )
      bad = np.isnan(z_w); z_w[bad]=0
      
      order_y_est = np.argsort(y_est_best)
      y_est_best_pd = pd.DataFrame( np.vstack((y_true[order_y_est],y_est_best[order_y_est])).T, index = gene_bcs[order_y_est], columns=["true","est"] )
      performance = pd.Series( [best_mean_precision,best_auc,best_auc_p, len(y_true), int(np.sum(y_true))], index = ["AUPRC","AUROC","p-value","wildtype","mutations"], name = gene )
      z_w = pd.Series( z_w, index = z_names, name = gene )
    
      performances.append(performance)
      Z_W.append(z_w)
      #bcs_used_all.append(bcs_used)
    
      print gene, "AUC/p/pr/c", best_auc,  best_auc_p, best_mean_precision, best_c
    
      gene_dir =  os.path.join( tissue_dir, "%s"%(gene) )
      check_and_mkdir(gene_dir)
    
      y_est_best_pd.to_csv( gene_dir +"/y_est_best.csv",index_label="barcode" )
      performance.to_csv( gene_dir +"/performance.csv" )
      z_w.to_csv( gene_dir +"/rna_fair_w.csv" )
      tissue_performances[gene] = performance #pd.DataFrame( performance.values, index=performance.index, columns=performance.columns)
      pp.close('all')
      
      y_est_best = pd.Series( y_est_best, index = tissue_bcs, name = gene)
      y_true = pd.Series( y_true, index = tissue_bcs, name = gene)
      plot_roc( gene_dir, gene, y_est_best, y_true, None, performance.loc["AUROC"] )
      plot_pr( gene_dir, gene, y_est_best, y_true, None,  performance.loc["AUPRC"] )
      plot_predictions( gene_dir, gene, y_est_best, y_true, None, performance.loc["AUROC"], performance.loc["AUPRC"] )
    
      #plot_pr_tissues( gene_dir, gene, bcs, y_est_full, dna_gene, tissues_used, data.T, tissue_performances[gene] )
      #plot_roc_tissues( gene_dir, gene, bcs, y_est_full, dna_gene, tissues_used, data.T, tissue_performances[gene] )
      z_w            = np.abs(z_w).sort_values(ascending=False)[:nbr_z_to_keep]
      plot_heatmap( gene_dir, gene, DATA[z_w.index.values].loc[tissue_bcs], y_est_best, z_w, data_name, normalize=True )
      #plot_heatmap( gene_dir, gene, RNA_scale[rna_y_p_gene.index.values].loc[bcs], y_est_full.loc[bcs], rna_y_rho_gene, "RNA", normalize=True )
      #plot_heatmap( gene_dir, gene, miRNA_scale[mirna_y_p_gene.index.values].loc[bcs], y_est_full.loc[bcs], mirna_y_rho_gene, "miRNA", normalize=True )
      #plot_heatmap( gene_dir, gene, METH_scale[meth_y_p_gene.index.values].loc[bcs], y_est_full.loc[bcs], meth_y_rho_gene, "METH", normalize=True )
      #plot_heatmap( gene_dir, gene, T.loc[bcs], y_est_full.loc[bcs], None, "TISSUE" )
      #pdb.set_trace()
    
    pp.close('all')
        
def plot_heatmap( gene_dir, gene, X, y_est, X_score, name, size1=12, size2=4, ax=None, save=True, normalize=False ):
  if ax is None:
    f=pp.figure( figsize=(size1,size2))
    ax=f.add_subplot(111)
    
  d1,d2 = X.values.shape
  sorted_y_est = y_est.sort_values()
  
  if X_score is not None:
    s = np.sign( X_score.values )
  
  
    X_sorted = X.loc[ sorted_y_est.index ] * s
  else:
    X_sorted = X.loc[ sorted_y_est.index ]
  
  V = X_sorted.values.T
  if normalize is True:
    V -= V.mean(1)[:,np.newaxis]
    V /= V.std(1)[:,np.newaxis]
    
    
  r = size1/size2

  ax.imshow( V, cmap='rainbow', interpolation='nearest', aspect=float(d1)/(d2*r) )
  ax.set_yticks( np.arange(V.shape[0]))
  ax.set_yticklabels( X.columns )
  pp.grid('off')
  if save is True:
    pp.savefig( gene_dir + "/heatmap_%s.png"%(name), fmt='png')
    
  return ax


  
  
def plot_predictions( gene_dir, gene, y_est, y_true, tissues, auc, mean_precision, size1=12, size2=4, ax=None, save=True ):
  if ax is None:
    f=pp.figure( figsize=(size1,size2))
    ax=f.add_subplot(111)
    
  
  sorted_y_est = y_est.sort_values()
  sorted_y_true = y_true.loc[ sorted_y_est.index ]
  
  #ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  ax.vlines( pp.find(sorted_y_true.values), 0, 1, color='blue', lw=0.25, alpha=0.25, label="mutations" )
  ax.plot( sorted_y_est.values, 'r-', lw=2, label='Pr(mutation) (AUC = %0.2f, mean Precision = %0.2f)' % (auc,mean_precision) )
  #ax.plot( sorted_y_true.values, 'b|', ms=5, mew=0.5, alpha=0.5,  label='Mutations' )
  #ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.0])
  ax.set_xlim([0,len(sorted_y_true)])
  ax.set_ylabel('Pr(mutation)')
  ax.set_xlabel('')
  ax.set_title('%s'%(gene))
  ax.legend(loc="lower right")
  pp.grid('off')
  if save is True:
    pp.savefig( gene_dir + "/predictions.png", fmt='png', dpi=300, bbox_inches='tight')
    
  return ax
     
    
def plot_roc( gene_dir, gene, y_est, y_true, tissues, auc, ax=None, save=True ):
  fpr, tpr, _ = roc_curve(y_true, y_est )
  
  if ax is None:
    f=pp.figure()
    ax=f.add_subplot(111)
    
  
  ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  ax.plot( fpr, tpr, 'r-', lw=2, label='ROC curve (area = %0.2f)' % auc )
  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.05])
  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  ax.set_title('%s'%(gene))
  ax.legend(loc="lower right")
  pp.grid('off')
  if save is True:
    pp.savefig( gene_dir + "/roc.png", fmt='png', dpi=300)
    
  return ax

def plot_pr( gene_dir, gene, y_est, y_true, tissues, mean_precision, ax=None, save=True ):
  precision, recall, _ = precision_recall_curve(y_true, y_est )
  
  if ax is None:
    f=pp.figure()
    ax=f.add_subplot(111)
    
  
  ax.plot([0, 1], [mean_precision, mean_precision], color='navy', lw=1, linestyle='--')
  ax.plot( recall, precision, 'r-', lw=2, label='PR curve (mean = %0.2f)' % mean_precision )
  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.05])
  ax.set_xlabel('Recall')
  ax.set_ylabel('Precision')
  ax.set_title('%s'%(gene))
  ax.legend(loc="lower right")
  
  if save is True:
    pp.savefig( gene_dir + "/precision_recall.png", fmt='png', dpi=300, bbox_inches='tight')
    
  return ax
    
    # check_and_mkdir(gene_dir)
    #
    # print performances_concat[ gene ]
    #
    # results.append( {"a_dna":gene,"cv": [float(best_auc),  float(best_auc_p), float(best_mean_precision)],\
    #                 "c_wildtype":len(y_true),  "c_mutations":int(np.sum(y_true)),\
    #                 "tissues":relevant_tissues,\
    #                 "rna_y":list(rna_z_p_gene[:20].index.values) ,\
    #                 "rna_z":list( np.abs(w_rna).sort_values(ascending=False).index.values[:20]),\
    #                 "mirna_y":list(mirna_z_p_gene[:20].index.values) ,\
    #                 "mirna_z":list( np.abs(w_mirna).sort_values(ascending=False).index.values[:20]),\
    #                 "meth_y":list(meth_z_p_gene[:20].index.values) ,\
    #                 "meth_z":list( np.abs(w_meth).sort_values(ascending=False).index.values[:20]),\
    #                 "top_z":list(np.abs(z_w).sort_values(ascending=False).index.values[:20]) } )
    
  #dna_y_rho.to_csv( save_dir + "/dna_y_rho.csv", index_label="gene" )
  #dna_y_p.to_csv( save_dir + "/dna_y_p.csv", index_label="gene" )
  
  # fptr = open( save_dir + "/pan_cancer_dna.yaml","w+" )
  # fptr.write( yaml.dump(results))
  # fptr.close()
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