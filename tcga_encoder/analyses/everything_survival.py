from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.analyses.everything_functions import *
from tcga_encoder.analyses.everything_long import *
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
import lifelines
from lifelines import CoxPHFitter, AalenAdditiveFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
#except:
#  print "could not import graphviz "
#from networkx.drawing.nx_agraph import spring_layout 
  
from scipy import stats  
def load_data_and_fill( data_location, results_location ):
  input_sources = ["RNA","miRNA","METH"]
  data_store = load_store( data_location, "data.h5")
  model_store = load_store( results_location, "full_vae_model.h5")
  fill_store = load_store( results_location, "full_vae_fill.h5")
  
  subtypes = load_subtypes( data_store )
  
  #tissues = data_store["/CLINICAL/TISSUE"].loc[barcodes]
  
  survival = PanCancerSurvival( data_store )
  
  #pdb.set_trace()
  Z,Z_std = load_latent( fill_store )
  model_barcodes = Z.index.values 
  
  H = load_hidden( fill_store, model_barcodes )
  
  RNA_fair, miRNA_fair, METH_fair    = load_fair_data( data_store, model_barcodes )
  RNA_scale, miRNA_scale, METH_scale = load_scaled_data( fill_store, model_barcodes )
  
  rna_names = RNA_scale.columns
  mirna_names = miRNA_scale.columns
  meth_names = METH_scale.columns
  h_names = H.columns
  z_names = Z.columns
  
  n_rna   = len(rna_names)
  n_mirna = len(mirna_names)
  n_meth  = len(meth_names)
  n_h     = len(h_names)
  n_z     = len(z_names)
  
  everything_dir = os.path.join( os.path.join( HOME_DIR, results_location ), "everything2" )
  check_and_mkdir(everything_dir)

  data                = EverythingObject()
  data.input_sources  = input_sources
  data.data_store     = data_store
  data.model_store    = model_store
  data.fill_store     = fill_store
  data.subtypes       = subtypes
  data.survival       = survival
  data.Z              = Z
  
  try:
    data.T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ Z.index ]
  except:
    data.T=data.data_store["/CLINICAL/TISSUE"].loc[ Z.index ]
  
  data.Z_std           = Z_std
  data.H              = H
  data.W_input2h      = get_hidden_weights( model_store, input_sources, data_store )
  data.W_h2z          = get_hidden2z_weights( model_store )
  data.weighted_W_h2z = join_weights( data.W_h2z, data.W_input2h  )
  data.dna            = data.data_store["/DNA/channel/0"].loc[data.Z.index].fillna(0)
  data.RNA_scale      = RNA_scale
  data.miRNA_scale    = miRNA_scale
  data.METH_scale     = METH_scale  
  
  data.RNA_fair      = RNA_fair
  data.miRNA_fair    = miRNA_fair
  data.METH_fair     = METH_fair

  data.rna_names      = rna_names
  data.mirna_names    = mirna_names
  data.meth_names     = meth_names
  data.h_names        = h_names
  data.z_names        = z_names
  data.n_rna          = n_rna
  data.n_mirna        = n_mirna
  data.n_meth         = n_meth
  data.n_h            = n_h
  data.n_z            = n_z
  data.save_dir       = everything_dir
  
  data.data_store.close()
  data.fill_store.close()
  data.model_store.close()
  
  return data
  
def make_graph_from_correlations( C, t ):
  G=nx.Graph()
  names = C.index.values
  node_colors = []
  json_node = []
  added_nodes = {}
  i=0
  tau=600
  g_weights = []
  g_edge_weights = []
  links=[]
  
  X = C.values
  n = len(X)
  nbr = 0
  for i in xrange(n-1):
    row = i
    for j in xrange( n - i - 1 ):
      col = i + j + 1
      w = X[row,col]
      pdb.set_trace()
      if np.abs(w) > t:
        #print "adding edge %s to %s with weight %f"%(barcodes[i], barcodes[j],w)
        #G.add_edge(barcodes[i][-7:], barcodes[j][-7:], weight=w)
        print "adding edge %s to %s with weight %f"%(names[row], names[col],w)
        
        if added_nodes.has_key(names[row]) is False:
          added_nodes[ names[row] ] = 1
        if added_nodes.has_key(names[col]) is False:
          added_nodes[ names[col] ] = 1
        
        G.add_edge(names[row], names[col], weight=float(w))
        nbr+=1
        #g_weights.append(w)
        #g_edge_weights.append([barcodes[i], barcodes[j],w])
        #links.append( {"source":int(i),"target":int(j),"w":float(w)} )
  #  i+=1
  #g_weights = np.array(g_weights)
  print "last row/col/n", row, col, n
  print "added %d edges"%nbr
  return G

def make_graph_from_mwst( C, t, names, sizes ):
  G=nx.Graph()
  #names = C.index.values
  node_colors = []
  json_node = []
  added_nodes = {}
  i=0
  tau=600
  g_weights = []
  g_edge_weights = []
  links=[]
  
  #X = C.values
  #n = len(X)
  nbr = 0
  i = 0
  for x in C:
    #print x, x.indices, x.data
    indices = x.indices
    weights = x.data
    #print indices, weights
    for j,w in zip(indices,weights):
      #pdb.set_trace()
      if w < t:
        if added_nodes.has_key(names[i]) is False:
          added_nodes[ names[i] ] = 1
          G.add_node( names[i], size=sizes[i] )
        if added_nodes.has_key(names[j]) is False:
          added_nodes[ names[j] ] = 1
          G.add_node( names[j], size=sizes[j] )
        
        
        G.add_edge(names[i], names[j], weight=w)
        nbr+=1
        
    i+=1


  #print "last row/col/n", row, col, n
  print "added %d edges"%nbr
  return G

def make_graph_from_coo( C, t, names, sizes, max_nbr = 1 ):
  G=nx.Graph()
  #names = C.index.values
  node_colors = []
  json_node = []
  added_nodes = {}
  i=0
  tau=600
  g_weights = []
  g_edge_weights = []
  links=[]
  
  max_node_weight = OrderedDict()
  min_node_weight = OrderedDict()
  for n in names:
    max_node_weight[n] = -np.inf
    min_node_weight[n] = np.inf
    
  #X = C.values
  #n = len(X)
  order = np.argsort( -C.data )
  nbr = 0
  i = 0
  for row,col,data in zip(C.row[order],C.col[order],C.data[order]):
    if row == col:
      continue
    row_name = names[row]
    col_name = names[col]
    max_row_w = max_node_weight[row_name]
    max_col_w = max_node_weight[col_name] 
    min_row_w = min_node_weight[row_name]
    min_col_w = min_node_weight[col_name] 
    row_n = len( G.edges(row_name) )
    col_n = len( G.edges(col_name) )
    
    print "row: ",row_name, max_row_w, min_row_w, row_n
    print "col: ", col_name, max_col_w, min_col_w, col_n
    if added_nodes.has_key(row_name) is False:
      added_nodes[ row_name ] = 1
      G.add_node( row_name, size=sizes[row] )
    if added_nodes.has_key(col_name) is False:
      added_nodes[ col_name ] = 1
      G.add_node( col_name, size=sizes[col] )
    
    if row_n == 0 or col_n == 0:    
      G.add_edge(row_name, col_name, weight=data)
      
      max_node_weight[row_name] = max( data, max_row_w )
      max_node_weight[col_name] = max( data, max_col_w )
      min_node_weight[row_name] = min( data, min_row_w )
      min_node_weight[col_name] = min( data, min_col_w )
      
    elif data > t and (row_n < max_nbr or col_n < max_nbr):
      G.add_edge(row_name, col_name, weight=data)
      
      max_node_weight[row_name] = max( data, max_row_w )
      max_node_weight[col_name] = max( data, max_col_w )
      min_node_weight[row_name] = min( data, min_row_w )
      min_node_weight[col_name] = min( data, min_col_w )
      
      # join row to col
      
      # if col_w < tau, remove col from neighbours unless it disconnects
      
    #pdb.set_trace()
    nbr+=1
    
    if nbr == 100:
      break
        
  #print "last row/col/n", row, col, n
  print "added %d edges"%nbr
  return G

def make_graph_from_labels( C, labels, names, sizes ):
  G=nx.Graph()

  nbr = 0
  default_size = 10
  default_weight = 1
  Ks = np.unique(labels)
  for K in Ks:
    
    I = pp.find(labels == K )
    
    cluster_name = "K%d"%K
    k_names = np.sort( names[I] )
    
    G.add_node( cluster_name, size = default_size*len(k_names) )
    for name in k_names:
      G.add_node( name, size = default_size )
      G.add_edge( cluster_name, name, weight=default_weight )

      nbr+=1

  print "added %d edges"%nbr
  return G
  
def make_graph_from_indices( indices, names ):
  G=nx.Graph()

  nbr = 0
  default_size = 10
  default_weight = 1
  
  # for ids in indices:
  #   G.add_node( names[ids[0]], size=default_size )
  #   #G.add_node( name, size=default_size )
  
  for ids in indices:
    i = ids[0]
    
    for j in ids[1:]:
      G.add_edge( names[i], names[j], weight=default_weight )

      nbr+=1

  print "added %d edges"%nbr
  return G
 
       
def draw_graph( G, save_name, node_colors=[], with_labels=True, alpha=1, font_size=10, figsize=(10,10) ):
  layout=g_layout
  try:
    pos=layout(G)
  except:
    pos = nx.spring_layout(G)
  pp.figure(figsize=figsize)
  nx.draw( G, pos, \
              node_color=node_colors, 
              with_labels=with_labels, 
              hold=with_labels, \
              alpha=alpha, \
              font_size=font_size
              )
  pp.savefig(save_name, fmt='png',dpi=300)
  pp.close('all')

def normalize_w( W, quantile = 0.01 ):
  # find significant weights per hidden unit
  I = np.argsort( -np.abs(W.values), axis = 0 )
  n = len(I)
  d = I.shape[1]
  m = int( quantile*n )
  I_m = I[:m,:]
  S = np.zeros( W.values.shape, dtype=int )
  
  for j in xrange( d ):
    S[ :, j ][ I_m[:,j] ] = 1
  
  S = pd.DataFrame( S, index=W.index, columns=W.columns)  
  #pdb.set_trace()  
  return S
  
def count_neighbours( S ):
  C = np.dot( S, S.T )
  
  order = np.argsort( -np.diag(C) )
  
  C = pd.DataFrame( C[order,:][:,order] , index = S.index.values[order], columns=S.index.values[order] )

  d = np.diag(C)
  I = pp.find(d>0)
  
  C = pd.DataFrame( C.values[I,:][:,I], index = C.index.values[I], columns = C.index.values[I] )
  
  return C

def normalize_counts( C ):
  d = np.diag(C).reshape( (len(C),1)).astype(float)
  CV = C.values.astype(float)
  #pdb.set_trace()
  CV /= d
  return pd.DataFrame(CV, index=C.index, columns=C.columns)  
     
# def cluster_genes_by_hidden_weights( data, correlation_thresholds = [0.99,0.9] ):
#   # take weight matrix, find small groupds of co-activated weights
#   W = pd.concat( data.W_input2h.values() )
#   W_norm = normalize_w( W, quantile=0.01 )
#   #W_norm = W / np.sqrt( ( W*W ).sum(0) )
#   print "computing hidden weight correlations..."
#   #pdb.set_trace()
#   C = count_neighbours( W_norm ) #W_norm.T.corr()
#   C2 = normalize_counts( C )
#   C = C.loc[ C2.index.values ]
#   C = C[ C2.columns ]
#   # csr = csr_matrix(1.0-C2.values)
#   #csr = csr_matrix(C2)
#   #pdb.set_trace()
#   coo = coo_matrix(C2)
#
#   data.input2input_w_corr = C2
#   data.coo = coo
#   data.C = C
#
#   #return None #data
#   #pdb.set_trace()
#   save_dir = os.path.join( data.save_dir, "input2hidden_weight_clustering3" )
#   check_and_mkdir(save_dir)
#   print "saving hidden weight correlations..."
#   data.input2input_w_corr.to_csv( save_dir + "/input2input_w_corr.csv" )
#
#   data.input2input_w_corr_graph = OrderedDict()
#   for correlation_threshold in correlation_thresholds:
#     print "making graph hidden weight correlations...", correlation_threshold
#     data.input2input_w_corr_graph[correlation_threshold] = make_graph_from_coo( data.coo, correlation_threshold, C2.columns, np.diag(C.values) )
#
#     draw_graph( data.input2input_w_corr_graph[correlation_threshold], save_dir+"/corr_%0.2f.png"%(correlation_threshold) )
#     G = data.input2input_w_corr_graph[correlation_threshold]
#
#     subgraphs = []
#     for subgraph in nx.connected_component_subgraphs(G):
#       nodes = subgraph.nodes()
#       subgraphs.append(nodes)
#
#     fptr = open( save_dir + "/subgraphs_%0.2f.yaml"%(correlation_threshold),"w+" )
#     fptr.write( yaml.dump(subgraphs))
#     fptr.close()
#     #pdb.set_trace()

def cluster_genes_by_hidden_weights_spectral( data, Ks = [400,200,100] ):
  results = {}
  save_dir = os.path.join( data.save_dir, "input2hidden_weight_clustering_spectral" )
  
  # take weight matrix, find small groupds of co-activated weights
  W = pd.concat( data.W_input2h.values() )
  W_norm_sig = normalize_w( W, quantile=0.01 )
  W_norm = W / np.sqrt( ( W*W ).sum(0) )
  
  print "computing hidden weight correlations..."
  #pdb.set_trace()
  C = count_neighbours( W_norm_sig ) #W_norm.T.corr()
  
  w_corr = W_norm.T.corr()
  w_abs_corr = np.abs(w_corr)
  check_and_mkdir(save_dir) 
  print "saving hidden weight correlations..."
  y = OrderedDict()
  M = OrderedDict()
  spectral_graph = OrderedDict()
  
  X = w_abs_corr
  for K in Ks:
    print "spectral clustering...", K
    M[K] = SpectralClustering(n_clusters=K, affinity="precomputed", n_init=20)
    y[K] = M[K].fit_predict( X )
    spectral_graph[K] = make_graph_from_labels( X, y[K], X.columns, np.diag(C.values) )
    
    draw_graph( spectral_graph[K], save_dir+"/clusters_K_%d.png"%(K) )
    
    G = spectral_graph[K]
    
    subgraphs = []
    
    #for subgraph in nx.connected_component_subgraphs(G):
    #K = len(np.unique(y))
    for k in range(K):
      I = pp.find( y[K]==k )  
      print len(I)
      #nodes = subgraph.nodes()
      nodes = list( np.sort( X.columns[I] ) )
      subgraphs.append(nodes)
    
    fptr = open( save_dir + "/clusters_K_%d.yaml"%(K),"w+" )
    #fptr = open( save_dir + "/K=%d_lists.yaml"%(K),"w+" )
    fptr.write( yaml.dump(subgraphs))
    fptr.close()
    
  results["C"] = C
  results["W"] = W
  results["w_corr"] = w_corr
  results["w_abs_corr"] = w_abs_corr
  results["w_norm_sig"] = W_norm_sig
  results["w_norm"] = W_norm
  results["M"] = M
  results["labels"] = y
  results["G"] = spectral_graph
  
  results["C"].to_csv( save_dir + "/C.csv" )
  results["W"].to_csv( save_dir + "/W.csv" )
  results["w_corr"].to_csv( save_dir + "/w_corr.csv" )
  results["w_abs_corr"].to_csv( save_dir + "/w_abs_corr.csv" )
  results["w_norm_sig"].to_csv( save_dir + "/w_norm_sig.csv" )
  results["w_norm"].to_csv( save_dir + "/w_norm.csv" )
  data.input2h_w_clustering = results
    #pdb.set_trace()

def cluster_genes_by_latent_weights_spectral( data, Ks = [400,200,100] ):
  results = {}
  save_dir = os.path.join( data.save_dir, "input2latent_weight_clustering_spectral" )
  
  # take weight matrix, find small groupds of co-activated weights
  W = pd.concat( data.weighted_W_h2z.values() )
  W_norm_sig = normalize_w( W, quantile=0.05 )
  W_norm = W / np.sqrt( ( W*W ).sum(0) )
  
  print "computing hidden weight correlations..."
  #pdb.set_trace()
  C = count_neighbours( W_norm_sig ) #W_norm.T.corr()
  
  w_corr = W_norm.T.corr()
  w_abs_corr = np.abs(w_corr)
  check_and_mkdir(save_dir) 
  print "saving hidden weight correlations..."
  y = OrderedDict()
  M = OrderedDict()
  spectral_graph = OrderedDict()
  
  X = w_abs_corr
  for K in Ks:
    print "spectral clustering...", K
    M[K] = SpectralClustering(n_clusters=K, affinity="precomputed", n_init=50)
    y[K] = M[K].fit_predict( X )
    spectral_graph[K] = make_graph_from_labels( X, y[K], X.columns, np.diag(C.values) )
    
    draw_graph( spectral_graph[K], save_dir+"/clusters_K_%d.png"%(K) )
    
    G = spectral_graph[K]
    
    subgraphs = []
    
    #for subgraph in nx.connected_component_subgraphs(G):
    for k in range(K):
      I = pp.find( y[K]==k )  
      #nodes = subgraph.nodes()
      nodes = list( np.sort( X.columns[I] ) )
      subgraphs.append(nodes)
    
    fptr = open( save_dir + "/clusters_K_%d.yaml"%(K),"w+" )
    fptr.write( yaml.dump(subgraphs))
    fptr.close()
    
  results["C"] = C
  results["W"] = W
  results["w_corr"] = w_corr
  results["w_abs_corr"] = w_abs_corr
  results["w_norm_sig"] = W_norm_sig
  results["w_norm"] = W_norm
  results["M"] = M
  results["labels"] = y
  results["G"] = spectral_graph
  
  results["C"].to_csv( save_dir + "/C.csv" )
  results["W"].to_csv( save_dir + "/W.csv" )
  results["w_corr"].to_csv( save_dir + "/w_corr.csv" )
  results["w_abs_corr"].to_csv( save_dir + "/w_abs_corr.csv" )
  results["w_norm_sig"].to_csv( save_dir + "/w_norm_sig.csv" )
  results["w_norm"].to_csv( save_dir + "/w_norm.csv" )
  data.input2z_w_clustering = results

def hidden_neighbours(data, nbr = 2):
  X = data.H
  
  save_dir = os.path.join( data.save_dir, "hidden_%d_nn"%(nbr) )
  check_and_mkdir(save_dir) 
  results = {}
  data.data_store.open()
  T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  data.data_store.close()
  tissue_pallette = sns.hls_palette(len(T.columns))
  #k_pallette[kp]
  
  bcs = X.index.values
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  
  
  V = X.values
  K=100
  nc=10; nr=10
  
  print "running kmeans"
  kmeans = MiniBatchKMeans(n_clusters=K, random_state=0).fit(V)
  yv = kmeans.labels_
  
  # print "making graph"
  # #MV = SpectralClustering(n_clusters=K, affinity="precomputed", n_init=50)
  # #yv = MV.fit_predict( V )
  #
  # GV=make_graph_from_labels( V, yv, X.index.values, [] )
  #
  # draw_graph(GV, save_dir+"/spectral.png" )

  subgraphs = []

  #for subgraph in nx.connected_component_subgraphs(G):
  f = pp.figure( figsize=(20,20))
  for k in range(K):
    I = pp.find( yv==k )  
    #nodes = subgraph.nodes()
    nodes = list( np.sort( X.index.values[I] ) )
    subgraphs.append(nodes)
    
    ax = f.add_subplot( nr,nc,k+1)
    print "working ", k
    for t_idx,tissue_name in zip( range(len(T.columns)), T.columns ):
      
      tids = pp.find( T[tissue_name]==1 )
      
      if len(tids)==0:
        continue
        
      ids = np.intersect1d( I,tids)
      if len(ids)<=5:
        continue
      kmf = KaplanMeierFitter()
      if k == K-1:
        kmf.fit(times[ids], event_observed=events[ids], label="%s"%(tissue_name)  )
      else:
        kmf.fit(times[ids], event_observed=events[ids]  )
      #pdb.set_trace()
      kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color=tissue_pallette[t_idx],ci_show=False,lw=2)
      #ax.set_ylim(0,1)

    kmf = KaplanMeierFitter()
    kmf.fit(times[I], event_observed=events[I], label=""  )
    kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color="black",ci_show=False,lw=4)
    ax.set_ylim(0,1)
    ax.set_xlim(0,6000)
    if k == K-1:
      ax.legend(loc=5)
    else:
      ax.legend([])
  pp.savefig(save_dir + "/survival_spectral.png", fmt='png',dpi=300)
  pp.close('all')
  
  fptr = open( save_dir + "/spectral_K_%d.yaml"%(K),"w+" )
  fptr.write( yaml.dump(subgraphs))
  fptr.close()
  #return
  
  print "generating KD tree"
  kdt =  KDTree(X.values, leaf_size=20, metric='euclidean')

  print "computing distances"
  distances, indices = kdt.query(X.values, k=nbr, return_distance=True)
  
  #KDTree(data.H.values, leaf_size=30, metric='euclidean')
  print "making graphs"
  data.data_store.open()
  T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  data.data_store.close()
  tissue_graphs = {}
  for tissue_name in T.columns:
    print "working ", tissue_name
    ids = pp.find( T[tissue_name]==1 )
    
    G = make_graph_from_indices( indices[ids,:nbr], X.index.values )
    print "  drawing graph"
    draw_graph( G, save_dir+"/nn_hidden_%s.png"%(tissue_name) )
    tissue_graphs[tissue_name] = G
    
    subgraphs = []
    
    for subgraph in nx.connected_component_subgraphs(G):
      nodes = subgraph.nodes()
      subgraphs.append(nodes)
    
    fptr = open( save_dir + "/hidden_clusters_%s.yaml"%(tissue_name),"w+" )
    fptr.write( yaml.dump(subgraphs))
    fptr.close()
  
  G = make_graph_from_indices( indices[:,:nbr], X.index.values )
  subgraphs = []
  
  for subgraph in nx.connected_component_subgraphs(G):
    nodes = subgraph.nodes()
    subgraphs.append(nodes)
  
  fptr = open( save_dir + "/FULL_hidden_clusters.yaml","w+" )
  fptr.write( yaml.dump(subgraphs))
  fptr.close()
  
  print "gathering results"
  results["distances"] = distances
  results["indices"] = indices
  results["tissue_G"] = tissue_graphs
  results["full_G"] = G
  results["save_dir"] = save_dir
  data.nn_hidden = results

def latent_neighbours(data, nbr = 2):
  save_dir = os.path.join( data.save_dir, "latent_%d_nn3"%(nbr) )
  check_and_mkdir(save_dir) 
  results = {}
  
  X = data.Z #[:5000]
  data.data_store.open()
  T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  data.data_store.close()
  tissue_pallette = sns.hls_palette(len(T.columns))
  #k_pallette[kp]
  
  bcs = X.index.values
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  #d_mat = pdist( X.values )
  #print "squareform DISTANCES..."
  #s_form = squareform(d_mat)
  #print "csr_matrix squareform..."
  #triu = np.triu(s_form)
  
  beta=20.0
  #V = np.exp( -s_form / beta)
  #V = np.abs( X.T.corr() )
  V = X.values
  K=20
  nc=4; nr=5
  
  print "running kmeans"
  kmeans = MiniBatchKMeans(n_clusters=K, random_state=0).fit(V)
  yv = kmeans.labels_
  
  # print "making graph"
  # #MV = SpectralClustering(n_clusters=K, affinity="precomputed", n_init=50)
  # #yv = MV.fit_predict( V )
  #
  # GV=make_graph_from_labels( V, yv, X.index.values, [] )
  #
  # draw_graph(GV, save_dir+"/spectral.png" )

  subgraphs = []

  #for subgraph in nx.connected_component_subgraphs(G):
  f = pp.figure( figsize=(20,20))
  for k in range(K):
    I = pp.find( yv==k )  
    #nodes = subgraph.nodes()
    nodes = list( np.sort( X.index.values[I] ) )
    subgraphs.append(nodes)
    
    ax = f.add_subplot( nr,nc,k+1)
    print "working ", k
    for t_idx,tissue_name in zip( range(len(T.columns)), T.columns ):
      
      tids = pp.find( T[tissue_name]==1 )
      
      if len(tids)==0:
        continue
        
      ids = np.intersect1d( I,tids)
      if len(ids)<=5:
        continue
      kmf = KaplanMeierFitter()
      if k == K-1:
        kmf.fit(times[ids], event_observed=events[ids], label="%s"%(tissue_name)  )
      else:
        kmf.fit(times[ids], event_observed=events[ids]  )
      #pdb.set_trace()
      kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color=tissue_pallette[t_idx],ci_show=False,lw=2)
      #ax.set_ylim(0,1)

    kmf = KaplanMeierFitter()
    kmf.fit(times[I], event_observed=events[I], label=""  )
    kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color="black",ci_show=False,lw=4)
    ax.set_ylim(0,1)
    ax.set_xlim(0,6000)
    if k == K-1:
      ax.legend(loc=5)
    else:
      ax.legend([])
  pp.savefig(save_dir + "/survival_spectral.png", fmt='png',dpi=300)
  pp.close('all')
  
  fptr = open( save_dir + "/spectral_K_%d.yaml"%(K),"w+" )
  fptr.write( yaml.dump(subgraphs))
  fptr.close()
  #return
  #pdb.set_trace()
  
  print "generating KD tree"
  kdt =  KDTree(X.values, leaf_size=20, metric='euclidean')

  print "computing distances"
  distances, indices = kdt.query(X.values, k=nbr, return_distance=True)
  
  #KDTree(data.H.values, leaf_size=30, metric='euclidean')
  print "making graphs"
  tissue_graphs = {}
  for tissue_name in T.columns:
    print "working ", tissue_name
    ids = pp.find( T[tissue_name]==1 )
    if len(ids)==0:
      continue
    G = make_graph_from_indices( indices[ids,:nbr], X.index.values )
    print "  drawing graph"
    draw_graph( G, save_dir+"/nn_latent_%s.png"%(tissue_name) )
    tissue_graphs[tissue_name] = G
    
    subgraphs = []
    
    for subgraph in nx.connected_component_subgraphs(G):
      nodes = subgraph.nodes()
      subgraphs.append(nodes)
    
    fptr = open( save_dir + "/latent_clusters_%s.yaml"%(tissue_name),"w+" )
    fptr.write( yaml.dump(subgraphs))
    fptr.close()
  
  G = make_graph_from_indices( indices[:,:nbr], X.index.values )
  subgraphs = []
  
  for subgraph in nx.connected_component_subgraphs(G):
    nodes = subgraph.nodes()
    subgraphs.append(nodes)
  
  fptr = open( save_dir + "/FULL_latent_clusters.yaml","w+" )
  fptr.write( yaml.dump(subgraphs))
  fptr.close()
    
  print "gathering results"
  results["distances"] = distances
  results["indices"] = indices
  results["tissue_G"] = tissue_graphs
  results["full_G"] = G
  results["save_dir"] = save_dir
  data.nn_latent = results
            
def cluster_genes_by_hidden_correlations(data):
  # for each hidden activation, find most correlated genes
  pass
  
def cluster_genes_by_weighted_latents(data):
  # take weighted hiddens, then weight from hidden to latent means, cluster these
  pass

def cluster_genes_by_correlated_latents(data):
  # find inputs most correlated with latent activations
  pass  

def cluster_patient_by_hiddens(data):
  # find correlations etween patients using hidden activations
  pass
  
def cluster_patients_by_latent(data):
  # find correlations between patients using latent space activations
  pass
  
  
# analyses:  find what makes two patients "neighbours" at local/global level
# cluster patients within and wthout cohorts.  how much difference?
# find survival directions, survival clusters

def neighbour_differences(data, X, Ws, title, nbr = 10):
  #X = data.H
  X=X[:100]
  W = pd.concat( Ws.values() )
  I_W = np.argsort( W, 0 )
  
  orders = []
  for h in I_W.columns:
    orders.append( I_W[h].sort_values() )
    
  save_dir = os.path.join( data.save_dir, "%s_nn_differences_K%d"%(title,nbr) )
  check_and_mkdir(save_dir) 
  results = {}
  data.data_store.open()
  T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  data.data_store.close()
  tissue_pallette = sns.hls_palette(len(T.columns))
  bcs = X.index.values
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  
  print "generating KD tree"
  kdt =  KDTree(X.values, leaf_size=20, metric='euclidean')

  print "computing distances"
  distances, indices = kdt.query(X.values, k=nbr, return_distance=True)
  
  #KDTree(data.H.values, leaf_size=30, metric='euclidean')
  print "making graphs"
  data.data_store.open()
  T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  data.data_store.close()
  tissue_graphs = {}
  for tissue_name in T.columns:
    print "working ", tissue_name
    ids = pp.find( T[tissue_name]==1 )
    
    subgraphs = []
    for idx in ids:
      this_indices = indices[idx,:]
      this_distances = distances[idx,:]
      bc = bcs[this_indices[0]]
      r = {}
      r["a_node"]      = bc
      r["a_p_event"]      = int( events[this_indices[0]] ) #[int(data.survival.data.loc[bc].values[0])
      r["a_p_time"]       = int( times[this_indices[0]] ) #int(data.survival.data.loc[bc].values[1])
      #r["neighbors"] = []
      neighbors = []
      x = X.values[this_indices[0]]
      for other_index, other_distance in zip(this_indices[1:], this_distances[1:]):
        n = {}
        bc = bcs[other_index]
        n["b_node"] = bc
        n["distance"] = float(other_distance)
        n["b_s_event"]      = int( events[other_index])# int(data.survival.data.loc[bc].values[0])
        n["b_s_time"]       = int( times[other_index] ) #int(data.survival.data.loc[bc].values[1])
        #n = {"node":bcs[other_index],"distance":float(other_distance)}
        y = X.values[other_index]
        
        most_similar = np.argsort( np.square( x-y) )
        n["most_similar"] = []
        n["top_gene"] = []
        for j in range(10):
          h_idx = most_similar[j]
          #h_order = I_W["h_%d"%(h_idx)].sort_values()
          n["most_similar"].append( int(h_idx) )
          #pdb.set_trace()
          n["top_gene"].append( orders[h_idx].index[0]  )
          #n["s_event"].append( )
        #n["most_similar"] = list(most_similar[:10].astype(int))
        neighbors.append(n)
      #pdb.set_trace()
      r["neighbors"] = neighbors
      subgraphs.append(r)
    
    #G = make_graph_from_indices( indices[ids,:nbr], X.index.values )
    print "  drawing graph"
    #draw_graph( G, save_dir+"/nn_hidden_%s.png"%(tissue_name) )
    #tissue_graphs[tissue_name] = G
    
    #subgraphs = []
    #
    # for subgraph in nx.connected_component_subgraphs(G):
    #   nodes = subgraph.nodes()
    #   subgraphs.append(nodes)
    
    fptr = open( save_dir + "/%s_clusters_%s.yaml"%(title,tissue_name),"w+" )
    fptr.write( yaml.dump(subgraphs))
    fptr.close()
  
  G = make_graph_from_indices( indices[:,:nbr], X.index.values )
  subgraphs = []
  
  for subgraph in nx.connected_component_subgraphs(G):
    nodes = subgraph.nodes()
    subgraphs.append(nodes)
  
  fptr = open( save_dir + "/FULL_%s_clusters.yaml"%title,"w+" )
  fptr.write( yaml.dump(subgraphs))
  fptr.close()
  
  print "gathering results"
  results["distances"] = distances
  results["indices"] = indices
  results["tissue_G"] = tissue_graphs
  results["full_G"] = G
  results["save_dir"] = save_dir
  data.results["nn_%s_dif"%title] = results

def within_tissue_neighbour_differences(data, X, Ws, title, nbr = 10):
  #X = data.H
  #X=X[:100]
  W = pd.concat( Ws.values() )
  I_W = np.argsort( W, 0 )
  
  orders = []
  for h in I_W.columns:
    orders.append( I_W[h].sort_values() )
    
  save_dir = os.path.join( data.save_dir, "%s_nn_differences_K%d"%(title,nbr) )
  check_and_mkdir(save_dir) 
  results = {}
  data.data_store.open()
  T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  data.data_store.close()
  tissue_pallette = sns.hls_palette(len(T.columns))
  bcs = X.index.values
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  good = pp.find( np.isnan(times) == False )
  #pdb.set_trace()

  bcs = bcs[good]
  X = X.loc[bcs]
  T = T.loc[bcs]
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  data.data_store.open()
  dna = data.data_store["/DNA/channel/0"].loc[bcs].fillna(0)
  data.data_store.close()
  #KDTree(data.H.values, leaf_size=30, metric='euclidean')
  print "making graphs"
  data.data_store.open()
  T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  data.data_store.close()
  tissue_graphs = {}
  for tissue_name in T.columns:
    print "working ", tissue_name
    ids = pp.find( T[tissue_name]==1 )
    
    if len(ids)==0:
      continue
    print "  generating KD tree"
    kdt =  KDTree(X.values[ids,:], leaf_size=20, metric='euclidean')

    print "  computing distances"
    distances, indices = kdt.query(X.values[ids,:], k=nbr, return_distance=True)
    
    subgraphs = []
    mean_times = []
    for this_indices,this_distances in zip(indices, distances):
      
      #this_indices = indices[idx,:]
      #this_distances = distances[idx,:]
      #pdb.set_trace()
      bc = bcs[ ids[this_indices[0]] ]
      r = {}
      r["a_node"]      = bc
      try:
        r["a_p_event"]      = int( events[ids[ this_indices[0]]] ) 
      except:
        r["a_p_event"]      = np.nan
      
      try:  
        r["a_p_time"]       = int( times[ids[this_indices[0]]] ) 
      except:
        r["a_p_time"]       = np.nan
      if r["a_p_event"] <1:
        continue
      r["b_events"] = []
      r["b_times"] = []
      top_dna_genes = Counter()
      neighbors = []
      x = X.values[ids[this_indices[0]]]
      
      #pdb.set_trace()
      dna_series = dna.loc[bc].sort_values()[-10:]

      top_dna_genes.update(dict( zip(dna_series.index.values, dna_series.values) ))
      
      top_genes=Counter()
      n_bcs = []
      for other_index, other_distance in zip(this_indices[1:], this_distances[1:]):
        n = {}
        bc = bcs[ids[other_index]]
        dna_series = dna.loc[bc].sort_values()[-10:]
        top_dna_genes.update(dict( zip(dna_series.index.values, dna_series.values) ))
        n["b_node"] = bc
        n_bcs.append(bc)
        n["distance"] = float(other_distance)
        try:
          n["b_s_event"]      = int( events[ids[other_index]])
        except:
          n["b_s_event"]      = np.nan  
          # int(data.survival.data.loc[bc].values[0])
        r["b_events"].append( n["b_s_event"] )
        try:
          n["b_s_time"]       = int( times[ids[other_index]] )
        except:
          n["b_s_time"]       = np.nan
        r["b_times"].append( n["b_s_time"] ) #int(data.survival.data.loc[bc].values[1])
        #n = {"node":bcs[other_index],"distance":float(other_distance)}
        y = X.values[ids[other_index]]
        
        most_similar = np.argsort( np.square( x-y) )
        n["most_similar"] = []
        n["top_gene"] = []
        
        for j in range(20):
          h_idx = most_similar[j]
          #h_order = I_W["h_%d"%(h_idx)].sort_values()
          n["most_similar"].append( int(h_idx) )
          #pdb.set_trace()
          n["top_gene"].extend( orders[h_idx].index[:20]  )
          
          #n["s_event"].append( )
        #n["most_similar"] = list(most_similar[:10].astype(int))
        neighbors.append(n)
        top_genes.update(n["top_gene"])
      #pdb.set_trace()
      
      r["a_top_genes"] = []
      for gene,cnt in top_genes.most_common()[:10]:
        if cnt>2:
          r["a_top_genes"].append(gene)
      r["a_top_dna"] = []
      r["a_top_dna_cnt"] = []
      for gene,cnt in top_dna_genes.most_common()[:20]:
        r["a_top_dna_cnt"].append( int(cnt) )
        r["a_top_dna"].append( gene )
      
      r["neighbors"] = n_bcs #neighbors
      
      r["mean_time"] = float(np.mean(r["b_times"]))
      r["mean_event_time"] = float(np.sum(r["b_events"])) #float(np.sum( np.array(r["b_times"])*np.array(r["b_events"])/float(np.sum(r["b_events"]))))
      mean_times.append( r["mean_event_time"] )
      #mean_times.append( r["a_p_time"] )
      subgraphs.append(r)
    mean_times = np.array(mean_times)
    #G = make_graph_from_indices( indices[ids,:nbr], X.index.values )
    print "  drawing graph"
    subgraphs2 = [subgraphs[jj] for jj in np.argsort(mean_times)]
    
    bsc_high = [ subgraphs2[0]["a_node"] ]
    bsc_high.extend( subgraphs2[0]["neighbors"] )
    bsc_high.extend( subgraphs2[1]["neighbors"] )
    bsc_high.extend( [ subgraphs2[1]["a_node"] ] )
    bsc_high = np.unique(bsc_high)
    
    bsc_low = [ subgraphs2[-1]["a_node"] ]
    bsc_low.extend( subgraphs2[-1]["neighbors"] )
    bsc_low.extend( subgraphs2[-2]["neighbors"] )
    bsc_low.extend( [ subgraphs2[-2]["a_node"] ] )
    bsc_low = np.unique(bsc_low)
    
    s_low = data.survival.data.loc[ bsc_low ] #["T"].values
    s_high = data.survival.data.loc[ bsc_high ]
    #events = data.survival.data.loc[ bcs ]["E"].values
    
    f = pp.figure()
    ax = f.add_subplot(111)
    kmf = KaplanMeierFitter()
    kmf.fit(s_low["T"].values, event_observed=s_low["E"].values, label="low %s"%(tissue_name)  )
    kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color="blue",ci_show=False,lw=2)
    
    kmf.fit(s_high["T"].values, event_observed=s_high["E"].values, label="high %s"%(tissue_name)  )
    kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color="red",ci_show=False,lw=2)
    pp.savefig( save_dir + "/survival_%s.png"%(tissue_name), fmt="png", dpi=300 )
    pp.close('all')
    #draw_graph( G, save_dir+"/nn_hidden_%s.png"%(tissue_name) )
    #tissue_graphs[tissue_name] = G
    
    #subgraphs = []
    #
    # for subgraph in nx.connected_component_subgraphs(G):
    #   nodes = subgraph.nodes()
    #   subgraphs.append(nodes)
    
    fptr = open( save_dir + "/by_tissue_%s_clusters_%s.yaml"%(title,tissue_name),"w+" )
    fptr.write( yaml.dump(subgraphs2))
    fptr.close()
    
    
  
  # G = make_graph_from_indices( indices[:,:nbr], X.index.values )
  # subgraphs = []
  #
  # for subgraph in nx.connected_component_subgraphs(G):
  #   nodes = subgraph.nodes()
  #   subgraphs.append(nodes)
  #
  # fptr = open( save_dir + "/FULL_%s_clusters.yaml"%title,"w+" )
  # fptr.write( yaml.dump(subgraphs))
  # fptr.close()
  #
  print "gathering results"
  results["distances"] = distances
  results["indices"] = indices
  results["tissue_G"] = tissue_graphs
  #results["full_G"] = G
  results["save_dir"] = save_dir
  data.results["nn_by_tissue_%s_dif"%title] = results    

def cosine_within_tissue_neighbour_differences(data, X, Ws, title, nbr = 10):
  #X = data.H
  #X=X[:100]
  W = pd.concat( Ws.values() )
  I_W = np.argsort( W, 0 )
  
  orders = []
  for h in I_W.columns:
    orders.append( I_W[h].sort_values() )
    
  save_dir = os.path.join( data.save_dir, "cosine_%s_nn_differences_K%d"%(title,nbr) )
  check_and_mkdir(save_dir) 
  results = {}
  data.data_store.open()
  #pdb.set_trace()
  try:
    T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  except:
    #pdb.set_trace()
    T=data.data_store["/CLINICAL/TISSUE"].loc[ X.index ]
  data.data_store.close()
  tissue_pallette = sns.hls_palette(len(T.columns))
  bcs = X.index.values
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  good = pp.find( np.isnan(times) == False )
  #pdb.set_trace()

  bcs = bcs[good]
  X = X.loc[bcs]
  T = T.loc[bcs]
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  data.data_store.open()
  dna = data.data_store["/DNA/channel/0"].loc[bcs].fillna(0)
  data.data_store.close()
  #KDTree(data.H.values, leaf_size=30, metric='euclidean')
  print "making graphs"
  # data.data_store.open()
  # T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  # data.data_store.close()
  tissue_graphs = {}
  for tissue_name in T.columns:
    print "working ", tissue_name
    ids = pp.find( T[tissue_name]==1 )
    
    if len(ids)==0:
      continue
      
    print "  computing cosine distance"
    XN = X.values[ids,:]
    XN = XN / np.sqrt(np.sum( XN*XN,1)[:,np.newaxis])
    cosine_distance = 1.0 - np.dot( XN, XN.T )
    cosine_distance = np.maximum(cosine_distance,0)
    indices = np.argsort( cosine_distance, 1 )[:,:nbr]
    distances = []

    print " computing indices"
    
    subgraphs = []
    mean_times = []
    for this_indices in indices:
      
      #this_indices = indices[idx,:]
      this_distances = cosine_distance[ this_indices[0], :][ this_indices ]
      #pdb.set_trace()
      bc = bcs[ ids[this_indices[0]] ]
      r = {}
      r["a_node"]      = bc
      try:
        r["a_p_event"]      = int( events[ids[ this_indices[0]]] ) 
      except:
        r["a_p_event"]      = np.nan
      
      try:  
        r["a_p_time"]       = int( times[ids[this_indices[0]]] ) 
      except:
        r["a_p_time"]       = np.nan
      if r["a_p_event"] <1:
        continue
      r["b_events"] = []
      r["b_times"] = []
      top_dna_genes = Counter()
      neighbors = []
      x = X.values[ids[this_indices[0]]]
      
      #pdb.set_trace()
      dna_series = dna.loc[bc].sort_values()[-10:]

      top_dna_genes.update(dict( zip(dna_series.index.values, dna_series.values) ))
      
      top_genes=Counter()
      n_bcs = []
      for other_index, other_distance in zip(this_indices[1:], this_distances[1:]):
        n = {}
        bc = bcs[ids[other_index]]
        dna_series = dna.loc[bc].sort_values()[-10:]
        top_dna_genes.update(dict( zip(dna_series.index.values, dna_series.values) ))
        n["b_node"] = bc
        n_bcs.append(bc)
        n["distance"] = float(other_distance)
        try:
          n["b_s_event"]      = int( events[ids[other_index]])
        except:
          n["b_s_event"]      = np.nan  
          # int(data.survival.data.loc[bc].values[0])
        r["b_events"].append( n["b_s_event"] )
        try:
          n["b_s_time"]       = int( times[ids[other_index]] )
        except:
          n["b_s_time"]       = np.nan
        r["b_times"].append( n["b_s_time"] ) #int(data.survival.data.loc[bc].values[1])
        #n = {"node":bcs[other_index],"distance":float(other_distance)}
        y = X.values[ids[other_index]]
        
        most_similar = np.argsort( np.square( x-y) )
        n["most_similar"] = []
        n["top_gene"] = []
        
        for j in range(20):
          h_idx = most_similar[j]
          #h_order = I_W["h_%d"%(h_idx)].sort_values()
          n["most_similar"].append( int(h_idx) )
          #pdb.set_trace()
          n["top_gene"].extend( orders[h_idx].index[:20]  )
          
          #n["s_event"].append( )
        #n["most_similar"] = list(most_similar[:10].astype(int))
        neighbors.append(n)
        top_genes.update(n["top_gene"])
      #pdb.set_trace()
      
      r["a_top_genes"] = []
      for gene,cnt in top_genes.most_common()[:10]:
        if cnt>2:
          r["a_top_genes"].append(gene)
      r["a_top_dna"] = []
      r["a_top_dna_cnt"] = []
      for gene,cnt in top_dna_genes.most_common()[:20]:
        r["a_top_dna_cnt"].append( int(cnt) )
        r["a_top_dna"].append( gene )
      
      r["neighbors"] = n_bcs #neighbors
      
      r["mean_time"] = float(np.mean(r["b_times"]))
      r["mean_event_time"] = float(np.sum(r["b_events"])) #float(np.sum( np.array(r["b_times"])*np.array(r["b_events"])/float(np.sum(r["b_events"]))))
      mean_times.append( r["mean_time"] )
      #mean_times.append( r["a_p_time"] )
      subgraphs.append(r)
    mean_times = np.array(mean_times)
    #G = make_graph_from_indices( indices[ids,:nbr], X.index.values )
    print "  drawing graph"
    subgraphs2 = [subgraphs[jj] for jj in np.argsort(mean_times)]
    
    bsc_high = [ subgraphs2[0]["a_node"] ]
    bsc_high.extend( subgraphs2[0]["neighbors"] )
    bsc_high.extend( subgraphs2[1]["neighbors"] )
    bsc_high.extend( [ subgraphs2[1]["a_node"] ] )
    bsc_high = np.unique(bsc_high)
    
    bsc_low = [ subgraphs2[-1]["a_node"] ]
    bsc_low.extend( subgraphs2[-1]["neighbors"] )
    bsc_low.extend( subgraphs2[-2]["neighbors"] )
    bsc_low.extend( [ subgraphs2[-2]["a_node"] ] )
    bsc_low = np.unique(bsc_low)
    
    s_low = data.survival.data.loc[ bsc_low ] #["T"].values
    s_high = data.survival.data.loc[ bsc_high ]
    #events = data.survival.data.loc[ bcs ]["E"].values
    
    f = pp.figure()
    ax = f.add_subplot(111)
    kmf = KaplanMeierFitter()
    kmf.fit(s_low["T"].values, event_observed=s_low["E"].values, label="low %s"%(tissue_name)  )
    kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color="blue",ci_show=False,lw=2)
    
    kmf.fit(s_high["T"].values, event_observed=s_high["E"].values, label="high %s"%(tissue_name)  )
    kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color="red",ci_show=False,lw=2)
    pp.savefig( save_dir + "/survival_%s.png"%(tissue_name), fmt="png", dpi=300 )
    pp.close('all')

    
    fptr = open( save_dir + "/by_tissue_%s_clusters_%s.yaml"%(title,tissue_name),"w+" )
    fptr.write( yaml.dump(subgraphs2))
    fptr.close()
  #
  print "gathering results"
  results["distances"] = distances
  results["indices"] = indices
  results["tissue_G"] = tissue_graphs
  #results["full_G"] = G
  results["save_dir"] = save_dir
  data.results["cosine_nn_by_tissue_%s_dif"%title] = results    

  
# def latent_neighbour_differences(data, nbr = 10):
#   X = data.Z#[:100]
#   Ws = data.W_input2h
#
#   save_dir = os.path.join( data.save_dir, "latent_nn_differences_K%d"%(nbr) )
#   check_and_mkdir(save_dir)
#   results = {}
#   data.data_store.open()
#   T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
#   data.data_store.close()
#   tissue_pallette = sns.hls_palette(len(T.columns))
#   bcs = X.index.values
#   times = data.survival.data.loc[ bcs ]["T"].values
#   events = data.survival.data.loc[ bcs ]["E"].values
#
#
#   print "generating KD tree"
#   kdt =  KDTree(X.values, leaf_size=20, metric='euclidean')
#
#   print "computing distances"
#   distances, indices = kdt.query(X.values, k=nbr, return_distance=True)
#
#   #KDTree(data.H.values, leaf_size=30, metric='euclidean')
#   print "making graphs"
#   data.data_store.open()
#   T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
#   data.data_store.close()
#   tissue_graphs = {}
#   for tissue_name in T.columns:
#     print "working ", tissue_name
#     ids = pp.find( T[tissue_name]==1 )
#
#     subgraphs = []
#     for idx in ids:
#       this_indices = indices[idx,:]
#       this_distances = distances[idx,:]
#
#       r = {}
#       r["a_node"]      = bcs[this_indices[0]]
#       #r["neighbors"] = []
#       neighbors = []
#
#       x = X.values[this_indices[0]]
#       for other_index, other_distance in zip(this_indices[1:], this_distances[1:]):
#         n = {}
#         n["b_node"] = bcs[other_index]
#         n["distance"] = float(other_distance)
#         #n = {"node":bcs[other_index],"distance":float(other_distance)}
#         y = X.values[other_index]
#
#         most_similar = np.argsort( np.square( x-y) )
#         n["most_similar"] = []
#         for j in range(10):
#           n["most_similar"].append( int(most_similar[j]) )
#         #n["most_similar"] = list(most_similar[:10].astype(int))
#         neighbors.append(n)
#       #pdb.set_trace()
#       r["neighbors"] = neighbors
#       subgraphs.append(r)
#
#     #G = make_graph_from_indices( indices[ids,:nbr], X.index.values )
#     print "  drawing graph"
#     #draw_graph( G, save_dir+"/nn_hidden_%s.png"%(tissue_name) )
#     #tissue_graphs[tissue_name] = G
#
#     #subgraphs = []
#     #
#     # for subgraph in nx.connected_component_subgraphs(G):
#     #   nodes = subgraph.nodes()
#     #   subgraphs.append(nodes)
#
#     fptr = open( save_dir + "/latent_clusters_%s.yaml"%(tissue_name),"w+" )
#     fptr.write( yaml.dump(subgraphs))
#     fptr.close()
#
#   G = make_graph_from_indices( indices[:,:nbr], X.index.values )
#   subgraphs = []
#
#   for subgraph in nx.connected_component_subgraphs(G):
#     nodes = subgraph.nodes()
#     subgraphs.append(nodes)
#
#   fptr = open( save_dir + "/FULL_latent_clusters.yaml","w+" )
#   fptr.write( yaml.dump(subgraphs))
#   fptr.close()
#
#   print "gathering results"
#   results["distances"] = distances
#   results["indices"] = indices
#   results["tissue_G"] = tissue_graphs
#   results["full_G"] = G
#   results["save_dir"] = save_dir
#   data.nn_latent_dif = results
def  spearmanr_latent_space_by_inputs( data, force = False ):
  Z           = data.Z
  RNA_scale   = 2.0 / (1+np.exp(-data.RNA_scale)) -1   
  miRNA_scale = 2.0 / (1+np.exp(-data.miRNA_scale))-1 
  METH_scale  = 2.0 / (1+np.exp(-data.METH_scale ))-1  
  
  
  dna_names = data.dna.sum(0).sort_values(ascending=False).index.values
  n_dna = len(dna_names)
  dna = data.dna[ dna_names ]
  
  save_dir = os.path.join( data.save_dir, "spearmans_latent_tissue" )
  check_and_mkdir(save_dir)
  
  data.data_store.open()
  try:
    T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ Z.index ]
  except:
    T=data.data_store["/CLINICAL/TISSUE"].loc[ Z.index ]
  data.data_store.close()
  
  #pdb.set_trace()
  rna_names   = data.rna_names    
  mirna_names = data.mirna_names 
  meth_names  = data.meth_names   
  
  z_names = data.z_names
  n_rna   = len(rna_names)
  n_mirna = len(mirna_names)
  n_meth  = len(meth_names)


  barcodes = T.index.values
  
  try:
    rna_z_rho = pd.read_csv( save_dir + "/rna_z_rho.csv", index_col="gene" )
    rna_z_p   = pd.read_csv( save_dir + "/rna_z_p.csv", index_col="gene" )
  
    mirna_z_rho = pd.read_csv( save_dir + "/mirna_z_rho.csv", index_col="gene" )
    mirna_z_p   = pd.read_csv( save_dir + "/mirna_z_p.csv", index_col="gene" )
   
    meth_z_rho = pd.read_csv( save_dir + "/meth_z_rho.csv", index_col="gene" )
    meth_z_p   = pd.read_csv( save_dir + "/meth_z_p.csv", index_col="gene" )

    dna_z_rho = pd.read_csv( save_dir + "/dna_z_rho.csv", index_col="gene" )
    dna_z_p   = pd.read_csv( save_dir + "/dna_z_p.csv", index_col="gene" )
  
  except: 
    print "could not load, forcing..."  
    force=True
    
  if force is True:
    print "computing RNA-Z spearman rho's"
    rho_rna_z = stats.spearmanr( RNA_scale.values, Z.values )
    #pdb.set_trace()
    print "computing miRNA-Z spearman rho's"
    rho_mirna_z = stats.spearmanr( miRNA_scale.values, Z.values )
    print "computing METH-Z spearman rho's"
    rho_meth_z = stats.spearmanr( METH_scale.values, Z.values )
    print "computing DNA-Z spearman rho's"
    rho_dna_z = stats.spearmanr(2*dna.values-1, Z.values )
  
    rna_z_rho = pd.DataFrame( rho_rna_z[0][:n_rna,:][:,n_rna:], index = rna_names, columns=z_names)
    rna_z_p   = pd.DataFrame( rho_rna_z[1][:n_rna,:][:,n_rna:], index = rna_names, columns=z_names)
  
    mirna_z_rho = pd.DataFrame( rho_mirna_z[0][:n_mirna,:][:,n_mirna:], index = mirna_names, columns=z_names)
    mirna_z_p   = pd.DataFrame( rho_mirna_z[1][:n_mirna,:][:,n_mirna:], index = mirna_names, columns=z_names)
   
    meth_z_rho = pd.DataFrame( rho_meth_z[0][:n_meth,:][:,n_meth:], index = meth_names, columns=z_names)
    meth_z_p   = pd.DataFrame( rho_meth_z[1][:n_meth,:][:,n_meth:], index = meth_names, columns=z_names)

    dna_z_rho = pd.DataFrame( rho_dna_z[0][:n_dna,:][:,n_dna:], index = dna_names, columns=z_names)
    dna_z_p   = pd.DataFrame( rho_dna_z[1][:n_dna,:][:,n_dna:], index = dna_names, columns=z_names)

  
  rna_z_rho.to_csv( save_dir + "/rna_z_rho.csv", index_label="gene" )
  rna_z_p.to_csv( save_dir + "/rna_z_p.csv", index_label="gene" )
  
  mirna_z_rho.to_csv( save_dir + "/mirna_z_rho.csv", index_label="gene" )
  mirna_z_p.to_csv( save_dir + "/mirna_z_p.csv", index_label="gene" )
  
  meth_z_rho.to_csv( save_dir + "/meth_z_rho.csv", index_label="gene" )
  meth_z_p.to_csv( save_dir + "/meth_z_p.csv", index_label="gene" )
  
  dna_z_rho.to_csv( save_dir + "/dna_z_rho.csv", index_label="gene" )
  dna_z_p.to_csv( save_dir + "/dna_z_p.csv", index_label="gene" )
  
  f=pp.figure( figsize=(24,12) )
  
  nbr_genes = 20
  nbr_zs    = 10
  genes = dna_names[:nbr_genes]
  k_idx = 1
  for gene in genes:
    best_z_names = dna_z_p.loc[gene].sort_values()[:nbr_zs].index.values
    dna_values = dna[gene].values
    
    #ids_with_n = ids_with_at_least_n_mutations( dna_values, T, n = 5 )
    ids_with_n = ids_with_at_least_p_mutations( dna_values, T, p = 0.01 )
    
    barcodes_with_n = barcodes[ids_with_n]
    
    mutations = pp.find( dna_values[ids_with_n] == 1)
    wildtype = pp.find( dna_values[ids_with_n]==0)
    

    z_idx = 0
    for z_name in best_z_names:
      z_values = Z[z_name].loc[barcodes_with_n].values
      z_all_wild = Z[z_name].values[pp.find( dna_values==0)] 
      
      ax = f.add_subplot(nbr_genes, nbr_zs ,k_idx)

      ax.hist( z_all_wild, 30, normed=True,histtype="step", lw=1, color="black" )
      ax.hist( z_values[wildtype], 30, normed=True,histtype="step", lw=2, color="blue" )
      ax.hist( z_values[mutations], 15, normed=True,histtype="step", lw=2, color="red" )

      ax.set_title(gene+"-"+z_name)
      # if z_idx == 0:
      #   ax.set_ylabel(gene)
      # ax.set_xlabel(z_name)
      z_idx+=1
      k_idx+=1
  pp.savefig( save_dir + "/dna_top_z.png", fmt="png", dpi=300)

  z_scores = -np.sum( np.log2(dna_z_p),1)
  
  f=pp.figure( figsize=(24,12) )
  genes = z_scores.sort_values()[-nbr_genes:].index.values #dna_names[:nbr_genes]
  #pdb.set_trace()
  k_idx = 1
  for gene in genes:
    best_z_names = dna_z_p.loc[gene].sort_values()[:nbr_zs].index.values
    dna_values = dna[gene].values
    
    #ids_with_n = ids_with_at_least_n_mutations( dna_values, T, n = 5 )
    ids_with_n = ids_with_at_least_p_mutations( dna_values, T, p = 0.01 )
    
    barcodes_with_n = barcodes[ids_with_n]
    
    mutations = pp.find( dna_values[ids_with_n] == 1)
    wildtype = pp.find( dna_values[ids_with_n]==0)
    

    z_idx = 0
    for z_name in best_z_names:
      z_values = Z[z_name].loc[barcodes_with_n].values
      z_all_wild = Z[z_name].values[pp.find( dna_values==0)] 
      
      ax = f.add_subplot(nbr_genes, nbr_zs ,k_idx)

      ax.hist( z_all_wild, 30, normed=True,histtype="step", lw=1, color="black" )
      ax.hist( z_values[wildtype], 30, normed=True,histtype="step", lw=2, color="blue" )
      ax.hist( z_values[mutations], 15, normed=True,histtype="step", lw=2, color="red" )

      ax.set_title(gene+"-"+z_name)
      # if z_idx == 0:
      #   ax.set_ylabel(gene)
      # ax.set_xlabel(z_name)
      z_idx+=1
      k_idx+=1
  pp.savefig( save_dir + "/dna_top_genes.png", fmt="png", dpi=300)
    
  global_order = np.argsort( dna_z_p.values.flatten() )
  #
  rr = np.unravel_index(global_order[:nbr_genes*nbr_zs], dims=dna_z_p.values.shape )
  dna_s = dna_names[rr[0]]
  z_s = z_names[rr[1]]
  
  f=pp.figure( figsize=(24,12) )
  #order = np.argsort(dna_s)
  #dna_s = dna_s[order]
  #z_s = z_s[order]
  k_idx=1
  for gene, z_name in zip(dna_s,z_s):
    #best_z_names = dna_z_p.loc[gene].sort_values()[:nbr_zs].index.values
    dna_values = dna[gene].values
    # mutations = pp.find( dna_values == 1)
    # wildtype = pp.find( dna_values==0)
    #
    # z_values = Z[z_name].values
    
    #ids_with_n = ids_with_at_least_n_mutations( dna_values, T, n = 5 )
    ids_with_n = ids_with_at_least_p_mutations( dna_values, T, p = 0.01 )
    
    #if gene == "APC":
    #  pdb.set_trace()
    barcodes_with_n = barcodes[ids_with_n]
    
    mutations = pp.find( dna_values[ids_with_n] == 1)
    wildtype = pp.find( dna_values[ids_with_n]==0)

    z_values = Z[z_name].loc[barcodes_with_n].values

    z_all_wild = Z[z_name].values[pp.find( dna_values==0)] 

    ax = f.add_subplot(nbr_genes, nbr_zs ,k_idx)

    #ax.hist( z_all_wild, 30, normed=True,histtype="step", lw=1, color="black" )
    ax.hist( z_values[wildtype], 30, normed=True,histtype="step", lw=2, color="blue" )
    ax.hist( z_values[mutations], 15, normed=True,histtype="step", lw=2, color="red" )
    ax.set_title(gene+"-"+z_name)
    ax.set_xlabel("")
    k_idx+=1
  pp.savefig( save_dir + "/dna_top_z2.png", fmt="png", dpi=300)


  
def  correlation_latent_space_by_inputs( data, force = False ):
  Z           = data.Z
  RNA_scale   = 2.0 / (1+np.exp(-data.RNA_scale)) -1   
  miRNA_scale = 2.0 / (1+np.exp(-data.miRNA_scale))-1 
  METH_scale  = 2.0 / (1+np.exp(-data.METH_scale ))-1  
  
  n_dna = 200
  dna_names = data.dna.sum(0).sort_values(ascending=False)[:n_dna].index.values
  dna = data.dna[ dna_names ]
  
  save_dir = os.path.join( data.save_dir, "pearsons_latent" )
  check_and_mkdir(save_dir)
  
  
  #pdb.set_trace()
  rna_names   = data.rna_names    
  mirna_names = data.mirna_names 
  meth_names  = data.meth_names   
  
  z_names = data.z_names
  n_rna   = len(rna_names)
  n_mirna = len(mirna_names)
  n_meth  = len(meth_names)


  try:
    rna_z_rho = pd.read_csv( save_dir + "/rna_z_rho.csv", index_col="gene" )
    rna_z_p   = pd.read_csv( save_dir + "/rna_z_p.csv", index_col="gene" )
  
    mirna_z_rho = pd.read_csv( save_dir + "/mirna_z_rho.csv", index_col="gene" )
    mirna_z_p   = pd.read_csv( save_dir + "/mirna_z_p.csv", index_col="gene" )
   
    meth_z_rho = pd.read_csv( save_dir + "/meth_z_rho.csv", index_col="gene" )
    meth_z_p   = pd.read_csv( save_dir + "/meth_z_p.csv", index_col="gene" )

    dna_z_rho = pd.read_csv( save_dir + "/dna_z_rho.csv", index_col="gene" )
    dna_z_p   = pd.read_csv( save_dir + "/dna_z_p.csv", index_col="gene" )
  
  except: 
    print "could not load, forcing..."  
    force=True
    
  if force is True:
    print "computing RNA-Z pearsonr rho's"
    rho_rna_z = pearsonr( RNA_scale.values, Z.values )
    #pdb.set_trace()
    print "computing miRNA-Z pearsonr rho's"
    rho_mirna_z = pearsonr( miRNA_scale.values, Z.values )
    print "computing METH-Z pearsonr rho's"
    rho_meth_z = pearsonr( METH_scale.values, Z.values )
    print "computing DNA-Z pearsonr rho's"
    rho_dna_z = pearsonr(2*dna.values-1, Z.values )
    # print "computing RNA-Z spearman rho's"
    # rho_rna_z = stats.spearmanr( RNA_scale.values, Z.values )
    # print "computing miRNA-Z spearman rho's"
    # rho_mirna_z = stats.spearmanr( miRNA_scale.values, Z.values )
    # print "computing METH-Z spearman rho's"
    # rho_meth_z = stats.spearmanr( METH_scale.values, Z.values )
    # print "computing DNA-Z spearman rho's"
    # rho_dna_z = stats.spearmanr(2*dna.values-1, Z.values )
  
    rna_z_rho = pd.DataFrame( rho_rna_z[0], index = rna_names, columns=z_names)
    rna_z_p   = pd.DataFrame( rho_rna_z[1], index = rna_names, columns=z_names)
  
    mirna_z_rho = pd.DataFrame( rho_mirna_z[0], index = mirna_names, columns=z_names)
    mirna_z_p   = pd.DataFrame( rho_mirna_z[1], index = mirna_names, columns=z_names)
   
    meth_z_rho = pd.DataFrame( rho_meth_z[0], index = meth_names, columns=z_names)
    meth_z_p   = pd.DataFrame( rho_meth_z[1], index = meth_names, columns=z_names)

    dna_z_rho = pd.DataFrame( rho_dna_z[0], index = dna_names, columns=z_names)
    dna_z_p   = pd.DataFrame( rho_dna_z[1], index = dna_names, columns=z_names)

  
  rna_z_rho.to_csv( save_dir + "/rna_z_rho.csv", index_label="gene" )
  rna_z_p.to_csv( save_dir + "/rna_z_p.csv", index_label="gene" )
  
  mirna_z_rho.to_csv( save_dir + "/mirna_z_rho.csv", index_label="gene" )
  mirna_z_p.to_csv( save_dir + "/mirna_z_p.csv", index_label="gene" )
  
  meth_z_rho.to_csv( save_dir + "/meth_z_rho.csv", index_label="gene" )
  meth_z_p.to_csv( save_dir + "/meth_z_p.csv", index_label="gene" )
  
  dna_z_rho.to_csv( save_dir + "/dna_z_rho.csv", index_label="gene" )
  dna_z_p.to_csv( save_dir + "/dna_z_p.csv", index_label="gene" )
  
  f = pp.figure( figsize=(20,20) )
  
  nbr_genes = 15
  nbr_zs    = 15
  genes = dna_names[:nbr_genes]
  k_idx = 1
  for gene in genes:
    best_z_names = dna_z_p.loc[gene].sort_values()[:nbr_zs].index.values
    dna_values = dna[gene].values
    mutations = pp.find( dna_values == 1)
    wildtype = pp.find( dna_values==0)

    z_idx = 0
    for z_name in best_z_names:
      z_values = Z[z_name].values
      ax = f.add_subplot(nbr_genes,nbr_zs,k_idx)

      ax.hist( z_values[wildtype], 20, normed=True,histtype="step", lw=2, color="blue" )
      ax.hist( z_values[mutations], 20, normed=True,histtype="step", lw=2, color="red" )

      if z_idx == 0:
        ax.set_ylabel(gene)
      ax.set_xlabel(z_name)
      z_idx+=1
      k_idx+=1
  pp.savefig( save_dir + "/dna_top_z.png", fmt="png", dpi=300)
  
  global_order = np.argsort( dna_z_p.values.flatten() )
  #
  rr = np.unravel_index(global_order[:nbr_genes*nbr_zs], dims=dna_z_p.values.shape )
  dna_s = dna_names[rr[0]]
  z_s = z_names[rr[1]]
  
  f = pp.figure( figsize=(20,20) ) 
  #order = np.argsort(dna_s)
  #dna_s = dna_s[order]
  #z_s = z_s[order]
  k_idx=1
  for gene, z_name in zip(dna_s,z_s):
    best_z_names = dna_z_p.loc[gene].sort_values()[:nbr_zs].index.values
    dna_values = dna[gene].values
    mutations = pp.find( dna_values == 1)
    wildtype = pp.find( dna_values==0)

    z_values = Z[z_name].values
    ax = f.add_subplot(nbr_genes,nbr_zs,k_idx)

    ax.hist( z_values[wildtype], 20, normed=True,histtype="step", lw=2, color="blue" )
    ax.hist( z_values[mutations], 20, normed=True,histtype="step", lw=2, color="red" )
    ax.set_title(gene+"-"+z_name)
    ax.set_xlabel("")
    k_idx+=1
  pp.savefig( save_dir + "/dna_top_z2.png", fmt="png", dpi=300)  
  #pdb.set_trace()


  
def auc_p_tissue_filter( dna, Z, T, p = 0.01 ):
  
  
  dna_names = dna.columns
  z_names   = Z.columns
  barcodes = Z.index.values
  
  n_z = len(z_names)
  n_dna = len(dna_names)
  
  dna_auc      = 0.5*np.ones( (n_dna,n_z) )
  dna_p_values = np.ones( (n_dna,n_z) )
  
  d_idx = 0
  z_idx = 0
  for dna_gene in dna_names:
    dna_values = dna[dna_gene].values
    ids_with_n = ids_with_at_least_p_mutations( dna_values, T, p=p )
    
    if np.sum(ids_with_n)==0:
      print "skipping ",dna_gene
      continue
      
    barcodes_with_n = barcodes[ids_with_n]
    
    mutations = pp.find( dna_values[ids_with_n] == 1)
    wildtype = pp.find( dna_values[ids_with_n]==0)
    
    dna_values2use = dna_values[ids_with_n]
    
    if len(mutations) == 0 or len(wildtype) == 0:
      print "skipping ",dna_gene
      continue
    
    print "working ",dna_gene, "  %d of %d"%(d_idx+1,n_dna)
    Z_gene = Z[ ids_with_n ]
    
    z_idx = 0
    for z_name in z_names:
      
      z_values = Z_gene[z_name].values
      
      auc, pvalue = auc_and_pvalue( dna_values2use, z_values )
      dna_auc[ d_idx, z_idx ] = auc
      dna_p_values[ d_idx, z_idx ] = pvalue
      z_idx+=1

    d_idx+=1
      
  return dna_auc, dna_p_values
    
      
    
  
def  dna_auc_using_latent_space( data, force = False ):
  Z           = data.Z

  data.data_store.open()
  try:
    T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ Z.index ]
  except:
    T=data.data_store["/CLINICAL/TISSUE"].loc[ Z.index ]
  data.data_store.close()
  
  dna_names = data.dna.sum(0).sort_values(ascending=False).index.values
  n_dna = len(dna_names)
  dna = data.dna[ dna_names ]
  
  save_dir = os.path.join( data.save_dir, "dna_auc_latent" )
  check_and_mkdir(save_dir)
  z_names = data.z_names
  barcodes = Z.index.values

  try:
    dna_z_auc     = pd.read_csv( save_dir + "/dna_z_auc.csv", index_col="gene" )
    dna_z_auc_p   = pd.read_csv( save_dir + "/dna_z_auc_p.csv", index_col="gene" )
  
  except: 
    print "could not load, forcing..."  
    force=True
    
  if force is True:
    print "computing DNA-Z AUCs"
    dna_z = auc_p_tissue_filter( dna, Z, T, p = 0.01 )
    
    dna_z_auc   = pd.DataFrame( dna_z[0], index = dna_names, columns=z_names)
    dna_z_auc_p   = pd.DataFrame( dna_z[1], index = dna_names, columns=z_names)

  
  dna_z_auc.to_csv( save_dir + "/dna_z_auc.csv", index_label="gene" )
  dna_z_auc_p.to_csv( save_dir + "/dna_z_auc_p.csv", index_label="gene" )  
  dna_z_p= dna_z_auc_p
  
  f=pp.figure( figsize=(24,12) )
  
  nbr_genes = 20
  nbr_zs    = 10
  genes = dna_names[:nbr_genes]
  k_idx = 1
  for gene in genes:
    best_z_names = dna_z_p.loc[gene].sort_values()[:nbr_zs].index.values
    dna_values = dna[gene].values
    
    #ids_with_n = ids_with_at_least_n_mutations( dna_values, T, n = 5 )
    ids_with_n = ids_with_at_least_p_mutations( dna_values, T, p = 0.01 )
    
    barcodes_with_n = barcodes[ids_with_n]
    
    mutations = pp.find( dna_values[ids_with_n] == 1)
    wildtype = pp.find( dna_values[ids_with_n]==0)
    

    z_idx = 0
    for z_name in best_z_names:
      z_values = Z[z_name].loc[barcodes_with_n].values
      z_all_wild = Z[z_name].values[pp.find( dna_values==0)] 
      
      ax = f.add_subplot(nbr_genes, nbr_zs ,k_idx)

      ax.hist( z_all_wild, 30, normed=True,histtype="step", lw=1, color="black" )
      ax.hist( z_values[wildtype], 30, normed=True,histtype="step", lw=2, color="blue" )
      ax.hist( z_values[mutations], 15, normed=True,histtype="step", lw=2, color="red" )

      ax.set_title(gene+"-"+z_name)
      # if z_idx == 0:
      #   ax.set_ylabel(gene)
      # ax.set_xlabel(z_name)
      z_idx+=1
      k_idx+=1
  pp.savefig( save_dir + "/dna_top_z.png", fmt="png", dpi=300)

  z_scores = -np.sum( np.log2(dna_z_p),1)
  
  f=pp.figure( figsize=(24,12) )
  genes = z_scores.sort_values()[-nbr_genes:].index.values #dna_names[:nbr_genes]
  #pdb.set_trace()
  k_idx = 1
  for gene in genes:
    best_z_names = dna_z_p.loc[gene].sort_values()[:nbr_zs].index.values
    dna_values = dna[gene].values
    
    #ids_with_n = ids_with_at_least_n_mutations( dna_values, T, n = 5 )
    ids_with_n = ids_with_at_least_p_mutations( dna_values, T, p = 0.01 )
    
    barcodes_with_n = barcodes[ids_with_n]
    
    mutations = pp.find( dna_values[ids_with_n] == 1)
    wildtype = pp.find( dna_values[ids_with_n]==0)
    

    z_idx = 0
    for z_name in best_z_names:
      z_values = Z[z_name].loc[barcodes_with_n].values
      z_all_wild = Z[z_name].values[pp.find( dna_values==0)] 
      
      ax = f.add_subplot(nbr_genes, nbr_zs ,k_idx)

      ax.hist( z_all_wild, 30, normed=True,histtype="step", lw=1, color="black" )
      ax.hist( z_values[wildtype], 30, normed=True,histtype="step", lw=2, color="blue" )
      ax.hist( z_values[mutations], 15, normed=True,histtype="step", lw=2, color="red" )

      ax.set_title(gene+"-"+z_name)
      # if z_idx == 0:
      #   ax.set_ylabel(gene)
      # ax.set_xlabel(z_name)
      z_idx+=1
      k_idx+=1
  pp.savefig( save_dir + "/dna_top_genes.png", fmt="png", dpi=300)
    
  global_order = np.argsort( dna_z_p.values.flatten() )
  #
  rr = np.unravel_index(global_order[:nbr_genes*nbr_zs], dims=dna_z_p.values.shape )
  dna_s = dna_names[rr[0]]
  z_s = z_names[rr[1]]
  
  f=pp.figure( figsize=(24,12) )
  #order = np.argsort(dna_s)
  #dna_s = dna_s[order]
  #z_s = z_s[order]
  k_idx=1
  for gene, z_name in zip(dna_s,z_s):
    #best_z_names = dna_z_p.loc[gene].sort_values()[:nbr_zs].index.values
    dna_values = dna[gene].values
    # mutations = pp.find( dna_values == 1)
    # wildtype = pp.find( dna_values==0)
    #
    # z_values = Z[z_name].values
    
    #ids_with_n = ids_with_at_least_n_mutations( dna_values, T, n = 5 )
    ids_with_n = ids_with_at_least_p_mutations( dna_values, T, p = 0.01 )
    
    #if gene == "APC":
    #  pdb.set_trace()
    barcodes_with_n = barcodes[ids_with_n]
    
    mutations = pp.find( dna_values[ids_with_n] == 1)
    wildtype = pp.find( dna_values[ids_with_n]==0)

    z_values = Z[z_name].loc[barcodes_with_n].values

    z_all_wild = Z[z_name].values[pp.find( dna_values==0)] 

    ax = f.add_subplot(nbr_genes, nbr_zs ,k_idx)

    #ax.hist( z_all_wild, 30, normed=True,histtype="step", lw=1, color="black" )
    ax.hist( z_values[wildtype], 30, normed=True,histtype="step", lw=2, color="blue" )
    ax.hist( z_values[mutations], 15, normed=True,histtype="step", lw=2, color="red" )
    ax.set_title(gene+"-"+z_name)
    ax.set_xlabel("")
    k_idx+=1
  pp.savefig( save_dir + "/dna_top_z2.png", fmt="png", dpi=300)
  
def repeat_kmeans( data, DATA, data_name, K = 20, repeats=10 ):
  Z           = data.Z
  X=DATA
  STD = data.Z_std
  #X = quantize(X)
  #X = normalize(X)
  
  save_dir = os.path.join( data.save_dir, "repeat_kmeans_K%d_%s"%(K,data_name) )
  check_and_mkdir(save_dir) 
  results = {}
  data.data_store.open()
  #pdb.set_trace()
  try:
    T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  except:
    #pdb.set_trace()
    T=data.data_store["/CLINICAL/TISSUE"].loc[ X.index ]
  data.data_store.close()
  tissue_pallette = sns.hls_palette(len(T.columns))
  bcs = X.index.values
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  good = pp.find( np.isnan(times) == False )
  #pdb.set_trace()

  bcs = bcs[good]
  X = X.loc[bcs]
  #STD = STD.loc[bcs]
  T = T.loc[bcs]
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  data.data_store.open()
  dna = data.data_store["/DNA/channel/0"].loc[bcs].fillna(0)
  data.data_store.close()
  
  dim = X.shape[1]
  
  
  for tissue_name in T.columns:
    print "working ", tissue_name
    ids = pp.find( T[tissue_name]==1 )
    n_ids = len(ids); n_tissue=n_ids
    if n_ids==0:
      continue
  
    affinity_matrix = np.zeros( (n_ids,n_ids), dtype=int )
    new_X = 0*X.values[ids,:]
    for r in range(repeats):
      print "repeat ",r+1
      
      shuffle_ids = np.random.permutation(n_ids)
      train_ids = shuffle_ids[: n_ids /2 ]
      test_ids = shuffle_ids[n_ids /2: ]
      train_X = X.values[ids[train_ids],:] # + STD.values[ids[train_ids],:]*np.random.randn(len(train_ids),dim)
      test_X  = X.values[ids[test_ids],:] #+ STD.values[ids[test_ids],:]*np.random.randn(len(test_ids),dim)
      mn = train_X.mean(0)
      std = train_X.std(0)
      
      zeros = pp.find(std==0)
      std[zeros]=1.0
      
      #pdb.set_trace()
      train_X -= mn; train_X /= std
      test_X -= mn; test_X /= std
      
      kmeans = MiniBatchKMeans(n_clusters=K, random_state=r ).fit( train_X )
      kmeans_labels = kmeans.predict(test_X ) #labels_
      
      for k in range(K):
        Ik = test_ids[ pp.find( kmeans_labels == k ) ]
        new_X[Ik,:] += kmeans.cluster_centers_[k]

      print "switching ids"
      train_ids = shuffle_ids[n_ids /2: ]
      test_ids = shuffle_ids[:n_ids /2]
      
      train_X = X.values[ids[train_ids],:]# + STD.values[ids[train_ids],:]*np.random.randn(len(train_ids),dim)
      test_X  = X.values[ids[test_ids],:] #+ STD.values[ids[test_ids],:]*np.random.randn(len(test_ids),dim)
      mn = train_X.mean(0)
      std = train_X.std(0)
      zeros = pp.find(std==0)
      std[zeros]=1.0
      
      train_X -= mn; train_X /= std
      test_X -= mn; test_X /= std
      
      kmeans = MiniBatchKMeans(n_clusters=K, random_state=r ).fit( train_X )
      kmeans_labels = kmeans.predict(test_X ) #labels_
      
      for k in range(K):
        Ik = test_ids[ pp.find( kmeans_labels == k ) ]
        new_X[Ik,:] += kmeans.cluster_centers_[k]
        
        # for i in Ik:
        #   for j in Ik:
        #     affinity_matrix[i,j] +=1
            
    affinity_matrix /= repeats    
    new_X /= repeats    
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=r ).fit(new_X)
    kmeans_labels = kmeans.labels_
    
    #M = SpectralClustering(n_clusters=K, affinity="precomputed", n_init=10)
    y = kmeans_labels #M.fit_predict( affinity_matrix )
    #pdb.set_trace()
    spectral_graph = make_graph_from_labels( X.values[ids,:], y, X.index.values[ids], [] )
    
    k_pallette = sns.color_palette("rainbow", K)
    patient_order = np.argsort(y)
    k_colors = np.array([k_pallette[int(i)] for i in y[patient_order]] )

    kmeans_T = MiniBatchKMeans(n_clusters=K, random_state=0 ).fit(new_X.T)
    kmeans_labels_T = kmeans_T.labels_
    
    z_order = np.argsort(kmeans_labels_T)
    
    #X_sorted = pd.DataFrame( X.values[ ids[patient_order],:], index = X.index.values[ids[patient_order]], columns=X.columns )
    X_sorted = pd.DataFrame( new_X[patient_order,:][:,z_order], index = X.index.values[ids[patient_order]], columns=X.columns[z_order] )
    
    h = sns.clustermap( X_sorted, row_colors=k_colors, row_cluster=False, col_cluster=False, figsize=(10,10) )
    pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
    pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
    h.ax_row_dendrogram.set_visible(False)
    h.ax_col_dendrogram.set_visible(False)
    h.cax.set_visible(False)
    h.ax_heatmap.hlines(n_tissue-pp.find(np.diff(y[patient_order]))-1, *h.ax_heatmap.get_xlim(), color="black", lw=5)
    pp.savefig( save_dir + "/%s_repeat_kmeans.png"%(tissue_name), fmt="png" )#, dpi=300, bbox_inches='tight')
    pp.close('all')
    
    f = pp.figure()
    ax=f.add_subplot(111)
    for k in range(K):
      Ik = pp.find( y == k )
      k_bcs = bcs[ ids[Ik] ]
      
      k_times = times[ids[Ik]]
      k_events = events[ids[Ik]]
      kmf = KaplanMeierFitter()
      if len(k_bcs) > 0:
        kmf.fit(k_times, event_observed=k_events, label="k%d"%(k)  )
        ax=kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color=k_pallette[k],ci_show=False)
    pp.title("%s using %s"%(tissue_name, data_name))
    pp.savefig( save_dir + "/%s_survival.png"%(tissue_name), format="png", dpi=300)
    pp.close('all')  

def survival_regression_global( data, DATA, data_name, L2s, K = 5, K_groups = 4, repeats = 5, fitter=AalenAdditiveFitter ):
  Z           = data.Z
  X=DATA
  z_names = X.columns.values
  dim_z = len(z_names)
  if fitter==AalenAdditiveFitter:
    save_dir = os.path.join( data.save_dir, "survival_regression_global_%s_K_%d_Aalen"%(data_name,K) )
  elif fitter == CoxPHFitter:
    save_dir = os.path.join( data.save_dir, "survival_regression_global_%s_K_%d_Cox3"%(data_name,K) )
  check_and_mkdir(save_dir) 
  
  survival_fig_dir = os.path.join(save_dir, "survival_curves" )
  heatmap_fig_dir = os.path.join(save_dir, "heatmaps" )
  survival_info_dir = os.path.join(save_dir, "survival_info" )
  check_and_mkdir(survival_fig_dir)
  check_and_mkdir(survival_info_dir)
  check_and_mkdir(heatmap_fig_dir)
  results = {}
  data.data_store.open()
  #pdb.set_trace()
  # try:
  #   T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  # except:
  #   #pdb.set_trace()
  T=data.data_store["/CLINICAL/TISSUE"].loc[ X.index ]
    
  T = merge_tissues( T, ["coad","read"])
    
  data.data_store.close()
  tissue_pallette = sns.hls_palette(len(T.columns))
  bcs = X.index.values
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  age    = data.survival.data.loc[ bcs ]["Age"].values/ 3650.0
  good = pp.find( np.isnan(times) == False )
  #pdb.set_trace()

  bcs = bcs[good]
  X = X.loc[bcs]
  if data_name=="Z":
    STD = data.Z_std.loc[bcs]
  T = T.loc[bcs]
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  age    = data.survival.data.loc[ bcs ]["Age"].values/ 3650.0
  #pdb.set_trace()
  data_columns = list(X.columns.values)
  #pdb.set_trace()
  data_columns.append("Age"); data_columns.append("T"); data_columns.append("E");  data_columns.append("tissue")
  
  dataset = pd.DataFrame( np.hstack( (X.values, age[:,np.newaxis], times[:,np.newaxis], events[:,np.newaxis], np.argmax(T.values,1)[:,np.newaxis] ) ), index = X.index, columns = data_columns )
  
  data.data_store.open()
  dna = data.data_store["/DNA/channel/0"].loc[bcs].fillna(0)
  data.data_store.close()
  
  dim = X.shape[1]
  
  random_state=0
  # L2s = [0.01,0.1,1.0,2.5,5.0,10.0] #,20.0,50.0]
  # L2s = [0.0] #,0.01,0.1] #,1.0,2.5,5.0,10.0] #,20.0,50.0]
  mean_score = []
  for penalizer in L2s:
    cph = fitter( penalizer=penalizer, strata = "tissue" )
    fold_scores = k_fold_cross_validation(cph, dataset, 'T', event_col='E', k=K)
    mean_score.append( np.mean(fold_scores))
    
    print penalizer, fold_scores, mean_score[-1]
    
  mean_score = np.array(mean_score)
  best_idx = np.argmax( mean_score )  
  best_l2 = L2s[best_idx]
    
  
  
  coefs = np.zeros( (dim_z,repeats*K) )
  predicted_death = np.zeros( (len(X),repeats) )
  weighted_death = np.zeros( (len(X),repeats) )
  rk=0

  for r in range(repeats):
    print "repeat ",r
    if data_name == "Z":
      print "adding noise"
      dataset = pd.DataFrame( np.hstack( (X.values+STD.values*np.random.randn(STD.values.shape[0],STD.values.shape[1]), age[:,np.newaxis], times[:,np.newaxis], events[:,np.newaxis], np.argmax(T.values,1)[:,np.newaxis] ) ), index = X.index, columns = data_columns )
      
    folds = StratifiedKFold(n_splits=K, shuffle = True, random_state=0)
    for train_ids, test_ids in folds.split( X.values, events ): #[:,np.newaxis].astype(int) ):

      train_bcs = bcs[train_ids]
      test_bcs  = bcs[test_ids]

      train_X = dataset.loc[ train_bcs ]
      test_X  = dataset.loc[ test_bcs ]
      
      cph = fitter(penalizer=best_l2, strata="tissue" )
      cph.fit(train_X, duration_col='T', event_col='E')
      test_survival = cph.predict_survival_function( test_X )

      wd = np.squeeze( np.dot( test_X[z_names].values, cph.hazards_[z_names].T ) )
      prd = test_survival.index.values[ np.argmin(test_survival.values,0) ]
      weighted_death[:,r][ test_ids] = wd
      predicted_death[:,r][ test_ids ] = prd
      coefs[:,rk] = cph.hazards_[z_names].values
      rk+=1
      #pdb.set_trace()
  predicted_death_mean = pd.Series( predicted_death.mean(1), index = bcs, name="PAN" )
  weighted_death_mean = pd.Series( weighted_death.mean(1), index = bcs, name="PAN" )
  coefs_mean = pd.Series( coefs.mean(1), index = z_names, name="PAN" )
  
  predicted_death_std = pd.Series( predicted_death.std(1), index = bcs, name="PAN" )
  weighted_death_std = pd.Series( weighted_death.std(1), index = bcs, name="PAN" )
  coefs_std = pd.Series( coefs.std(1), index = z_names, name="PAN" )
  
  global_concordance = lifelines.utils.concordance_index( times, -weighted_death_mean, events )
  z_order = np.argsort(coefs_mean)
  max_coef = np.max(np.abs(coefs_mean.values))
  XV = X.values #np.dot( X.values, np.diag(coefs_mean.values) )
  XV2 = np.hstack( (weighted_death_mean[:,np.newaxis],XV) )
  
  max_ = np.max(XV2,0)
  min_ = np.min(XV2,0)
  rg = max_ - min_
  
  XV2 = XV2 - max_[np.newaxis,:]
  XV2 = 2* XV2 / rg[np.newaxis,:] + 1
  
  XV2[:,1:] = np.dot( XV2[:,1:], np.diag(coefs_mean.values/max_coef) )
  
  z_ids2use = np.hstack( (z_order[:10], z_order[-10:] ))
  z_names2use = np.hstack( (z_names[z_order[:10]], z_names[z_order[-10:]] ))
  XV2cols = ["weighted"]
  XV2cols.extend(z_names2use)
  z_ids2use2 = [0]
  z_ids2use2.extend( list(z_ids2use+1))
  
  patient_order = np.argsort( weighted_death_mean ) 
  

  k_pallette = sns.color_palette("RdBu_r", K_groups+2)
  k_pallette = [k_pallette[0],k_pallette[1],k_pallette[-2],k_pallette[-1]]
  
  I_splits_global = survival_splits( events, patient_order, K_groups )
  groups_global   = groups_by_splits( len(X), I_splits_global )
  k_colors_global = np.array([k_pallette[int(i)] for i in groups_global[patient_order]] )
  f=pp.figure()
  ax = plot_survival_by_splits( times, events, I_splits_global, at_risk_counts=False,show_censors=True,ci_show=False, cmap = None, colors=k_pallette, labels=["v low","low","high","v high"])
  
  
  
  #pp.title( "%s concordance = %0.2f  p_value = %g test stat = %0.1f"%( tissue_name.upper(), tissue_concordance,tissue_p_value ,tissue_test_statistic ) )
  pp.savefig( survival_fig_dir + "/PAN.png", format="png" )

  f=pp.figure()
  X_sorted = pd.DataFrame( XV[patient_order,:][:,z_ids2use], index = X.index.values[patient_order], columns=z_names2use )
  X_sorted2 = pd.DataFrame( XV2[patient_order,:][:,z_ids2use2], index = X_sorted.index.values, columns=XV2cols )
  h = sns.clustermap( X_sorted2, \
                      row_colors=k_colors_global, \
                      row_cluster=False, \
                      col_cluster=False, \
                      figsize=(10,10),\
                      yticklabels = False )
                      
  pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  h.ax_row_dendrogram.set_visible(False)
  h.ax_col_dendrogram.set_visible(False)
  h.cax.set_visible(False)
  pp.savefig( heatmap_fig_dir + "/PAN.png", format="png" )#, dpi=300, bbox_inches='tight')
  pp.close('all')
  
  
  
  predicted_death_sort = predicted_death_mean.sort_values()
  f = pp.figure()
  pp.plot( weighted_death[patient_order,:], '.', alpha=0.5  )
  pp.title("global concordance = %0.3f"%(global_concordance))
  pp.savefig( save_dir + "/predicted_death.png", format="png" )
  f = pp.figure()
  pp.plot(coefs[z_order,:], '.'  )
  pp.title("global concordance = %0.3f"%(global_concordance))
  pp.savefig( save_dir + "/coefs.png", format="png" )
  #pdb.set_trace()
  
  f_subplots = pp.figure(figsize=(16,24))
  subplot_rows = 6; subplot_cols = 5
  avoid_tissues = {"dlbc":1}
  tissue_results = []
  tissue_survivals=[]
  t_idx=0
  for tissue_name in np.sort(T.columns): #[8:]:
    if avoid_tissues.has_key(tissue_name) is True:
      continue
    print "working ", tissue_name
    ids = pp.find( T[tissue_name]==1 )
    n_ids = len(ids); n_tissue=n_ids
    if n_ids==0:
      continue
  
    K2use = K_groups
    n_events = events[ids].sum()
    if n_events < 10:
      K2use=2
      
    tissue_patient_order = np.argsort( weighted_death_mean[ids] ) 
    
    I_splits = survival_splits( events[ids], tissue_patient_order, K_groups )
    groups   = groups_by_splits( len(ids), I_splits )
    
    wd = weighted_death_mean[ids]
    prd = predicted_death_mean[ids]
    tissue_concordance = lifelines.utils.concordance_index( times[ids], -weighted_death_mean[ids], events[ids] )
    tissue_result = multivariate_logrank_test(times[ids], groups=groups, event_observed=events[ids] )
    tissue_p_value = tissue_result.p_value
    tissue_test_statistic = tissue_result.test_statistic
    
    assert K_groups==4, "not set up for differnt groups"
    #k_pallette = sns.color_palette("RdBu_r", K_groups+2)
    #k_pallette = [k_pallette[0],k_pallette[1],k_pallette[-2],k_pallette[-1]]
    k_colors = np.array([k_pallette[int(i)] for i in groups[tissue_patient_order]] )
    k_colors2 = np.array([k_pallette[int(i)] for i in groups] )
    
    f=pp.figure()
    ax = plot_survival_by_splits( times[ids], events[ids], I_splits, at_risk_counts=False,show_censors=True,ci_show=False, cmap = None, colors=k_pallette, labels=["v low","low","high","v high"])
    pp.title( "%s concordance = %0.2f  P = %g test stat = %0.1f"%( tissue_name.upper(), tissue_concordance,tissue_p_value ,tissue_test_statistic ) )
    pp.savefig( survival_fig_dir + "/%s.png"%(tissue_name), format="png" )

    ax_subplot = f_subplots.add_subplot(subplot_rows,subplot_cols, t_idx+1)
    ax_subplot = plot_survival_by_splits( times[ids], events[ids], I_splits, at_risk_counts=False,show_censors=True,ci_show=False, cmap = None, colors=k_pallette, labels=["v low","low","high","v high"], ax=ax_subplot)
    a  = ax_subplot.axis()
    ax_subplot.text( 0.3*(a[1]-a[0]), 0.8, "%s\nP=%g"%(tissue_name.upper(),tissue_p_value), fontsize=16, bbox=dict(facecolor='green', alpha=0.5 ) )
    #ax_subplot.set_title( "%s concordance = %0.2f  P = %g"%( tissue_name.upper(), tissue_concordance,tissue_p_value  ), fontsize=16 )
    #ax_subplot.set_xt
    if tissue_name == "uvm":
      pass
    else:
      ax_subplot.legend([])
    
    X_sorted = pd.DataFrame( XV[ids[tissue_patient_order],:][:,z_ids2use], index = X.index.values[ids[tissue_patient_order]], columns=z_names2use )
    X_sorted2 = pd.DataFrame( XV2[ids[tissue_patient_order],:][:,z_ids2use2], index = X_sorted.index.values, columns=XV2cols )
    h = sns.clustermap( X_sorted2, \
                        row_colors=k_colors, \
                        row_cluster=False, \
                        col_cluster=False, \
                        figsize=(10,10),\
                        yticklabels = False )
                        
    pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
    pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
    h.ax_row_dendrogram.set_visible(False)
    h.ax_col_dendrogram.set_visible(False)
    h.cax.set_visible(False)
    #h.ax_heatmap.hlines(n_tissue-pp.find(np.diff(groups[patient_order]))-1, *h.ax_heatmap.get_xlim(), color="black", lw=5)
    pp.savefig( heatmap_fig_dir + "/%s.png"%(tissue_name), format="png" )#, dpi=300, bbox_inches='tight')
    pp.close('all')
    t_idx+=1
    

    
    tissue_results.append( pd.Series( [tissue_concordance,tissue_p_value,tissue_test_statistic], index=["concordance","p_value","test_statistic"], name=tissue_name ) )
    
    ts_data = np.vstack( (times[ids][tissue_patient_order], \
                          events[ids][tissue_patient_order],\
                          groups[tissue_patient_order],\
                          prd[tissue_patient_order], wd[tissue_patient_order]) ).T
    tissue_survival = pd.DataFrame( ts_data, index = bcs[ ids[tissue_patient_order] ], columns = ["T","E","group","predicted_death","weighted_death"] )
    tissue_survival.to_csv( survival_info_dir  + "/%s_survival_info.csv"%(tissue_name), index_label="barcode")
    
    tissue_survivals.append( pd.DataFrame( tissue_survival.values, index=tissue_survival.index, columns=tissue_survival.columns ) )
  tissue_results = pd.concat( tissue_results, axis=1).T
  tissue_results.to_csv( save_dir + "/tissue_results.csv", index_label="tissue")
  
  coef = pd.DataFrame( np.vstack((coefs_mean,coefs_std)).T, index = z_names, columns=["mean","std"] ) 
  coef.to_csv(save_dir + "/coefs.csv", index_label="feature" )
  
  global_predictions = pd.DataFrame( np.vstack( (predicted_death_mean,predicted_death_std,weighted_death_mean,weighted_death_std) ).T, index = bcs, columns=["predicted_death","predicted_death_std","weighted_death","weighted_death_std"] )
  global_predictions.to_csv( save_dir + "/global_predictions.csv", index_label="barcode")
  
  global_survivals = pd.concat( tissue_survivals, axis = 0)
  global_survivals.to_csv( save_dir + "/global_survival.csv", index_label="barcode")
  
  f_subplots.savefig( survival_fig_dir + "/survival_subplots.png", format="png", dpi=300, bbox_inches="tight" )
  #pdb.set_trace()
    #
    # I_splits = survival_splits( events[ids], patient_order, K_groups )
    # groups   = groups_by_splits( len(ids), I_splits )
    #
    # f=pp.figure()
    # ax = plot_survival_by_splits( times[ids], events[ids], I_splits, at_risk_counts=False,show_censors=True,ci_show=False, cmap = "rainbow")
    # pp.title( "%s"%( tissue_name ) )
    # pp.savefig( save_dir + "/%s.png"%(tissue_name), format="png" )

def survival_regression_global_g0_v_g3( data, DATA, data_name, L2s, K = 5, K_groups = 4, repeats = 5, fitter=AalenAdditiveFitter ):
  Z           = data.Z
  X=DATA
  z_names = X.columns.values
  dim_z = len(z_names)
  if fitter==AalenAdditiveFitter:
    save_dir = os.path.join( data.save_dir, "survival_regression_global_%s_K_%d_Aalen"%(data_name,K) )
  elif fitter == CoxPHFitter:
    save_dir = os.path.join( data.save_dir, "survival_regression_global_%s_K_%d_Cox3"%(data_name,K) )
  check_and_mkdir(save_dir) 
  
  survival_fig_dir = os.path.join(save_dir, "survival_curves" )
  heatmap_fig_dir = os.path.join(save_dir, "heatmaps" )
  survival_info_dir = os.path.join(save_dir, "survival_info" )
  check_and_mkdir(survival_fig_dir)
  check_and_mkdir(survival_info_dir)
  check_and_mkdir(heatmap_fig_dir)
  results = {}
  data.data_store.open()
  #pdb.set_trace()
  # try:
  #   T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  # except:
  #   #pdb.set_trace()
  #   T=data.data_store["/CLINICAL/TISSUE"].loc[ X.index ]
  T=data.data_store["/CLINICAL/TISSUE"].loc[ X.index ]
  T = merge_tissues( T, ["coad","read"])
  
  data.data_store.close()
  tissue_pallette = sns.hls_palette(len(T.columns))
  bcs = X.index.values
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  good = pp.find( np.isnan(times) == False )
  #pdb.set_trace()

  bcs = bcs[good]
  X = X.loc[bcs]
  #STD = STD.loc[bcs]
  T = T.loc[bcs]
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  data_columns = list(X.columns.values)
  #pdb.set_trace()
  data_columns.append("T"); data_columns.append("E"); data_columns.append("tissue")
  
  dataset = pd.DataFrame( np.hstack( (X.values, times[:,np.newaxis], events[:,np.newaxis], np.argmax(T.values,1)[:,np.newaxis] ) ), index = X.index, columns = data_columns )
  
  data.data_store.open()
  dna = data.data_store["/DNA/channel/0"].loc[bcs].fillna(0)
  data.data_store.close()
  
  dim = X.shape[1]

  #pdb.set_trace()
  f_subplots = pp.figure(figsize=(16,24))
  subplot_rows = 6; subplot_cols = 5
  avoid_tissues = {"dlbc":1}
  tissue_results = []
  tissue_survivals=[]
  t_idx=0
  row_idx = 1
  col_idx = 1
  for tissue_name in np.sort(T.columns):
    if tissue_name == "dlbc":
      continue
    print "working ", tissue_name
    ids = pp.find( T[tissue_name]==1 )
    n_ids = len(ids); n_tissue=n_ids
    if n_ids==0:
      continue
  
    K2use = K_groups
    n_events = events[ids].sum()
    if n_events < 10:
      K2use=2
      
    tissue_survival = pd.read_csv( survival_info_dir  + "/%s_survival_info.csv"%(tissue_name), index_col="barcode")
    
    groups = tissue_survival["group"]
    
    
    I_splits = [ [pp.find(groups.values==0)], [pp.find(groups.values==3)] ] 
    
    I = np.squeeze( np.hstack( (I_splits[0],I_splits[1])) ) 
    groups = groups.values[I]
    #survival_splits( events[ids], tissue_patient_order, K_groups )
    #groups   = groups_by_splits( len(ids), I_splits )
    tissue_barcodes = tissue_survival.index.values
    
    t_events = tissue_survival["E"].values
    t_times  = tissue_survival["T"].values
    t_weighted_death = tissue_survival["weighted_death"].values
    
    tissue_concordance = lifelines.utils.concordance_index( t_times[I], -t_weighted_death[I], t_events[I] )
    
    tissue_result = logrank_test(t_times[I_splits[0]], t_times[I_splits[1]], t_events[I_splits[0]],t_events[I_splits[1]] )
    
    tissue_p_value = tissue_result.p_value
    tissue_test_statistic = tissue_result.test_statistic
    
    
    #assert K_groups==4, "not set up for differnt groups"
    k_pallette = sns.color_palette("RdBu_r", 6)
    k_pallette = [k_pallette[0],k_pallette[-1]]
    
    f=pp.figure()
    ax = plot_survival_by_splits( t_times, t_events, I_splits, at_risk_counts=False,show_censors=True,ci_show=False, cmap = None, colors=k_pallette, labels=["v low","v high"])
    pp.title( "%s concordance = %0.2f  p_value = %g test stat = %0.1f"%( tissue_name.upper(), tissue_concordance,tissue_p_value ,tissue_test_statistic ) )
    pp.savefig( survival_fig_dir + "/%s_p0_v_p3.png"%(tissue_name), format="png" )


    ax_subplot = f_subplots.add_subplot(subplot_rows,subplot_cols, t_idx+1)
    ax_subplot = plot_survival_by_splits( t_times, t_events, I_splits, at_risk_counts=False,show_censors=True,ci_show=False, cmap = None, colors=k_pallette, labels=["v low","v high"], ax=ax_subplot)
    a  = ax_subplot.axis()
    ax_subplot.text( 0.3*(a[1]-a[0]), 0.8, "%s\nP=%g"%(tissue_name.upper(),tissue_p_value), fontsize=16, bbox=dict(facecolor='green', alpha=0.35 ) )
    if col_idx > 1:
      ax_subplot.set_ylabel("")
    if row_idx < subplot_rows:
      ax_subplot.set_xlabel("")
      
    #ax_subplot.set_title( "%s concordance = %0.2f  P = %g"%( tissue_name.upper(), tissue_concordance,tissue_p_value  ), fontsize=16 )
    #ax_subplot.set_xt
    if tissue_name == "uvm":
      pass
    else:
      ax_subplot.legend([])
    t_idx+=1
      
    col_idx += 1
    if col_idx > subplot_cols:
      col_idx = 1
      row_idx +=1
      
    tissue_results.append( pd.Series( [tissue_concordance,tissue_p_value,tissue_test_statistic], index=["concordance","p_value","test_statistic"], name=tissue_name ) )
  tissue_results = pd.concat( tissue_results, axis=1).T
  tissue_results.to_csv( save_dir + "/tissue_results_g0_v_g3.csv", index_label="tissue")
  f_subplots.savefig( survival_fig_dir + "/survival_subplots_g0_v_g3.png", format="png", dpi=300, bbox_inches="tight" )

      
def survival_regression_local( data, DATA, data_name, K = 5, K_groups = 4, fitter=AalenAdditiveFitter ):
  Z           = data.Z
  X=DATA
  z_names = X.columns.values
  if fitter==AalenAdditiveFitter:
    save_dir = os.path.join( data.save_dir, "survival_regression_local_%s_K_%d_Aalen"%(data_name,K) )
  elif fitter == CoxPHFitter:
    save_dir = os.path.join( data.save_dir, "survival_regression_local_%s_K_%d_Cox2"%(data_name,K) )
  check_and_mkdir(save_dir) 
  results = {}
  data.data_store.open()
  #pdb.set_trace()
  try:
    T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  except:
    #pdb.set_trace()
    T=data.data_store["/CLINICAL/TISSUE"].loc[ X.index ]
  data.data_store.close()
  tissue_pallette = sns.hls_palette(len(T.columns))
  bcs = X.index.values
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  good = pp.find( np.isnan(times) == False )
  #pdb.set_trace()

  bcs = bcs[good]
  X = X.loc[bcs]
  #STD = STD.loc[bcs]
  T = T.loc[bcs]
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  data_columns = list(X.columns.values)
  #pdb.set_trace()
  data_columns.append("T"); data_columns.append("E")
  
  dataset = pd.DataFrame( np.hstack( (X.values, times[:,np.newaxis], events[:,np.newaxis]) ), index = X.index, columns = data_columns )
  
  data.data_store.open()
  dna = data.data_store["/DNA/channel/0"].loc[bcs].fillna(0)
  data.data_store.close()
  
  dim = X.shape[1]
  
  random_state=0
  for tissue_name in T.columns: #[8:]:
    if tissue_name == "dlbc":
      continue
    print "working ", tissue_name
    ids = pp.find( T[tissue_name]==1 )
    n_ids = len(ids); n_tissue=n_ids
    if n_ids==0:
      continue
  
    K2use = K
    n_events = events[ids].sum()
    if n_events < 10:
      K2use=2
    
    train_bcs = bcs[ids]
    test_bcs  = bcs[ids]

    # model selection for L2
    L2s = [1.0,2.5,5.0,10.0,20.0,50.0]
    mean_score = []
    for penalizer in L2s:
      if fitter==AalenAdditiveFitter:
        cph = fitter(smoothing_penalizer=penalizer )
      else:
        cph = fitter(penalizer=penalizer )
      fold_scores = k_fold_cross_validation(cph, dataset.loc[ bcs[ids] ], 'T', event_col='E', k=K2use)
      mean_score.append( np.mean(fold_scores))
      print penalizer, fold_scores, mean_score[-1]
    
    mean_score = np.array(mean_score)
    best_idx = np.argmax( mean_score )  
    best_l2 = L2s[best_idx]
    
    repeats = 5
    predicted_death = np.zeros( (len(ids),repeats) )
    weighted_death = np.zeros( (len(ids),repeats) )
    for r in range(repeats):
      print "repeat ",r
      folds = StratifiedKFold(n_splits=K2use, shuffle = True, random_state=r)
      for train_ids, test_ids in folds.split( X.values[ids,:], events[ids] ): #[:,np.newaxis].astype(int) ):
    
        train_bcs = bcs[ids[train_ids]]
        test_bcs  = bcs[ids[test_ids]]
      
        train_X = dataset.loc[ train_bcs ]
        test_X  = dataset.loc[test_bcs] 
        if fitter==AalenAdditiveFitter:
          cph = fitter(smoothing_penalizer=best_l2 )
        else:
          cph = fitter(penalizer=best_l2 )
        #cph = fitter(penalizer=best_l2 )
        cph.fit(train_X, duration_col='T', event_col='E')
        test_survival = cph.predict_survival_function( test_X )
        
        wd = np.squeeze( np.dot( test_X[z_names].values, cph.hazards_[z_names].T ) )
        prd = test_survival.index.values[ np.argmin(test_survival.values,0) ]
        weighted_death[:,r][ test_ids] = wd
        predicted_death[:,r][ test_ids ] = prd
        
        #pdb.set_trace()   
    predicted_death = pd.Series( predicted_death.mean(1), index = bcs[ids], name=tissue_name )
    weighted_death = pd.Series( weighted_death.mean(1), index = bcs[ids], name=tissue_name )
    
    patient_order = np.argsort( weighted_death ) #-
    #patient_order = np.argsort( -predicted_death.values )
    predicted_death_sort = predicted_death.sort_values()
    #pdb.set_trace()
    
    I_splits = survival_splits( events[ids], patient_order, K_groups )
    groups   = groups_by_splits( len(ids), I_splits )

    f=pp.figure()
    ax = plot_survival_by_splits( times[ids], events[ids], I_splits, at_risk_counts=False,show_censors=True,ci_show=False, cmap = "rainbow")
    pp.title( "%s"%( tissue_name ) )
    pp.savefig( save_dir + "/%s.png"%(tissue_name), format="png" )
    #
    # pdb.set_trace()
    # k_pallette = sns.color_palette("rainbow", K)
    # #patient_order = np.argsort(y)
    # k_colors = np.array([k_pallette[int(i)] for i in y[patient_order]] )
    #
    # kmeans_T = MiniBatchKMeans(n_clusters=K, random_state=0 ).fit(new_X.T)
    # kmeans_labels_T = kmeans_T.labels_
    #
    # z_order = np.argsort(kmeans_labels_T)
    #
    # #X_sorted = pd.DataFrame( X.values[ ids[patient_order],:], index = X.index.values[ids[patient_order]], columns=X.columns )
    # X_sorted = pd.DataFrame( new_X[patient_order,:][:,z_order], index = X.index.values[ids[patient_order]], columns=X.columns[z_order] )
    #
    # h = sns.clustermap( X_sorted, row_colors=k_colors, row_cluster=False, col_cluster=False, figsize=(10,10) )
    # pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    # pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    # pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
    # pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
    # h.ax_row_dendrogram.set_visible(False)
    # h.ax_col_dendrogram.set_visible(False)
    # h.cax.set_visible(False)
    # h.ax_heatmap.hlines(n_tissue-pp.find(np.diff(y[patient_order]))-1, *h.ax_heatmap.get_xlim(), color="black", lw=5)
    # pp.savefig( save_dir + "/%s_repeat_kmeans.png"%(tissue_name), fmt="png" )#, dpi=300, bbox_inches='tight')
    # pp.close('all')
    #
    # f = pp.figure()
    # ax=f.add_subplot(111)
    # for k in range(K):
    #   Ik = pp.find( y == k )
    #   k_bcs = bcs[ ids[Ik] ]
    #
    #   k_times = times[ids[Ik]]
    #   k_events = events[ids[Ik]]
    #   kmf = KaplanMeierFitter()
    #   if len(k_bcs) > 0:
    #     kmf.fit(k_times, event_observed=k_events, label="k%d"%(k)  )
    #     ax=kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color=k_pallette[k],ci_show=False)
    # pp.title("%s using %s"%(tissue_name, data_name))
    # pp.savefig( save_dir + "/%s_survival.png"%(tissue_name), format="png", dpi=300)
    # pp.close('all')
          
def repeat_gmm( data, K = 20, repeats=10 ):
  Z           = data.Z
  X=Z
  STD = data.Z_std
  
  
  
  
  save_dir = os.path.join( data.save_dir, "repeat_gmm_K%d_std"%(K) )
  check_and_mkdir(save_dir) 
  results = {}
  data.data_store.open()
  #pdb.set_trace()
  try:
    T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  except:
    #pdb.set_trace()
    T=data.data_store["/CLINICAL/TISSUE"].loc[ X.index ]
  data.data_store.close()
  #X = quantize(X)
  X = normalize_by_tissue(X,T)

  tissue_pallette = sns.hls_palette(len(T.columns))
  bcs = X.index.values
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  good = pp.find( np.isnan(times) == False )
  #pdb.set_trace()

  bcs = bcs[good]
  X = X.loc[bcs]
  STD = STD.loc[bcs]
  T = T.loc[bcs]
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  data.data_store.open()
  dna = data.data_store["/DNA/channel/0"].loc[bcs].fillna(0)
  data.data_store.close()
  
  dim = X.shape[1]
  
  f_all = pp.figure( figsize=(16,12) )
  
  t_idx = 0
  for tissue_name in T.columns:
    print "working ", tissue_name
    ids = pp.find( T[tissue_name]==1 )
    n_ids = len(ids); n_tissue=n_ids
    if n_ids==0:
      continue
  
    affinity_matrix = np.zeros( (n_ids,n_ids), dtype=int )
    new_X = 0*X.values[ids,:]
    for r in range(repeats):
      print "repeat ",r+1
      
      shuffle_ids = np.random.permutation(n_ids)
      train_ids = shuffle_ids[: n_ids /2 ]
      test_ids = shuffle_ids[n_ids /2: ]
      train_X = X.values[ids[train_ids],:] + STD.values[ids[train_ids],:]*np.random.randn(len(train_ids),dim)
      test_X  = X.values[ids[test_ids],:] + STD.values[ids[test_ids],:]*np.random.randn(len(test_ids),dim)
      mn = train_X.mean(0)
      std = train_X.std(0)
      
      train_X -= mn; train_X /= std
      test_X -= mn; test_X /= std
      
      kmeans = GaussianMixture(n_components=K, covariance_type ='diag', random_state=r ).fit( train_X )
      kmeans_labels = kmeans.predict(test_X ) #labels_
      
      for k in range(K):
        Ik = test_ids[ pp.find( kmeans_labels == k ) ]
        new_X[Ik,:] += kmeans.means_[k]

      print "switching ids"
      train_ids = shuffle_ids[n_ids /2: ]
      test_ids = shuffle_ids[:n_ids /2]
      
      train_X = X.values[ids[train_ids],:] + STD.values[ids[train_ids],:]*np.random.randn(len(train_ids),dim)
      test_X  = X.values[ids[test_ids],:] + STD.values[ids[test_ids],:]*np.random.randn(len(test_ids),dim)
      mn = train_X.mean(0)
      std = train_X.std(0)
      
      train_X -= mn; train_X /= std
      test_X -= mn; test_X /= std
      
      kmeans = GaussianMixture(n_components=K, covariance_type ='diag', random_state=r ).fit( train_X )
      kmeans_labels = kmeans.predict(test_X ) #labels_
      
      for k in range(K):
        Ik = test_ids[ pp.find( kmeans_labels == k ) ]
        new_X[Ik,:] += kmeans.means_[k]
        
        # for i in Ik:
        #   for j in Ik:
        #     affinity_matrix[i,j] +=1
            
    affinity_matrix /= repeats    
    new_X /= repeats    
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=r ).fit(new_X)
    kmeans_labels = kmeans.labels_
    
    #M = SpectralClustering(n_clusters=K, affinity="precomputed", n_init=10)
    y = kmeans_labels #M.fit_predict( affinity_matrix )
    #pdb.set_trace()
    spectral_graph = make_graph_from_labels( X.values[ids,:], y, X.index.values[ids], [] )
    
    k_pallette = sns.color_palette("rainbow", K)
    patient_order = np.argsort(y)
    k_colors = np.array([k_pallette[int(i)] for i in y[patient_order]] )

    kmeans_T = MiniBatchKMeans(n_clusters=K, random_state=0 ).fit(new_X.T)
    kmeans_labels_T = kmeans_T.labels_
    
    z_order = np.argsort(kmeans_labels_T)
    
    #X_sorted = pd.DataFrame( X.values[ ids[patient_order],:], index = X.index.values[ids[patient_order]], columns=X.columns )
    X_sorted = pd.DataFrame( new_X[patient_order,:][:,z_order], index = X.index.values[ids[patient_order]], columns=X.columns[z_order] )
    
    h = sns.clustermap( X_sorted, row_colors=k_colors, row_cluster=False, col_cluster=False, figsize=(10,10) )
    pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
    pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
    h.ax_row_dendrogram.set_visible(False)
    h.ax_col_dendrogram.set_visible(False)
    h.cax.set_visible(False)
    h.ax_heatmap.hlines(n_tissue-pp.find(np.diff(y[patient_order]))-1, *h.ax_heatmap.get_xlim(), color="black", lw=5)
    pp.savefig( save_dir + "/%s_repeat_kmeans.png"%(tissue_name), fmt="png" )#, dpi=300, bbox_inches='tight')
    #pp.close('all')
    
    f = pp.figure()
    ax=f.add_subplot(111)
    ax_all = f_all.add_subplot(4,8,t_idx+1)
    for k in range(K):
      Ik = pp.find( y == k )
      k_bcs = bcs[ ids[Ik] ]
      
      k_times = times[ids[Ik]]
      k_events = events[ids[Ik]]
      kmf = KaplanMeierFitter()
      if len(k_bcs) > 0:
        kmf.fit(k_times, event_observed=k_events, label="k%d"%(k)  )
        ax=kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color=k_pallette[k],ci_show=False)
        kmf.plot(ax=ax_all,at_risk_counts=False,show_censors=True, color=k_pallette[k],ci_show=False)
    pp.title("%s"%(tissue_name))
    pp.savefig( save_dir + "/%s_survival.png"%(tissue_name), format="png" ) #, dpi=300)
    pp.close() 
    t_idx+=1
  f_all.savefig( save_dir + "/AA_survival.png", format="png" , dpi=300)

def repeat_kmeans_global( data, DATA, data_name, K = 5, K2=10, repeats=10 ):
  Z           = data.Z
  X=DATA
  #STD = data.Z_std
  #X = quantize(X)
  #X = normalize(X)
  
  save_dir = os.path.join( data.save_dir, "repeat_kmeans_K%d_%s_global"%(K,data_name) )
  check_and_mkdir(save_dir) 
  results = {}
  data.data_store.open()
  #pdb.set_trace()
  try:
    T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ X.index ]
  except:
    #pdb.set_trace()
    T=data.data_store["/CLINICAL/TISSUE"].loc[ X.index ]
  data.data_store.close()
  tissue_pallette = sns.hls_palette(len(T.columns))
  bcs = X.index.values
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  good = pp.find( np.isnan(times) == False )
  #pdb.set_trace()

  bcs = bcs[good]
  X = X.loc[bcs]
  #STD = STD.loc[bcs]
  T = T.loc[bcs]
  times = data.survival.data.loc[ bcs ]["T"].values
  events = data.survival.data.loc[ bcs ]["E"].values
  
  event_ranks = 0*events.astype(float)
  for tissue_name in T.columns:
    print "working ", tissue_name
    ids = pp.find( T[tissue_name]==1 )
    n_ids = len(ids); n_tissue=n_ids
    if n_ids==0:
      continue
    t_times = times[ids]
    t_events = events[ids]
    
    event_times = t_times[ t_events==1 ]
    
    #event_rank = 2*np.argsort( event_times ).astype(float) / float(len(event_times)) - 1.0
    event_rank = event_times.astype(float) / float(np.max(event_times)) 
    
    #event_ranks[ ids[t_events==1] ] = event_rank
    event_ranks[ ids ] = event_rank
    
    #pdb.set_trace()
    
  
  
  data.data_store.open()
  dna = data.data_store["/DNA/channel/0"].loc[bcs].fillna(0)
  data.data_store.close()
  
  dim = X.shape[1]
  
  # first run kmeans r times
  #affinity_matrix = np.zeros( (n_ids,n_ids), dtype=int )
  new_X = 0*X.values
  N,D = X.shape

  for r in range(repeats):
    print "repeat ",r+1
    
    shuffle_ids = np.random.permutation(N)
    
    train_ids = shuffle_ids[:N /2 ]
    test_ids  = shuffle_ids[N/2:]
    train_X = X.values[train_ids,:] #+ STD.values[train_ids,:]*np.random.randn(len(train_ids),D)
    test_X  = X.values[test_ids,:] #+ STD.values[test_ids,:]*np.random.randn(len(test_ids),D)
    mn = train_X.mean(0)
    std = train_X.std(0)
    
    train_X -= mn; train_X /= std
    test_X -= mn; test_X /= std
    
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=r ).fit( train_X )
    kmeans_labels = kmeans.predict(test_X ) #labels_
    for k in range(K):
      Ik = test_ids[ pp.find( kmeans_labels == k ) ]
      new_X[Ik,:] += kmeans.cluster_centers_[k]


    train_ids = shuffle_ids[N /2: ]
    test_ids  = shuffle_ids[:N/2]
    train_X = X.values[train_ids,:]# + STD.values[train_ids,:]*np.random.randn(len(train_ids),D)
    test_X  = X.values[test_ids,:] #+ STD.values[test_ids,:]*np.random.randn(len(test_ids),D)
    mn = train_X.mean(0)
    std = train_X.std(0)
    
    train_X -= mn; train_X /= std
    test_X -= mn; test_X /= std
    
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=r ).fit( train_X )
    kmeans_labels = kmeans.predict(test_X ) #labels_
    for k in range(K):
      Ik = test_ids[ pp.find( kmeans_labels == k ) ]
      new_X[Ik,:] += kmeans.cluster_centers_[k]
      
  
  #affinity_matrix /= repeats    
  new_X /= repeats    
  
  kmeans = MiniBatchKMeans(n_clusters=K2, random_state=r ).fit(new_X)
  kmeans_labels = kmeans.labels_
  
  mean_event_ranks = np.zeros( K2, dtype=float )
  std_event_ranks = np.zeros( K2, dtype=float )
  for k in range(K2):
    k_ids = kmeans_labels==k
    
    k_times = times[k_ids]
    k_events = events[k_ids]
    k_event_ranks = event_ranks[k_ids]
    
    mean_event_ranks[k] = np.mean(k_events[k_events==1])
    std_event_ranks[k] = np.std(k_events[k_events==1])
    
  #pdb.set_trace()
  
  y = kmeans_labels #M.fit_predict( affinity_matrix )
  k_pallette = sns.color_palette("rainbow", K2)
  k_order = np.argsort(mean_event_ranks/std_event_ranks)
  
  #k_pallette = [k_pallette[i] for i in np.argsort(mean_event_ranks/std_event_ranks)]
  #pdb.set_trace()
  kmeans_T = MiniBatchKMeans(n_clusters=K2, random_state=0 ).fit(new_X.T)
  kmeans_labels_T = kmeans_T.labels_
  
  z_order = np.argsort(kmeans_labels_T)
  
  #X_sorted = pd.DataFrame( new_X[patient_order,:][:,z_order], index = X.index.values[patient_order], columns=X.columns[z_order] )
  
  f_all = pp.figure( figsize=(24,12) )
  t_idx=0
  for tissue_name in T.columns:
    print "working ", tissue_name
    ids = pp.find( T[tissue_name]==1 )
    n_ids = len(ids); n_tissue=n_ids
    if n_ids==0:
      continue
  
    y_tissue = y[ ids ]
    patient_order = np.argsort([k_order[i] for i in y_tissue])
    k_colors = np.array([k_pallette[k_order[int(i)]] for i in y_tissue[patient_order]] )
    
    X_sorted = pd.DataFrame( new_X[ids[patient_order],:][:,z_order], index = X.index.values[ids[patient_order]], columns=X.columns[z_order] )
    
    h = sns.clustermap( X_sorted, row_colors=k_colors, row_cluster=False, col_cluster=False, figsize=(10,10) )
    pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
    pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
    h.ax_row_dendrogram.set_visible(False)
    h.ax_col_dendrogram.set_visible(False)
    h.cax.set_visible(False)
    h.ax_heatmap.hlines(n_tissue-pp.find(np.diff(y_tissue[patient_order]))-1, *h.ax_heatmap.get_xlim(), color="black", lw=5)
    pp.savefig( save_dir + "/%s_repeat_kmeans.png"%(tissue_name), fmt="png" )#, dpi=300, bbox_inches='tight')
    pp.close('all')
    
    f = pp.figure()
    ax=f.add_subplot(111)
    ax_all = f_all.add_subplot(4,8,t_idx+1)
    for k in range(K2):
      Ik = pp.find( y_tissue == k )
      k_bcs = bcs[ ids[Ik] ]
      
      k_times = times[ids[Ik]]
      k_events = events[ids[Ik]]
      kmf = KaplanMeierFitter()
      if len(k_bcs) > 0:
        kmf.fit(k_times, event_observed=k_events, label="k%d"%(k)  )
        pdb.set_trace()
        ax=kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color=k_pallette[k],ci_show=False)
        kmf.plot(ax=ax_all,at_risk_counts=False,show_censors=True, color=k_pallette[k],ci_show=False)
    ax_all.set_title("%s"%(tissue_name))
    ax_all.set_ylim(0,1)
    ax.set_ylim(0,1)
    pp.title("%s using %s"%(tissue_name, data_name))
    pp.savefig( save_dir + "/%s_survival.png"%(tissue_name), format="png") #, dpi=300)
    pp.close() 
    t_idx+=1
  f_all.savefig( save_dir + "/AA_survival.png", format="png", dpi=300)

def describe_latent(data):
  
  Z = data.Z
  W_H = pd.concat( data.W_input2h)  
  
  spearman_dir = os.path.join( data.save_dir, "spearmans_latent" )
  pearson_dir = os.path.join( data.save_dir, "pearsons_latent" )
  save_dir = os.path.join( data.save_dir, "latent_description" )
  check_and_mkdir(save_dir)
  
  
  #pdb.set_trace()
  rna_names   = data.rna_names    
  mirna_names = data.mirna_names 
  meth_names  = data.meth_names   
  
  z_names = data.z_names
  n_z     = len(z_names)
  n_rna   = len(rna_names)
  n_mirna = len(mirna_names)
  n_meth  = len(meth_names)

  n_dna = 200
  dna_names = data.dna.sum(0).sort_values(ascending=False)[:n_dna].index.values
  dna = data.dna[ dna_names ]
  
  try:
    rna_z_rho = pd.read_csv( spearman_dir + "/rna_z_rho.csv", index_col="gene" )
  
  except: 
    print "could not load, forcing..."  
    spearmanr_latent_space_by_inputs( data, force = True ) 

  rna_z_rho = pd.read_csv( spearman_dir + "/rna_z_rho.csv", index_col="gene" )
  rna_z_p   = pd.read_csv( spearman_dir + "/rna_z_p.csv", index_col="gene" )

  mirna_z_rho = pd.read_csv( spearman_dir + "/mirna_z_rho.csv", index_col="gene" )
  mirna_z_p   = pd.read_csv( spearman_dir + "/mirna_z_p.csv", index_col="gene" )
 
  meth_z_rho = pd.read_csv( spearman_dir + "/meth_z_rho.csv", index_col="gene" )
  meth_z_p   = pd.read_csv( spearman_dir + "/meth_z_p.csv", index_col="gene" )

  dna_z_rho = pd.read_csv( spearman_dir + "/dna_z_rho.csv", index_col="gene" )
  dna_z_p   = pd.read_csv( spearman_dir + "/dna_z_p.csv", index_col="gene" )

  p_rna_z_rho = pd.read_csv( pearson_dir + "/rna_z_rho.csv", index_col="gene" )
  p_rna_z_p   = pd.read_csv( pearson_dir + "/rna_z_p.csv", index_col="gene" )

  p_mirna_z_rho = pd.read_csv( pearson_dir + "/mirna_z_rho.csv", index_col="gene" )
  p_mirna_z_p   = pd.read_csv( pearson_dir + "/mirna_z_p.csv", index_col="gene" )
 
  p_meth_z_rho = pd.read_csv( pearson_dir + "/meth_z_rho.csv", index_col="gene" )
  p_meth_z_p   = pd.read_csv( pearson_dir + "/meth_z_p.csv", index_col="gene" )

  p_dna_z_rho = pd.read_csv( pearson_dir + "/dna_z_rho.csv", index_col="gene" )
  p_dna_z_p   = pd.read_csv( pearson_dir + "/dna_z_p.csv", index_col="gene" )

  
  try:
    z_z_rho = pd.read_csv( save_dir + "/z_z_rho.csv", index_col="z" )
    z_z_p   = pd.read_csv( save_dir + "/z_z_p.csv", index_col="z" )
    p_z_z_rho = pd.read_csv( save_dir + "/p_z_z_rho.csv", index_col="z" )
    p_z_z_p   = pd.read_csv( save_dir + "/p_z_z_p.csv", index_col="z" )
  except:  
    rho_z_z = stats.spearmanr( Z.values, Z.values )
    p_rho_z_z = pearsonr( Z.values, Z.values )
    z_z_rho = pd.DataFrame( rho_z_z[0][:n_z,:][:,n_z:], index = z_names, columns=z_names)
    z_z_p   = pd.DataFrame( rho_z_z[1][:n_z,:][:,n_z:], index = z_names, columns=z_names)
    z_z_rho.to_csv( save_dir + "/z_z_rho.csv", index_label="z" )
    z_z_p.to_csv( save_dir + "/z_z_p.csv", index_label="z" )
    
    p_z_z_rho = pd.DataFrame( p_rho_z_z[0], index = z_names, columns=z_names)
    p_z_z_p   = pd.DataFrame( p_rho_z_z[1], index = z_names, columns=z_names)
    p_z_z_rho.to_csv( save_dir + "/p_z_z_rho.csv", index_label="z" )
    p_z_z_p.to_csv( save_dir + "/p_z_z_p.csv", index_label="z" )
    
  
  # ordered_rna = np.argsort( rna_z_p, axis = 1 )
  # ordered_mirna = np.argsort( mirna_z_p, axis = 1 )
  # ordered_meth = np.argsort( meth_z_p, axis = 1 )
  # ordered_dna = np.argsort( dna_z_p, axis = 1 )
  
  wW = data.weighted_W_h2z
  
  W = data.W_h2z
  order_H = np.argsort( -np.abs(W), axis=0 )
  print "getting z infos"
  results = []
  nbr_genes=40
  nbr_dna = 10
  nbr_hidden = 9
  for z_idx, z_name in zip( xrange(n_z), z_names ):
    #ordered_rna[z_name]
    rna_p = list(rna_z_p[z_name].sort_values()[:nbr_genes].index.values)
    mirna_p = list(mirna_z_p[z_name].sort_values()[:nbr_genes].index.values)
    meth_p = list(meth_z_p[z_name].sort_values()[:nbr_genes].index.values)
    dna_p = list(dna_z_p[z_name].sort_values()[:nbr_dna].index.values)
    
    p_rna_p = list(p_rna_z_p[z_name].sort_values()[:nbr_genes].index.values)
    p_mirna_p = list(p_mirna_z_p[z_name].sort_values()[:nbr_genes].index.values)
    p_meth_p = list(p_meth_z_p[z_name].sort_values()[:nbr_genes].index.values)
    p_dna_p = list(p_dna_z_p[z_name].sort_values()[:nbr_dna].index.values)
    
    h_values = list(data.h_names[ np.argsort( -np.abs(W[:,z_idx] ))[:nbr_hidden] ])
    
    rna_w = list((-np.abs(data.weighted_W_h2z["RNA"]["z_%d"%(z_idx)] )).sort_values()[:nbr_genes].index.values)
    mirna_w = list((-np.abs(data.weighted_W_h2z["miRNA"]["z_%d"%(z_idx)] )).sort_values()[:nbr_genes].index.values)
    meth_w = list((-np.abs(data.weighted_W_h2z["METH"]["z_%d"%(z_idx)] )).sort_values()[:nbr_genes].index.values)
    meth_w = list( [s.split("_")[1] for s in meth_w] )
    #pdb.set_trace()
    rna_overlap = list(np.intersect1d( rna_p, rna_w ).astype(str))
    mirna_overlap = list(np.intersect1d( mirna_p, mirna_w ).astype(str))
    meth_overlap = list(np.intersect1d( meth_p, meth_w ).astype(str))
    meth_overlap = [str(s) for s in meth_overlap]
    mirna_overlap = [str(s) for s in mirna_overlap]
    rna_overlap = [str(s) for s in rna_overlap]
    
    p_rna_overlap = list(np.intersect1d( p_rna_p, rna_w ).astype(str))
    p_mirna_overlap = list(np.intersect1d( p_mirna_p, mirna_w ).astype(str))
    p_meth_overlap = list(np.intersect1d( p_meth_p, meth_w ).astype(str))
    p_meth_overlap = [str(s) for s in p_meth_overlap]
    p_mirna_overlap = [str(s) for s in p_mirna_overlap]
    p_rna_overlap = [str(s) for s in p_rna_overlap]
    
    results.append( [z_name, {"rna_spear":rna_p, "rna_pear":p_rna_p, \
                              "rna_w":rna_w, "rna_over_spear":rna_overlap, "rna_over_pear":p_rna_overlap,\
                              "meth_spear":meth_p, "meth_pear":p_meth_p, \
                              "meth_w":meth_w, "meth_over_spear":meth_overlap, "meth_over_pear":p_meth_overlap,\
                              #"mirna_spear":mirna_p, "mirna_pear":p_mirna_p, \
                              #"mirna_w":mirna_w, "mirna_over_spear":mirna_overlap, "mirna_over_pear":p_mirna_overlap,\
                              "dna_spear":dna_p, "dna_pear":p_dna_p, \
                              "h":h_values,}])

    h_idx = 1
    for h_name in h_values:
      #pdb.set_trace()
      h_w = list( (-np.abs( W_H["h_%d"%(int(h_name.split("h")[1]))] )).sort_values()[:10].index.values)
      results[-1][1]["h%d"%h_idx] = [h[1] for h in h_w]
      h_idx+=1
  fptr = open( save_dir + "/z_description.yaml","w+" )
  fptr.write( yaml.dump(results))
  fptr.close()
    
    #pdb.set_trace()
    #results
  
  #pdb.set_trace()

def deeper_meaning_dna_and_z( data, min_p_value=1e-3, threshold = 0.01, ridges = [0.00001, 0.001,0.1,10.0,1000.0] ):
  save_dir   = os.path.join( data.save_dir, "deeper_meaning_dna_and_z_p_tissue_%0.2f_p_spear_%g_logreg"%(threshold,min_p_value) )
  check_and_mkdir(save_dir) 
  
  dna_auc_dir   = os.path.join( data.save_dir, "dna_auc_latent" )
  spearmanr_dir = os.path.join( data.save_dir, "spearmans_latent_tissue" )
  
  spear_dna_z_rho = pd.read_csv( spearmanr_dir + "/dna_z_rho.csv", index_col="gene" )
  spear_dna_z_p   = pd.read_csv( spearmanr_dir + "/dna_z_p.csv", index_col="gene" )
  dna_z_auc     = pd.read_csv( dna_auc_dir + "/dna_z_auc.csv", index_col="gene" )
  dna_z_auc_p   = pd.read_csv( dna_auc_dir + "/dna_z_auc_p.csv", index_col="gene" )
  dna_z_p = dna_z_auc_p
  
  rna_z_rho = pd.read_csv( spearmanr_dir + "/rna_z_rho.csv", index_col="gene" )
  rna_z_p   = pd.read_csv( spearmanr_dir + "/rna_z_p.csv", index_col="gene" )

  mirna_z_rho = pd.read_csv( spearmanr_dir + "/mirna_z_rho.csv", index_col="gene" )
  mirna_z_p   = pd.read_csv( spearmanr_dir + "/mirna_z_p.csv", index_col="gene" )
 
  meth_z_rho = pd.read_csv( spearmanr_dir + "/meth_z_rho.csv", index_col="gene" )
  meth_z_p   = pd.read_csv( spearmanr_dir + "/meth_z_p.csv", index_col="gene" )
  
  # first compare AUCs and Rho's
  
  # f = pp.figure()
  # ax1 = f.add_subplot(221)
  # ax2 = f.add_subplot(222)
  # ax3 = f.add_subplot(223)
  # ax4 = f.add_subplot(224)
  # ax1.plot( spear_dna_z_rho.values.flatten(), dna_z_auc.values.flatten(), '.', alpha=0.25); ax1.set_xlabel("Spearman Rho"); ax1.set_ylabel("AUC"); ax1.set_title( "rho v auc")
  # ax2.loglog( spear_dna_z_p.values.flatten(), dna_z_auc_p.values.flatten(), '.', alpha=0.25); ax2.set_xlabel("Spearman Rho"); ax2.set_ylabel("AUC"); ax2.set_title( "spearman v auc p-values")
  #
  # ax3.semilogy( spear_dna_z_rho.values.flatten(), spear_dna_z_p.values.flatten(), '.', alpha=0.25); ax3.set_xlabel("Spearman Rho"); ax3.set_ylabel("Spearman p-value"); ax3.set_title( "rho v p-value")
  # ax4.semilogy( dna_z_auc.values.flatten(), dna_z_auc_p.values.flatten(), '.', alpha=0.25); ax4.set_xlabel("AUC"); ax4.set_ylabel("AUC p-value"); ax4.set_title( "auc v p-value")
  #
  # f.savefig( save_dir + "/spearman_v_auc.png", format="png", dpi=300)
  
  Z           = data.Z

  data.data_store.open()
  try:
    T=data.data_store["/CLINICAL_USED/TISSUE"].loc[ Z.index ]
  except:
    T=data.data_store["/CLINICAL/TISSUE"].loc[ Z.index ]
  data.data_store.close()
  
  dna_names = data.dna.sum(0).sort_values(ascending=False).index.values
  n_dna = len(dna_names)
  dna = data.dna[ dna_names ]
  
  z_scores = -np.sum( np.log2(dna_z_p),1)
  
  f=pp.figure( figsize=(24,12) )
  genes = z_scores.sort_values().index.values #dna_names[:nbr_genes]
  #pdb.set_trace()
  #min_p_value = 1e-3
  k_idx = 1
  results = []
  for gene in genes:
    p_values = dna_z_p.loc[gene][ dna_z_p.loc[gene] < min_p_value ]
    if len(p_values)==0:
      print "skipping ",gene
      continue
    
    dna_values = dna[gene].values
    ids_with_n, relevant_tissues = ids_with_at_least_p_mutations( dna_values, T, p = threshold )
    
    nbr_cancer_types = len(relevant_tissues)
    mutations = pp.find( dna_values[ids_with_n] == 1)
    wildtype = pp.find( dna_values[ids_with_n]==0)  
    
    if len(mutations)==0:
      continue
    
    best_z_names = p_values.sort_values().index.values
    if len(best_z_names) < 5:
      best_z_names = dna_z_p.loc[gene].sort_values()[:5].index.values
    elif len(best_z_names) > 20:
      best_z_names = best_z_names[:20]
    best_z_rna_z_p = rna_z_p[ best_z_names ]
    best_z_rna_z_rho = rna_z_rho[ best_z_names ]
    print "================"
    best_z_score_rna = -np.sum(np.log2(best_z_rna_z_p+1e-200),1)
    print best_z_score_rna.sort_values(ascending=False)[:20].index.values
    # for z_name in best_z_names:
    #   print best_z_rna_z_p[ z_name ].sort_values()[:10]
    #   print (-np.abs(best_z_rna_z_rho[ z_name ])).sort_values()[:10]
      #pdb.set_trace()
    print gene, p_values.sort_values().index.values
    
    y_true = dna_values[ids_with_n]
    X = Z[ids_with_n][best_z_names].values
    
    #MCV = GenerativeBinaryClassifierKFold( K = 10 )
    MCV = LogisticBinaryClassifierKFold( K = 10 )
    
    best_auc = -np.inf
    best_ridge = 0.0
    best_y_est = None
    best_auc_p = -np.inf
    for ridge in ridges:
     
      #y_est_cv = MCV.fit_and_prob( y_true, X, ridge=ridge, cov_type="shared" )
      y_est_cv = MCV.fit_and_prob( y_true, X, C=ridge )
      #if gene == "BRAF":
      #  pdb.set_trace()
      auc_y_est_cv, p_value_y_est_cv = auc_and_pvalue( y_true, y_est_cv )
      print "for ridge in ridges ",ridge, auc_y_est_cv
      if auc_y_est_cv > best_auc:
        best_auc = auc_y_est_cv
        best_auc_p = p_value_y_est_cv
        best_ridge = ridge
        best_y_est = y_est_cv
    
    y_est_cv = best_y_est
    auc_y_est_cv, p_value_y_est_cv =   best_auc, best_auc_p
       
    # M=LogisticBinaryClassifier()
    # #M = GenerativeBinaryClassifier()
    # M.fit( y_true, X, C=best_ridge )
    # #M.fit( y_true, X, ridge=best_ridge )
    # y_est = M.prob(X)
    # auc_y_est, p_value_y_est = auc_and_pvalue( y_true, y_est )
    
    
    #print "learned auc (tr) = %0.3f  p-value: %0.f"%(auc_y_est, p_value_y_est)
    print "learned auc (cv) = %0.3f  p-value: %0.f"%(auc_y_est_cv, p_value_y_est_cv)
    print "compare to:"
    other_aucs = []
    signs = []
    for z_name in best_z_names:
      print z_name, dna_z_auc[z_name].loc[gene], dna_z_auc_p[z_name].loc[gene]
      other_aucs.append( max( float(dna_z_auc[z_name].loc[gene]), float(1-dna_z_auc[z_name].loc[gene])) )
      if dna_z_auc[z_name].loc[gene]<0.5:
        signs.append(-1)
      else:
        signs.append(1)
    signs = np.array(signs)[np.newaxis,:]
    #pdb.set_trace()
    print "================"
    results.append( [gene, {"tissues":relevant_tissues, "nbr_tissues":nbr_cancer_types,"wildtype":len(wildtype),\
                    "mutations":len(mutations)},list(p_values.sort_values().index.values),\
                    list(best_z_score_rna.sort_values(ascending=False)[:20].index.values),\
                     #["train", float(auc_y_est), float(p_value_y_est)], \
                     ["cv", float(auc_y_est_cv), float(p_value_y_est_cv)],\
                     other_aucs ] )
     

    if len(best_z_names)>5:
      gene_dir =  os.path.join( save_dir, "%s"%(gene) )
      check_and_mkdir(gene_dir)
      y_order = np.argsort(y_est_cv)
      
      #signs = []
      X_order = X[y_order,:]
      X_order -= X_order.mean(0)
      X_order /= X_order.std(0)
      X_order *=signs
      Z_order = pd.DataFrame( X_order, index = Z[ids_with_n].index.values[y_order], columns = best_z_names )

      #k_colors = np.array([k_pallette[kmeans_patients_labels[i]] for i in order_labels] )

      print "MAKNING ", gene
      size1=12
      size2=4
      r = size1/size2
      f = pp.figure( figsize=(size1,size2))
      ax = f.add_subplot(111)
      ax.imshow(Z_order.T, cmap='rainbow', interpolation='nearest', aspect=float(len(y_true))/(len(best_z_names)*r))
      #ax.imshow(differ)
      ax.autoscale(False)
      
      #h = sns.heatmap( Z_order, row_colors=None, row_cluster=False, col_cluster=False, figsize=(12,12) )
      #pdb.set_trace()
      #pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
      #pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
      #pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
      #pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
      #h.ax_row_dendrogram.set_visible(False)
      #h.ax_col_dendrogram.set_visible(False)
      #h.cax.set_visible(False)
    #h.ax_heatmap.hlines(len(kmeans_patients_labels)-pp.find(np.diff(np.array(kmeans_patients_labels)[order_labels]))-1, *h.ax_heatmap.get_xlim(), color="black", lw=5)
    #h.ax_heatmap.vlines(pp.find(np.diff(np.array(kmeans_z_labels)[order_labels_z]))+1, *h.ax_heatmap.get_ylim(), color="black", lw=5)

      f.savefig( gene_dir + "/sorted_by_%s.png"%(gene), fmt="png", bbox_inches='tight')
      pp.close('all')
    
    
    
  check_and_mkdir(save_dir) 
  
  fptr = open( save_dir + "/pan_cancer_dna.yaml","w+" )
  fptr.write( yaml.dump(results))
  fptr.close()
  
  #pdb.set_trace()
  
if __name__ == "__main__":
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  data = load_data_and_fill( data_location, results_location )
  
  #dna_auc_using_latent_space( data, force =True )
  #spearmanr_latent_space_by_inputs(data, force=True)
  ridges = [0.00001, 0.001,1.0]
  
  #deeper_meaning_dna_and_z_correct( data, K=10, min_p_value=1e-3, threshold=0, Cs = [0.00001,0.0001, 0.001,0.1,1.0,10.0,1000.0] )  
  nbr_dna_genes2process = 100
  Cs = [0.00001,0.0001, 0.001,0.01,0.1,1.0,10.0,100.0,1000.0] 

  #correlation_latent_space_by_inputs(data, force=True)
  
  #describe_latent(data)
  #cluster_latent_space_by_inputs( data )
  
  #repeat_kmeans( data, data.RNA_fair, "RNA", K = 5, repeats=10 )
  #repeat_kmeans( data, data.Z, "Z", K = 5, repeats=10 )
  #survival_regression_global( data, data.Z, "Z", K = 20, fitter = CoxPHFitter  )
  #survival_regression_global( data, data.Z, "Z", K = 10, fitter = CoxPHFitter  )
  #survival_regression_global( data, data.Z, "Z", K = 5, fitter = CoxPHFitter  )
  
  # for K in [2,5]:#,5]:
  #   #K=2
  #   L2s_Z = [0.0]
  #   L2s_RNA = [0.001,0.01,0.1,1.0]
  #   #survival_regression_global( data, data.Z, "Z", L2s_Z, K = K, repeats=20, fitter = CoxPHFitter  )
  #   survival_regression_global( data, data.Z, "Z", L2s_Z, K = K, repeats=50, fitter = CoxPHFitter  )
  #   survival_regression_global_g0_v_g3( data, data.Z, "Z", L2s_Z, K = K, repeats=50, fitter = CoxPHFitter  )
  #   #survival_regression_global( data, data.RNA_scale, "RNA_scale", L2s_RNA, K = K, repeats=5, fitter = CoxPHFitter  )
  #   #survival_regression_global( data, data.RNA_fair, "RNA_fair", L2s_RNA, K = K, repeats=5, fitter = CoxPHFitter  )
  
  for K in [2]:#,5]:
    L2s_Z = [0.001,0.01,0.1,1.0]
    L2s_RNA = [0.0] #01,0.01,0.1,1.0]
    #survival_regression_global( data, data.Z, "Z", L2s_Z, K = K, repeats=5, fitter = CoxPHFitter  )
    #survival_regression_global( data, data.RNA_scale, "RNA_scale", L2s_RNA, K = K, repeats=5, fitter = CoxPHFitter  )
    #survival_regression_global( data, data.RNA_fair, "RNA_fair", L2s_RNA, K = K, repeats=5, fitter = CoxPHFitter  )
    #survival_regression_global( data, data.RNA_fair, "RNA_rank", L2s_Z, K = K, repeats=5, fitter = CoxPHFitter  )
    #survival_regression_global_g0_v_g3( data, data.RNA_fair, "RNA_rank", L2s_Z, K = K, repeats=5, fitter = CoxPHFitter  )
    survival_regression_global( data, data.RNA_scale, "RNA_scale", L2s_Z, K = K, repeats=1, fitter = CoxPHFitter  )
    survival_regression_global_g0_v_g3( data, data.RNA_scale, "RNA_scale", L2s_Z, K = K, repeats=1, fitter = CoxPHFitter  )
