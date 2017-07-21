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
  
  everything_dir = os.path.join( os.path.join( HOME_DIR, results_location ), "everything" )
  check_and_mkdir(everything_dir)

  data                = EverythingObject()
  data.input_sources  = input_sources
  data.data_store     = data_store
  data.model_store    = model_store
  data.fill_store     = fill_store
  data.subtypes       = subtypes
  data.survival       = survival
  data.Z              = Z
  data.Z_std           = Z_std
  data.H              = H
  data.W_input2h      = get_hidden_weights( model_store, input_sources, data_store )
  data.W_h2z          = get_hidden2z_weights( model_store )
  data.weighted_W_h2z = join_weights( data.W_h2z, data.W_input2h  )
  data.RNA_scale      = RNA_scale
  data.miRNA_scale    = miRNA_scale
  data.METH_scale     = METH_scale
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
def  cluster_latent_space_by_inputs( data ):
  Z           = data.Z
  RNA_scale   = 2.0 / (1+np.exp(-data.RNA_scale)) -1   
  miRNA_scale = 2.0 / (1+np.exp(-data.miRNA_scale))-1 
  METH_scale  = 2.0 / (1+np.exp(-data.METH_scale ))-1  
  rna_names   = data.rna_names    
  mirna_names = data.mirna_names 
  meth_names  = data.meth_names   
  
  z_names = data.z_names
  n_rna   = len(rna_names)
  n_mirna = len(mirna_names)
  n_meth  = len(meth_names)

  print "computing RNA-Z spearman rho's"
  rho_rna_z = stats.spearmanr( RNA_scale.values, Z.values )
  print "computing miRNA-Z spearman rho's"
  rho_mirna_z = stats.spearmanr( miRNA_scale.values, Z.values )
  print "computing METH-Z spearman rho's"
  rho_meth_z = stats.spearmanr( METH_scale.values, Z.values )
  
  rna_z_rho = pd.DataFrame( rho_rna_z[0][:n_rna,:][:,n_rna:], index = rna_names, columns=z_names)
  rna_z_p   = pd.DataFrame( rho_rna_z[1][:n_rna,:][:,n_rna:], index = rna_names, columns=z_names)
  
  mirna_z_rho = pd.DataFrame( rho_mirna_z[0][:n_mirna,:][:,n_mirna:], index = mirna_names, columns=z_names)
  mirna_z_p   = pd.DataFrame( rho_mirna_z[1][:n_mirna,:][:,n_mirna:], index = mirna_names, columns=z_names)
   
  meth_z_rho = pd.DataFrame( rho_meth_z[0][:n_meth,:][:,n_meth:], index = meth_names, columns=z_names)
  meth_z_p   = pd.DataFrame( rho_meth_z[1][:n_meth,:][:,n_meth:], index = meth_names, columns=z_names)

  pdb.set_trace()
  
def repeat_kmeans( data, K = 20, repeats=10 ):
  Z           = data.Z
  X=Z
  STD = data.Z_std
  #X = quantize(X)
  #X = normalize(X)
  
  save_dir = os.path.join( data.save_dir, "repeat_kmeans_K%d_std"%(K) )
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
  STD = STD.loc[bcs]
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
      kmeans = MiniBatchKMeans(n_clusters=K, random_state=r ).fit( X.values[ids,:] + STD.values[ids,:]*np.random.randn(len(ids),dim))
      kmeans_labels = kmeans.labels_
      #return kmeans
      for k in range(K):
        Ik = pp.find( kmeans_labels == k )
        #pdb.set_trace()
        new_X[Ik,:] += kmeans.cluster_centers_[k]
        for i in Ik:
          for j in Ik:
            affinity_matrix[i,j] +=1
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
      if len(k_bcs) > 5:
        kmf.fit(k_times, event_observed=k_events, label="k%d"%(k)  )
        ax=kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color=k_pallette[k],ci_show=False)
    pp.title("%s"%(tissue_name))
    pp.savefig( save_dir + "/%s_survival.png"%(tissue_name), format="png", dpi=300)
    pp.close('all')  
      

  
if __name__ == "__main__":
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  data = load_data_and_fill( data_location, results_location )
  
  #cluster_latent_space_by_inputs( data )
  
  repeat_kmeans( data, K = 5, repeats=100 )
  # result = cluster_genes_by_hidden_weights_spectral(data, Ks = [200,100,50])
  # result = cluster_genes_by_latent_weights_spectral(data, Ks = [100,50,20])
  #
  # result = hidden_neighbours( data, nbr=3 )
  # result = latent_neighbours( data, nbr=3 )
  
  #result = neighbour_differences( data, data.Z, data.weighted_W_h2z, "latent", nbr = 10)
  #result = neighbour_differences( data, data.H, data.W_input2h, "hidden", nbr=10 )
  #within_tissue_neighbour_differences( data, data.Z, data.weighted_W_h2z, "within_tissue_latent", nbr = 20 )
  #within_tissue_neighbour_differences( data, data.H, data.W_input2h, "within_tissue_hidden", nbr = 20 )

  #cosine_within_tissue_neighbour_differences( data, data.Z, data.weighted_W_h2z, "within_tissue_latent", nbr = 20 )
  #cosine_within_tissue_neighbour_differences( data, data.H, data.W_input2h, "within_tissue_hidden", nbr = 20 )

  # G=data.nn_latent["full_G"]
  # adj_dict = G.adj
  # n = len(adj_dict)
  # A = np.zeros( (n,n), dtype=int )
  # name2idx = OrderedDict()
  # i=0
  # for name in data.Z.index.values:
  #   name2idx[name] = i; i+=1
  #
  # for k,v in adj_dict.iteritems():
  #   for k2 in v.keys():
  #     A[ name2idx[k], name2idx[k2]] = 1
  
  #cluster_genes_by_hidden_weights(data)
  #cluster_genes_by_hidden_weights(data)
  #cluster_genes_by_hidden_weights(data)
  
  
  