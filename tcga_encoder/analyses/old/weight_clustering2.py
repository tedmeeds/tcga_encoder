from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
#from tcga_encoder.data.pathway_data import Pathways
from tcga_encoder.data.hallmark_data import Pathways
from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
#from tcga_encoder.algorithms import *
import seaborn as sns
from sklearn.manifold import TSNE, locally_linear_embedding
from scipy import stats

from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import json
from networkx.readwrite import json_graph
size_per_unit=0.25

def process_all_sources( save_dir, source2w, prefix ):
  sources = source2w.keys()
  ws      = source2w.values()
  #pdb.set_trace()
  shapes2use = ["circle","square","triangle-up"]
  scores2use = [0,0.5,1.0]
  colors2use = ["red","blue","green"]
  counts  = [len(w) for w in ws]
  
  
  W = pd.concat(ws,0)
  
  #W=W/np.sqrt( np.sum( np.square( W.values ),0 ))
  #pdb.set_trace()
  n_features = len(W)
  shapes = []
  scores = []
  colors = []
  for i in xrange(n_features):
    if i < counts[0]:
      shapes.append( shapes2use[0] )
      scores.append( scores2use[0] )
      colors.append( colors2use[0] )
    elif i < counts[1]+counts[0]:
      shapes.append( shapes2use[1] )
      scores.append( scores2use[1] )
      colors.append( colors2use[1] )
    else:
      shapes.append( shapes2use[2] )
      scores.append( scores2use[2] )
      colors.append( colors2use[2] )
  
  shapes = np.array(shapes,dtype=str)
  colors = np.array(colors,dtype=str)
  scores = np.array(scores,dtype=float)
  sizes  = 10*np.ones(n_features)
  w_corr = W.T.corr()
  corr_v = w_corr.values
  names = w_corr.columns
  
  min_corr = 0.8
  keep_ids = []
  for i in xrange(n_features):
    c = corr_v[i]
    if sum( np.abs(c) > min_corr ) > 1:
      keep_ids.append(i )
  
  print "keeping %d of %d nodes"%(len(keep_ids),n_features)
  keep_ids = np.array(keep_ids)
  keep_names = names[keep_ids]
  keep_shapes = shapes[keep_ids]
  keep_sizes = sizes[keep_ids]
  keep_scores = scores[keep_ids]
  keep_colors = colors[keep_ids]
  w_corr = w_corr.loc[ keep_names ][keep_names]
  corr_v = w_corr.values
  n_features = len(w_corr)
  #pdb.set_trace()
  #
  
  tau = min_corr
  G=nx.Graph()
  i=0
  nodes = []
  links = []
  nodes_ids=[]
  node_ids = OrderedDict()
  #flare = OrderedDict()
  for i,c,name_i in zip( xrange( n_features ), corr_v, keep_names ):
    for j,name_j in zip( xrange(n_features), keep_names ):
      if j > i:
        if np.abs( c[j] ) > tau:
          
          if node_ids.has_key(name_i) is False:
            nodes.append( {"id":name_i})
          if node_ids.has_key(name_j) is False:
            nodes.append( {"id":name_j})
          links.append( {"source":i,"target":j,"w":c[j]} ) 
          nodes_ids.append(i)
          nodes_ids.append(j)
  nodes_ids = np.unique( np.array(nodes_ids))
  
  json_node = []
  for i,name,size,score,shape,color in zip( xrange( n_features ), keep_names, keep_sizes, keep_scores, keep_shapes, keep_colors ):
    # name = names[i]
    # size = int(80*total_weights[i])
    # score = 1
    # type = "circle"
    json_node.append( {"size":size,"score":score,"id":name,"type":shape})
    G.add_node(name, color=color, size=size )
    
  
  json.dump({"nodes":json_node,"links":links,"directed": False,
  "multigraph": False,"graph": []}, open(save_dir+'/all_force%s3.json'%(prefix),'w'))
    
  for link in links:
    G.add_edge( keep_names[link["source"]], keep_names[link["source"]], weight = np.abs(link["source"]) ) 
  from networkx.drawing.nx_agraph import graphviz_layout
  layout=graphviz_layout
  
  print "laying out graph"
  pos=layout(G)
  
  pp.figure(figsize=(45,45))
  print "drawing graph"
  nx.draw(G,pos,
              with_labels=True, hold=False, alpha=0.25, font_size=12
              )
  # d = json_graph.node_link_data(G)
  
  G.clear()

  pp.savefig(save_dir + "/mwst%s.png"%(prefix), fmt='png',dpi=300) 
  
  

def process_source( save_dir, source, w, percent_weights, prefix="" ):
  #corr = w.T.corr()
  sorted_flattened = np.sort( np.abs(w.values.flatten()) )
  n = len(sorted_flattened)
  threshold = sorted_flattened[ - int( float(n)*percent_weights) ]
  
  #w = w[ np.abs(w) >= threshold ].fillna(0)
  
  #w = np.sign(w)
  #pdb.set_trace()
  total_weights = np.abs(w.values).sum(1)
  corr = w.T.corr()
  corr.sort_index(inplace=True)
  corr = corr[ corr.index.values ]
  corr_v = corr.values
  names = corr.columns
  n_source = len(names)
  size1 = max( min( 40, int( w.values.shape[0]*size_per_unit ) ), 12 )

  size2 = max( min( 40, int( w.values.shape[0]*size_per_unit )), 12 )

  
  # cmap = sns.palplot(sns.light_palette((260, 75, 60), input="husl"))
  # htmap3 = sns.clustermap ( corr, cmap=cmap, square=True, figsize=(size1,size1) )
  # pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  # pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  # pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  # pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  # htmap3.ax_row_dendrogram.set_visible(False)
  # htmap3.ax_col_dendrogram.set_visible(False)
  # pp.savefig( save_dir + "/weights_%s_clustermap%s.png"%(source,prefix), fmt="png", bbox_inches = "tight")
  #
  #labels = [s.get_text() for s in htmap3.ax_heatmap.yaxis.get_majorticklabels()]
  
  #corr = corr[labels]
  #corr = corr.loc[labels]
  corr_v = corr.values
  names = corr.columns
  # csr = csr_matrix(np.triu(1.0-np.abs(meth_corr.values)))
  # Tcsr = minimum_spanning_tree(csr)
  # as_mat = Tcsr.toarray()
  #pdb.set_trace()
  pp.figure(figsize=(45,45))
  
  tau = 0.5
  G=nx.Graph()
  i=0
  nodes = []
  links = []
  nodes_ids=[]
  node_ids = OrderedDict()
  #flare = OrderedDict()
  for i in xrange( n_source ):
    x = corr_v[i]
    name_i = names[i]
    #flare[name_i] = []
    for j in xrange(n_source):
      if j > i:
        if np.abs( x[j] ) > tau:
          name_j = names[j]
          G.add_edge(name_i, name_j, weight = np.abs(x[j]) )
          if node_ids.has_key(name_i) is False:
            nodes.append( {"id":name_i})
            #node_ids[name_i] = 1
            #flare[name_i] = []
          if node_ids.has_key(name_j) is False:
            nodes.append( {"id":name_j})
            #node_ids[name_i] = 1
          links.append( {"source":i,"target":j} ) #, "value":np.abs(x[j])} )
          #flare[name_i].append( name_j )
          nodes_ids.append(i)
          nodes_ids.append(j)
  nodes_ids = np.unique( np.array(nodes_ids))
  
  json_node = []
  for i in xrange( n_source ):
    name = names[i]
    size = int(80*total_weights[i])
    score = 1
    type = "circle"
    json_node.append( {"size":size,"score":score,"id":name,"type":type})
    
  
  from networkx.drawing.nx_agraph import graphviz_layout
  layout=graphviz_layout
  #layout=nx.spectral_layout
  pos=layout(G)
  nx.draw(G,pos,
              with_labels=True,
              node_size=20, hold=False, node_color='b', alpha=0.25, font_size=12
              )
  d = json_graph.node_link_data(G)
  #pdb.set_trace()
  json.dump({"nodes":json_node,"links":links,"directed": False,
  "multigraph": False,"graph": []}, open(save_dir+'/%s_force%s2.json'%(source,prefix),'w'))
  
  # names = flare.keys()
  # targets = flare.values()
  # for target_list in targets:
    
  #
  # flares=[]
  # targets = []
  # for name_i,list_j in flare.iteritems():
  #   o=OrderedDict()
  #   o["name"] = name_i
  #   o["size"] =  100*len(list_j)
  #   o["imports"] = list_j
  #   flares.append( o )
  #
  #   #targets.extend( )
  #
  #
  # json.dump(flares, open(save_dir+'/%s_flare%s.json'%(source,prefix),'w'))
  #from networkx.readwrite import json_graph
  G.clear()
  #pp.title("%s"%(tissue_name))
  pp.savefig(save_dir + "/%s_mwst%s.png"%(source,prefix), fmt='png',dpi=300) 
  
  print " only doing one source now"
  
  


def join_weights( W_hidden2z, W_hidden ):
  W = {}
  n_z = W_hidden2z.shape[1]
  columns = np.array( ["z_%d"%i for i in range(n_z)])
  
  for input_source, source_w in W_hidden.iteritems():
    #pdb.set_trace()
    W[ input_source ] = pd.DataFrame( np.dot( source_w, W_hidden2z ), index = source_w.index, columns = columns )

  return W
  
def get_hidden2z_weights( model_store ):
  layer = "rec_z_space"
  model_store.open()
  w = model_store[ "%s"%(layer) + "/W/w%d"%(0)].values
  model_store.close()
  return w
  
def get_hidden_weights( model_store, input_sources, data_store ):
  
  rna_genes = data_store["/RNA/FAIR"].columns
  meth_genes = data_store["/METH/FAIR"].columns
  mirna_hsas = data_store["/miRNA/FAIR"].columns
  
  post_fix = "_scaled"
  idx=1
  n_sources = len(input_sources)
  W = {}
  for w_idx, input_source in zip( range(n_sources), input_sources ):
    w = model_store[ "rec_hidden" + "/W/w%d"%(w_idx)].values
    #pdb.set_trace()
    
    
    d,k = w.shape
    
    columns = np.array( ["h_%d"%i for i in range(k)])
    if input_source == "RNA":
      rows = rna_genes
      print input_source, w.shape, len(rows), len(columns)
      W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
      
    if input_source == "miRNA":
      rows = mirna_hsas
      print input_source, w.shape, len(rows), len(columns)
      W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
      
    if input_source == "METH":
      rows = meth_genes
      #rows = np.array( [ "M-%s"%g for g in meth_genes], dtype=str )
      print input_source, w.shape, len(rows), len(columns)
      W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
      
    if input_source == "TISSUE":
      rows = tissue_names
      print input_source, w.shape, len(rows), len(columns)
      W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
  model_store.close()
  return W
  
def auc_standard_error( theta, nA, nN ):
  # from: Hanley and McNeil (1982), The Meaning and Use of the Area under the ROC Curve
  # theta: estimated AUC, can be 0.5 for a random test
  # nA size of population A
  # nN size of population N
  
  Q1=theta/(2.0-theta); Q2=2*theta*theta/(1+theta)
  
  SE = np.sqrt( (theta*(1-theta)+(nA-1)*(Q1-theta*theta) + (nN-1)*(Q2-theta*theta) )/(nA*nN) )
  
  return SE
  
def auc_test( true_y, est_y ):
  n = len(true_y)
  n_1 = true_y.sum()
  n_0 = n - n_1 
  
  if n_1 == 0 or n_1 == n:
    return 0.5, 0.0, 0.0, 1.0
    
    
  auc = roc_auc_score( true_y, est_y )
  difference = auc - 0.5

  if difference < 0:
    # switch labels
    se  = auc_standard_error( auc, n_0, n_1 )
    se_null  = auc_standard_error( 0.5, n_0, n_1 )
  else:
    se  = auc_standard_error( 1-auc, n_1, n_0 )
    se_null  = auc_standard_error( 0.5, n_1, n_0 )
    
  se_combined = np.sqrt( se**2 + se_null**2 )
  
  z_value = np.abs(difference) / se_combined 
  p_value = 1.0 - stats.norm.cdf( np.abs(z_value) )
  
  return auc, se, z_value, p_value
  
  
def find_keepers_over_groups( z, groups, name, nbr2keep, stats2use ):
  inners = []; p_inners=[]
  mx_inner = 0.0
  norm_z = np.linalg.norm(z)
  for X, stat in zip( groups, stats2use ):
    
    pearsons = np.zeros( X.shape[1] )
    pvalues  = np.zeros( X.shape[1] )
    for x,x_idx in zip( X.values.T, range(X.shape[1])):
      
      if stat == "pearson":
        pearsons[x_idx], pvalues[x_idx] = stats.pearsonr( z, x )
      elif stat == "auc":
        true_y = (x>0).astype(int)
        auc, se, zvalue, pvalue = auc_test( true_y, z ) #np.sqrt( ses_tissue**2 + se_r_tissue**2 )
          
        pearsons[x_idx] = auc-0.5
        pvalues[x_idx] = pvalue
        #pdb.set_trace()
      
    #norms = norm_z*np.linalg.norm( X, axis=0 )
    
    #inner = pd.Series( np.dot( z, X )/norms, index = X.columns, name=name )
    inner = pd.Series( pearsons, index = X.columns, name=name )
    p_inner = pd.Series( pvalues, index = X.columns, name=name )
    
    inners.append(inner)
    p_inners.append(p_inner)
    
    this_mx = np.max(np.abs(inner))
    if this_mx > mx_inner:
      mx_inner = this_mx
  all_keepers = []
  #all_pvalues = []
  for inner,p_inner in zip(inners,p_inners):
    #inner.sort_values(inplace=True)
    #inner = inner / mx_inner
    
    #abs_inner = np.abs( inner )
    #ordered = np.argsort( -inner.values )
    ordered = np.argsort( p_inner.values )
    
    ordered = pd.DataFrame( np.vstack( (inner.values[ordered],p_inner.values[ordered] ) ).T, index =inner.index[ordered],columns=["r","p"] )
    #pdb.set_trace()
    #keepers = pd.concat( [ordered[:nbr2keep], ordered[-nbr2keep:]], axis=0 )
    keepers = ordered[:nbr2keep]
    #pdb.set_trace()
    #keepers = keepers.sort_values()
    all_keepers.append(keepers)
  
  return all_keepers
  
def find_keepers(z, X, name, nbr2keep):
  inner = pd.Series( np.dot( z, X ), index = X.columns, name=name )
  inner.sort_values(inplace=True)
  inner = inner / np.max(np.abs(inner))
  #signed = np.sign( inner )
  abs_inner = np.abs( inner )
  ordered = np.argsort( -abs_inner.values )
  ordered = pd.Series( inner.values[ordered], index =inner.index[ordered],name=name )
  
  keepers = ordered[:nbr2keep]
  keepers = keepers.sort_values()
  
  return keepers
  
def main( data_location, results_location ):
  pathway_info = Pathways()
  data_path    = os.path.join( HOME_DIR ,data_location ) #, "data.h5" )
  results_path = os.path.join( HOME_DIR, results_location )
  
  data_filename = os.path.join( data_path, "data.h5")
  fill_filename = os.path.join( results_path, "full_vae_fill.h5" )
  model_filename = os.path.join( results_path, "full_vae_model.h5" )
  
  save_dir = os.path.join( results_path, "weight_clustering" )
  check_and_mkdir(save_dir)
  z_dir = os.path.join( save_dir, "z_pics" )
  check_and_mkdir(z_dir)
  h_dir = os.path.join( save_dir, "h_pics" )
  check_and_mkdir(h_dir)
  print "HOME_DIR: ", HOME_DIR
  print "data_filename: ", data_filename
  print "fill_filename: ", fill_filename
  
  print "LOADING stores"
  data_store = pd.HDFStore( data_filename, "r" )
  fill_store = pd.HDFStore( fill_filename, "r" )
  model_store = pd.HDFStore( model_filename, "r" )
  
  Z_train = fill_store["/Z/TRAIN/Z/mu"]
  Z_val = fill_store["/Z/VAL/Z/mu"]
  
  #input_sources = ["METH","RNA","miRNA"]
  input_sources = ["RNA","miRNA","METH"]
  W_hidden = get_hidden_weights( model_store, input_sources, data_store )
  W_hidden2z = get_hidden2z_weights( model_store )
  
  size_per_unit = 0.25

  weighted_z = join_weights( W_hidden2z, W_hidden )
  barcodes = data_store["/CLINICAL/observed"][ data_store["/CLINICAL/observed"][["RNA","miRNA","METH","DNA"]].sum(1)==4 ].index.values
  tissues = data_store["/CLINICAL/TISSUE"].loc[barcodes]  
  
  tissue_names = tissues.columns
  tissue_idx = np.argmax( tissues.values, 1 )
  #n = len(Z)
  n_tissues = len(tissue_names)
  
  n_h = W_hidden2z.shape[0]
  
  print "+++++++++++++++++++++++++++"
  print " find weights that are significant together, not"
  #W_hidden["RNA_miRNA"] = pd.concat( [W_hidden["RNA"],W_hidden["miRNA"] ],0 )
  
  percent_weights = 0.05
  
  process_all_sources( save_dir, weighted_z, prefix="_all_Z" )
  process_all_sources( save_dir, W_hidden, prefix="_all" )
  
  # for source, w in weighted_z.iteritems():
  #
  #   process_source( save_dir, source, w, percent_weights, prefix="_Z" )
  #
  # for source, w in W_hidden.iteritems():
  #
  #   process_source( save_dir, source, w, percent_weights )
  #   #break
  pp.close('all')

  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )