from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.analyses.everything_functions import *
from tcga_encoder.analyses.survival_functions import *
import networkx as nx
#try:
from networkx.drawing.nx_agraph import graphviz_layout as g_layout
#except:
#  print "could not import graphviz "
#from networkx.drawing.nx_agraph import spring_layout 
  
def load_data_and_fill( data_location, results_location ):
  input_sources = ["RNA","miRNA","METH"]
  data_store = load_store( data_location, "data.h5")
  model_store = load_store( results_location, "full_vae_model.h5")
  fill_store = load_store( results_location, "full_vae_fill.h5")
  
  subtypes = load_subtypes( data_store )
  
  #tissues = data_store["/CLINICAL/TISSUE"].loc[barcodes]
  
  survival = PanCancerSurvival( data_store )
  
  Z = load_latent( fill_store )
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
  #for name in names:
  #  G.add_node( name )
  #  # tissue = bc.split("_")[0]
  #  # node_colors.append( tissue2color[bc.split("_")[0]] )
  #  # json_node.append( {"size":10,"tissue":tissue,"id":bc,"score":tissue2idx[ tissue ]})
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
      if np.abs(w) >= t:
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
  
def draw_graph( G, save_name, node_colors=[], with_labels=True, alpha=1, font_size=12 ):
  layout=g_layout
  try:
    pos=layout(G)
  except:
    pos = nx.spring_layout(G)
  pp.figure(figsize=(20,20))
  nx.draw( G, pos, \
              node_color=node_colors, 
              with_labels=with_labels, 
              hold=with_labels, \
              alpha=alpha, \
              font_size=font_size
              )
  pp.savefig(save_name, fmt='png',dpi=300)
  pp.close('all')
     
def cluster_genes_by_hidden_weights( data, correlation_thresholds = [0.15,0.25,0.5,0.75] ):
  # take weight matrix, find small groupds of co-activated weights
  W = pd.concat( data.W_input2h.values() )
  W_norm = W / np.sqrt( ( W*W ).sum(0) )
  print "computing hidden weight correlations..."
  #pdb.set_trace()
  C = W_norm.T.corr()
  data.input2input_w_corr = C
  
  save_dir = os.path.join( data.save_dir, "input2hidden_weight_clustering2" )
  check_and_mkdir(save_dir) 
  print "saving hidden weight correlations..."
  data.input2input_w_corr.to_csv( save_dir + "/input2input_w_corr.csv" )
  
  data.input2input_w_corr_graph = OrderedDict()
  for correlation_threshold in correlation_thresholds:
    print "making graph hidden weight correlations...", correlation_threshold
    data.input2input_w_corr_graph[correlation_threshold] = make_graph_from_correlations( data.input2input_w_corr, correlation_threshold )
    
    draw_graph( data.input2input_w_corr_graph[correlation_threshold], save_dir+"/corr_%0.2f.png"%(correlation_threshold) )
  #pdb.set_trace()
  
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
  
  
if __name__ == "__main__":
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  data = load_data_and_fill( data_location, results_location )
  
  result = cluster_genes_by_hidden_weights(data)
  #cluster_genes_by_hidden_weights(data)
  #cluster_genes_by_hidden_weights(data)
  #cluster_genes_by_hidden_weights(data)
  
  
  