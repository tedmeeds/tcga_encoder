from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
from tcga_encoder.algorithms import *
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("talk")

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#from mutation_variants.helpers import *


def viz_weights_vertical( w, names  ):
  order = np.argsort( -w )
  
  d = len(w)
  
  #centers = 1+np.arange( n )
  x_values = np.arange( d )
  
  f = pp.figure( figsize=(6,10))
  
  ax1 = f.add_subplot(111)
  w_ordered = w[order]
  neg = w_ordered<0
  pos = w_ordered >=0
  ax1.plot( w_ordered[pos], x_values[pos] , 'bo' )
  ax1.plot( w_ordered[neg], x_values[neg] , 'ro' )
  pp.yticks(x_values, names[order], rotation='horizontal', fontsize=6)
  pp.margins(0.05)
  pp.subplots_adjust(left=0.15)
  ax1.grid(color='k', linestyle='--', linewidth=0.5,axis='x',alpha=0.5)
  return f, ax1

def viz_weights_horizontal( w, names ):
  
  #max_w = np.max(np.abs(W))
  #normed_W = W / max_w 
  
  order = np.argsort( -w )
  
  d = len(w)
  
  #centers = 1+np.arange( n )
  x_values = np.arange( d )
  
  f = pp.figure( figsize=(18,8))
  
  ax1 = f.add_subplot(111)
  w_ordered = w[order]
  neg = w_ordered<0
  pos = w_ordered >=0
  ax1.plot( x_values[pos], w_ordered[pos], 'bo' )
  ax1.plot( x_values[neg], w_ordered[neg], 'ro' )
  pp.xticks(x_values, names[order], rotation=90, fontsize=8)
  #pp.margins(0.05)
  #pp.subplots_adjust(bottom=0.15)
  ax1.grid(color='k', linestyle='--', linewidth=0.5,axis='y',alpha=0.5)
  return f, ax1
  
def decifer_weights( model_store, arch_dict, data_dict, weights, fold ):
  layers = arch_dict["generative"]["layers"]
  for layer in layers:
    if layer["name"] == "gen_dna_space":
      
      if layer["inputs"][0] != "hidden":
        these_weights = []
        for idx,input in zip( range(len(layer["inputs"])),layer["inputs"]):
          print "adding weights from ", input

          source_name = input.split("_")[0]
          if len(layer["inputs"])>1:
            pdb.set_trace()
            weight_names = ["%s_%s"%(source_name,s) for s in data_dict["store"]["/%s/FAIR"%(source_name)].columns]
          else:
            weight_names = data_dict["store"]["/%s/FAIR"%(source_name)].columns
          index = pd.MultiIndex.from_tuples(list(zip(weight_names,fold*np.ones(len(weight_names),dtype=int))), names=['name', 'fold'])
          try:
            W = pd.DataFrame( model_store["/dna_predictions/W/w%d"%(idx)].values, index = index, columns = data_dict["dna_genes"] )
          except:
            print "could not get /dna_predictions/W/w%d"%(idx)
            W = None
          print W
          these_weights.append(W)
        these_weights = pd.concat( these_weights, axis=0)
        #pdb.set_trace()
        weights.append(these_weights)
          
def plot_weights( w, title, dirname = None, figsize=(12,6), max_nbr = 100 ):
  
  top2use = np.argsort( -np.abs(w.values) )[:max_nbr]
  
  f,ax = viz_weights_vertical(w.values[top2use], w.index.values[top2use])
  if dirname is not None:
    f.savefig( dirname + "/w_%d_%s.svg"%(max_nbr,title), transparent=True, bbox_inches = 'tight', pad_inches=0.15, dpi=300 )
    f.savefig( dirname + "/w_%d_%s.png"%(max_nbr,title), transparent=False, bbox_inches = 'tight', pad_inches=0.15, dpi=300)
  
  top2use = np.argsort( -np.abs(w.values) )[:500]
  f,ax = viz_weights_horizontal(w.values[top2use], w.index.values[top2use])
  if dirname is not None:
    f.savefig( dirname + "/w_%d_%s.svg"%(500,title), transparent=True, bbox_inches = 'tight', pad_inches=0.15, dpi=300 )
    f.savefig( dirname + "/w_%d_%s.png"%(500,title), transparent=False, bbox_inches = 'tight', pad_inches=0.15, dpi=300)
    
def plot_binary_classification_result( y_true, y_est, title = None, dirname = None, figsize=(12,6) ):
  
  f = pp.figure(figsize=figsize)
  ax = f.add_subplot(111)
  
  I = pp.argsort( - y_est )
  
  #ax.plot( y_est[I], 'b.-', label = "est", alpha=0.75 )
  ax.scatter( np.arange(len(I)),  y_est[I], marker='o', s=20, linewidth=0.5, edgecolor='black', facecolor='blue', label = "est" )
  ax.scatter( np.arange(len(I)),  y_true[I], marker='o', s=20, linewidth=0.5, edgecolor='black', facecolor='red', label = "true" )
  ax.set_xlabel("Ranked Prediction")
  ax.set_ylabel("Estimate / Label")
  ax.legend()
  
  if title is not None:
    pp.title( title )
  if dirname is not None:  
    if title is not None:
      name = title
    else:
      name = ""
    f.savefig( dirname + "/classification_%s.svg"%(name), transparent=True, bbox_inches = 'tight', pad_inches=0.15, dpi=300 )
    f.savefig( dirname + "/classification_%s.png"%(name), transparent=False, bbox_inches = 'tight', pad_inches=0.15, dpi=300)


def add_variables( var_dict, data_dict ):
  # add very specific numbers:
  var_dict["dna_dim"]    = data_dict['dataset'].GetDimension("DNA")
  var_dict["meth_dim"]   = data_dict['dataset'].GetDimension("METH")
  var_dict["rna_dim"]    = data_dict['dataset'].GetDimension("RNA")
  var_dict["mirna_dim"]  = data_dict['dataset'].GetDimension("miRNA")
  var_dict["tissue_dim"] = data_dict['dataset'].GetDimension("TISSUE")
  
def load_architecture( arch_dict, data_dict ):
  add_variables( arch_dict[VARIABLES], data_dict )
  return arch_dict[NETWORK]( arch_dict, data_dict)
  
# def load_architectures( arches, data ):
#   networks = OrderedDict()
#   for arch in arches:
#     networks[ arch[NAME] ] = load_architecture( arch, data )
#   return networks

def main(yaml_file):
  y = load_yaml( yaml_file)
  
  logging_dict = {}
  #print "Loading data"
  load_data_from_dict( y[DATA] )
  algo_dict = y[ALGORITHM]
  arch_dict = y[ARCHITECTURE]
  data_dict = y[DATA] #{N_TRAIN:4000}
  logging_dict = y[LOGGING]
  
  n_xval_folds = algo_dict["n_xval_folds"]
  
  logging_dict[SAVEDIR] = os.path.join( HOME_DIR, os.path.join( logging_dict[LOCATION], logging_dict[EXPERIMENT] ) )
  
  
  summary_location_dir = os.path.join( logging_dict[SAVEDIR], "K=%d_xval_summary"%(n_xval_folds) )
  check_and_mkdir(summary_location_dir)
  print "LOGGING AT"
  print summary_location_dir
  fold_fill_dna = []
  fold_loglik_dna = []
  weights = []
  for fold in range(n_xval_folds):
    print "Gather XVAL = %d"%(fold)
    
    fold_location_dir = os.path.join( logging_dict[SAVEDIR],"fold_%d_of_%d"%(fold+1,y['algorithm']['n_xval_folds'] ) )
    
    #print fold_location_dir
    fill_store = pd.HDFStore( os.path.join( fold_location_dir, "full_vae_fill.h5" ), "r" )   
    model_store = pd.HDFStore( os.path.join( fold_location_dir, "full_vae_model.h5" ), "r" )   
    #print model_store
    fill_store = pd.HDFStore( os.path.join( fold_location_dir, "full_vae_fill.h5" ), "r" )
    
    #decifer_weights( model_store, arch_dict, data_dict, weights, fold )
    #print fill_store
    fold_fill_dna.append( fill_store["/Fill/VAL/DNA"] )
    fold_loglik_dna.append( fill_store["/Loglik/VAL/DNA"] )
    
    model_store.close()
    fill_store.close()
    pdb.set_trace()
   
  if len(weights) > 0:
    weights      = pd.concat(weights,axis=0) 
    mean_weights = weights.mean(level=0)
    std_weights  = weights.std(level=0)
  
  fill_dna  = pd.concat( fold_fill_dna ).sort_index()
  loglik_dna = pd.concat( fold_loglik_dna ).sort_index()

  fill_barcodes = fill_dna.index.values
  loglik_barcodes = loglik_dna.index.values
  
  barcodes = fill_barcodes
  data_store = data_dict["store"]
  dna = data_store["/DNA/channel/0"].loc[ barcodes ]
  
  if data_dict.has_key("dna_genes"):
    dna = dna[ data_dict["dna_genes"]]
    
  f_auc_curves = pp.figure( figsize=(10,10))
  ax_auc_curves = f_auc_curves.add_subplot(111)
  ax_auc_curves.plot( [0,0], [1,1], 'k--' )
  aucs = []  
  logliks  = []
  for gene in fill_dna.columns:
    y_true = dna[gene]
    y_est  = fill_dna[gene]
    
    ok = np.isnan(y_true.values)==False
    y_true=y_true[ok]
    y_est = y_est[ok]
    if np.sum(y_true.values)>0:
      aucs.append( roc_auc_score( y_true.values.astype(int), y_est.values ) )
    
      fpr,tpr,thresholds = roc_curve( y_true.values, y_est.values )
    
      gene_name = "%s auc=%0.3f"%(gene,aucs[-1])
      ax_auc_curves.plot( fpr, tpr, label = gene_name )
    
      plot_binary_classification_result( y_true.values, y_est.values, title = gene_name, dirname = summary_location_dir)

    if len(weights) > 0:
      plot_weights( mean_weights[gene], title = gene_name, dirname = summary_location_dir )
    
    
    
  #pdb.set_trace()
  aucs = pd.Series( aucs, fill_dna.columns ).sort_index( ascending=False )
  logliks = loglik_dna.mean(0)
  
  #pp.figure(f_auc_curves)
  if len(fill_dna.columns)<20:
    ax_auc_curves.legend(loc='lower right')
  ax_auc_curves.set_ylabel("True Positive Rate")
  ax_auc_curves.set_xlabel("False Positive Rate")
  
  auc_mean = aucs.values.mean()
  auc_median = np.median( aucs.values )
  f_auc_curves.savefig( summary_location_dir + "/auc_roc_curves_mean_%0.3f_median_%0.3f.svg"%(auc_mean,auc_median), transparent=True, bbox_inches = 'tight', pad_inches=0.15, dpi=300 )
  f_auc_curves.savefig( summary_location_dir + "/auc_roc_curves_mean_%0.3f_median_%0.3f.png"%(auc_mean,auc_median), transparent=False, bbox_inches = 'tight', pad_inches=0.15, dpi=300 )
  
  logliks = loglik_dna.mean(0)
  logliks.name = "loglik"
  aucs.name="auc"
  
  #f, ax = plt.subplots(figsize=(12, 9))

  # Draw the heatmap using seaborn
  #sns.heatmap(corrmat, vmax=.8, square=True)

  # Use matplotlib directly to emphasize known networks
  # networks = corrmat.columns.get_level_values("network")
  # for i, network in enumerate(networks):
  #     if i and network != networks[i - 1]:
  #         ax.axhline(len(networks) - i, c="w")
  #         ax.axvline(i, c="w")
  # f.tight_layout()
  
  #weight_summary = pd.DataFrame( pd.concat([mean_weights,std_weights], axis=1) ) )
  # fig_sns, ax_sns = pp.subplots()
  # g=sns.clustermap(mean_weights,square=False, yticklabels=mean_weights.index.values, xticklabels=mean_weights.columns,col_cluster=False)
  # pp.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7)
  # pp.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
  # fig_sns.tight_layout()
  # pp.show()
  # pdb.set_trace()
  # fig_sns.savefig( summary_location_dir + "/weights.png", transparent=True, bbox_inches = 'tight', pad_inches=0.15, dpi=300 )
  #
  #
  results = pd.DataFrame( pd.concat([aucs,logliks], axis=1) ) #, columns=["auc","loglik"])
  
  if len(weights) > 0:
    weights.to_csv( summary_location_dir + "/weights.csv" )
    mean_weights.to_csv( summary_location_dir + "/weights_means.csv" )
    std_weights.to_csv( summary_location_dir + "/weights_stds.csv" )
  results.to_csv( summary_location_dir + "/summary.csv" )
  fill_dna.to_csv( summary_location_dir + "/predictions.csv" )
  logliks.to_csv( summary_location_dir + "/logliks.csv" )
  dna.to_csv( summary_location_dir + "/dna.csv" )
  print results
  
  print "------------------------"
  print "mean auc   = ",auc_mean
  print "median auc = ",auc_median
  print "------------------------"
  print aucs
  print logliks
  #print data_store
  data_store.close()
  #dna_data = data_store[]
  assert len(np.intersect1d( fill_barcodes, loglik_barcodes)) == len(loglik_barcodes), "should be the same"
  if len(weights) > 0:
    return fill_dna, loglik_dna, dna, results, [weights,mean_weights,std_weights]
  else:
    return fill_dna, loglik_dna, dna, results, [None,None,None]
  
  
######################################################################################################
if __name__ == "__main__":
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_file = sys.argv[1]
  print "Running: ",yaml_file
  
  
    
  fill_dna, loglik_dna, dna,results,weights = main( yaml_file )

  
  
  