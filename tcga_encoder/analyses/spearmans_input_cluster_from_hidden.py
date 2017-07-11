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
  
  save_dir = os.path.join( results_path, "spearman" )
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
  
  print "next..."
  Z_train = fill_store["/Z/TRAIN/Z/mu"]
  Z_val = fill_store["/Z/VAL/Z/mu"]
  Z = pd.concat( [Z_train, Z_val] )
  barcodes = Z.index.values
  try:
    H = fill_store["hidden"].loc[barcodes]
  except:
    print "found no hidden"
    H = pd.DataFrame( [], index = barcodes )
  
  RNA_scale = fill_store["/scaled/RNA"].loc[barcodes]
  miRNA_scale = fill_store["/scaled/miRNA"].loc[barcodes]
  METH_scale = fill_store["/scaled/METH"].loc[barcodes]
  fill_store.close()
  
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
  
  print "computing RNA-H spearman rho's"
  rho_rna_h = stats.spearmanr( RNA_scale.values, H.values )
  print "computing miRNA-H spearman rho's"
  rho_mirna_h = stats.spearmanr( miRNA_scale.values, H.values )
  print "computing METH-H spearman rho's"
  rho_meth_h = stats.spearmanr( METH_scale.values, H.values )
  
  print "computing RNA-Z spearman rho's"
  rho_rna_z = stats.spearmanr( RNA_scale.values, Z.values )
  print "computing miRNA-Z spearman rho's"
  rho_mirna_z = stats.spearmanr( miRNA_scale.values, Z.values )
  print "computing METH-Z spearman rho's"
  rho_meth_z = stats.spearmanr( METH_scale.values, Z.values )
  
  rna_rna_rho = pd.DataFrame( rho_rna_h[0][:n_rna,:][:,:n_rna], index = rna_names, columns=rna_names)
  rna_rna_p   = pd.DataFrame( rho_rna_h[1][:n_rna,:][:,:n_rna], index = rna_names, columns=rna_names)
  
  rna_h_rho = pd.DataFrame( rho_rna_h[0][:n_rna,:][:,n_rna:], index = rna_names, columns=h_names)
  rna_h_p   = pd.DataFrame( rho_rna_h[1][:n_rna,:][:,n_rna:], index = rna_names, columns=h_names)

  rna_z_rho = pd.DataFrame( rho_rna_z[0][:n_rna,:][:,n_rna:], index = rna_names, columns=z_names)
  rna_z_p   = pd.DataFrame( rho_rna_z[1][:n_rna,:][:,n_rna:], index = rna_names, columns=z_names)
  
  mirna_mirna_rho = pd.DataFrame( rho_mirna_h[0][:n_mirna,:][:,:n_mirna], index = mirna_names, columns=mirna_names)
  mirna_mirna_p   = pd.DataFrame( rho_mirna_h[1][:n_mirna,:][:,:n_mirna], index = mirna_names, columns=mirna_names)
  
  mirna_h_rho = pd.DataFrame( rho_mirna_h[0][:n_mirna,:][:,n_mirna:], index = mirna_names, columns=h_names)
  mirna_h_p   = pd.DataFrame( rho_mirna_h[1][:n_mirna,:][:,n_mirna:], index = mirna_names, columns=h_names)

  mirna_z_rho = pd.DataFrame( rho_mirna_z[0][:n_mirna,:][:,n_mirna:], index = mirna_names, columns=z_names)
  mirna_z_p   = pd.DataFrame( rho_mirna_z[1][:n_mirna,:][:,n_mirna:], index = mirna_names, columns=z_names)
 
  meth_meth_rho = pd.DataFrame( rho_meth_h[0][:n_meth,:][:,:n_meth], index = meth_names, columns=meth_names)
  meth_meth_p   = pd.DataFrame( rho_meth_h[1][:n_meth,:][:,:n_meth], index = meth_names, columns=meth_names)
  
  meth_h_rho = pd.DataFrame( rho_meth_h[0][:n_meth,:][:,n_meth:], index = meth_names, columns=h_names)
  meth_h_p   = pd.DataFrame( rho_meth_h[1][:n_meth,:][:,n_meth:], index = meth_names, columns=h_names)
  
  meth_z_rho = pd.DataFrame( rho_meth_z[0][:n_meth,:][:,n_meth:], index = meth_names, columns=z_names)
  meth_z_p   = pd.DataFrame( rho_meth_z[1][:n_meth,:][:,n_meth:], index = meth_names, columns=z_names)

  rna_rna_rho.to_csv( save_dir + "/rna_rna_rho.csv" )
  rna_rna_p.to_csv( save_dir + "/rna_rna_p.csv" )
  rna_h_rho.to_csv( save_dir + "/rna_h_rho.csv" )
  rna_h_p.to_csv( save_dir + "/rna_h_p.csv" )
  rna_z_rho.to_csv( save_dir + "/rna_z_rho.csv" )
  rna_z_p.to_csv( save_dir + "/rna_z_p.csv" )
  
  mirna_mirna_rho.to_csv( save_dir + "/mirna_mirna_rho.csv" )
  mirna_mirna_p.to_csv( save_dir + "/mirna_mirna_p.csv" )
  mirna_h_rho.to_csv( save_dir + "/mirna_h_rho.csv" )
  mirna_h_p.to_csv( save_dir + "/mirna_h_p.csv" )
  mirna_z_rho.to_csv( save_dir + "/mirna_z_rho.csv" )
  mirna_z_p.to_csv( save_dir + "/mirna_z_p.csv" )
  

  meth_meth_rho.to_csv( save_dir + "/meth_meth_rho.csv" )
  meth_meth_p.to_csv( save_dir + "/meth_meth_p.csv" )
  meth_h_rho.to_csv( save_dir + "/meth_h_rho.csv" )
  meth_h_p.to_csv( save_dir + "/meth_h_p.csv" )
  meth_z_rho.to_csv( save_dir + "/meth_z_rho.csv" )
  meth_z_p.to_csv( save_dir + "/meth_z_p.csv" )
  
  
  
  #Z = np.vstack( (Z_train.values, Z_val.values) )
  #pdb.set_trace()
  n_z = Z.shape[1]

  # size_per_unit = 0.25
  # size1 = max( min( 40, int( W_hidden["RNA"].values.shape[0]*size_per_unit ) ), 12 )
  #
  # size2 = max( min( 40, int( W_hidden["miRNA"].values.shape[0]*size_per_unit )), 12 )
  #

  tissues = data_store["/CLINICAL/TISSUE"].loc[barcodes]
  
  rna   = RNA_scale #np.log(1+data_store["/RNA/RSEM"].loc[ barcodes ])
  mirna = miRNA_scale #np.log(1+data_store["/miRNA/RSEM"].loc[ barcodes ])
  meth  = METH_scale #np.log(0.1+data_store["/METH/METH"].loc[ barcodes ])
  
  dna   = data_store["/DNA/channel/0"].loc[ barcodes ]
  
  
  tissue_names = tissues.columns
  tissue_idx = np.argmax( tissues.values, 1 )
  n = len(Z)
  n_tissues = len(tissue_names)
  
  rna_rna_neighbourhoods = np.dot( (1.0-rna_h_p), (1.0-rna_h_p).T )
  rna_rna_neighbourhoods = np.dot( rna_h_rho, rna_h_rho.T )
  pdb.set_trace()

    
  # nbr = 20
  # for z_idx,z_name in zip( range(n_z), z_names):
  #   print "viewing ", z_name
  #   rna_p   = rna_z_p[ z_name ].sort_values()[:nbr]
  #   mirna_p = mirna_z_p[ z_name ].sort_values()[:nbr]
  #   meth_p  = meth_z_p[ z_name ].sort_values()[:nbr]
  #
  #   rna_rho = rna_z_rho[ z_name ].loc[ rna_p.index ].sort_values()
  #   mirna_rho = mirna_z_rho[ z_name ].loc[ mirna_p.index ].sort_values()
  #   meth_rho = meth_z_rho[ z_name ].loc[ meth_p.index ].sort_values()
  #
  #   f = pp.figure( figsize = (12,8))
  #   ax_rna = f.add_subplot(131); ax_mirna = f.add_subplot(133); ax_meth = f.add_subplot(132);
  #   #ax_pie1 = f.add_subplot(133); #ax_pie3 = f.add_subplot(424); ax_pie4 = f.add_subplot(426)
  #
  #   max_ax = np.max( np.hstack( (rna_rho.values,mirna_rho.values,meth_rho.values) ) )
  #   min_ax = np.min( np.hstack( (rna_rho.values,mirna_rho.values,meth_rho.values) ) )
  #
  #   h1=rna_rho.plot(kind='barh',ax=ax_rna,color="red",legend=False,title="RNA",fontsize=8); ax_rna.set_xlim(min_ax,max_ax);
  #   h2=meth_rho.plot(kind='barh',ax=ax_meth,color="blue",legend=False,title="METH",fontsize=8);ax_meth.set_xlim(min_ax,max_ax);
  #   h3=mirna_rho.plot(kind='barh',ax=ax_mirna,color="black",legend=False,title="miRNA",fontsize=8); ax_mirna.set_xlim(min_ax,max_ax);
  #
  #   pp.suptitle( z_name)
  #   pp.savefig( z_dir + "/%s_spearmanr.png"%(z_name), format="png", dpi=300 )
  #   pp.close('all')
  #
  # for h_idx, h_name in zip( range(n_h), h_names ):
  #   print "viewing ", h_name
  #   rna_p   = rna_h_p[ h_name ].sort_values()[:nbr]
  #   mirna_p = mirna_h_p[ h_name ].sort_values()[:nbr]
  #   meth_p  = meth_h_p[ h_name ].sort_values()[:nbr]
  #
  #   rna_rho = rna_h_rho[ h_name ].loc[ rna_p.index ].sort_values()
  #   mirna_rho = mirna_h_rho[ h_name ].loc[ mirna_p.index ].sort_values()
  #   meth_rho = meth_h_rho[ h_name ].loc[ meth_p.index ].sort_values()
  #
  #   f = pp.figure( figsize = (12,8))
  #   ax_rna = f.add_subplot(131); ax_mirna = f.add_subplot(133); ax_meth = f.add_subplot(132);
  #
  #   max_ax = np.max( np.hstack( (rna_rho.values,mirna_rho.values,meth_rho.values) ) )
  #   min_ax = np.min( np.hstack( (rna_rho.values,mirna_rho.values,meth_rho.values) ) )
  #
  #   h1=rna_rho.plot(kind='barh',ax=ax_rna,color="red",legend=False,title="RNA",fontsize=8); ax_rna.set_xlim(min_ax,max_ax);
  #   h2=meth_rho.plot(kind='barh',ax=ax_meth,color="blue",legend=False,title="METH",fontsize=8);ax_meth.set_xlim(min_ax,max_ax);
  #   h3=mirna_rho.plot(kind='barh',ax=ax_mirna,color="black",legend=False,title="miRNA",fontsize=8); ax_mirna.set_xlim(min_ax,max_ax);
  #
  #   pp.suptitle( h_name )
  #   pp.savefig( h_dir + "/%s_spearmanr.png"%(h_name), format="png", dpi=300 )
  #   pp.close('all')

        

  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )