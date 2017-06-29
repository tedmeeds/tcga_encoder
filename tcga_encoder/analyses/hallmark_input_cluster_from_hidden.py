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
  
  save_dir = os.path.join( results_path, "hallmark_clustering" )
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
  size1 = max( int( W_hidden["RNA"].values.shape[0]*size_per_unit ), 12 )
  #size2 = max( int( n_inputs*size_per_unit ), 12 )
  cmap = sns.palplot(sns.light_palette((260, 75, 60), input="husl"))
  htmap3 = sns.clustermap ( W_hidden["RNA"].T.corr(), cmap=cmap, square=True, figsize=(size1,size1) )
  pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  htmap3.ax_row_dendrogram.set_visible(False)
  htmap3.ax_col_dendrogram.set_visible(False)
  pp.savefig( save_dir + "/weights_rna_clustermap.png", fmt="png", bbox_inches = "tight")

  htmap3 = sns.clustermap ( W_hidden["miRNA"].T.corr(), cmap=cmap, square=True, figsize=(size1,size1) )
  pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  htmap3.ax_row_dendrogram.set_visible(False)
  htmap3.ax_col_dendrogram.set_visible(False)
  pp.savefig( save_dir + "/weights_mirna_clustermap.png", fmt="png", bbox_inches = "tight")

  htmap3 = sns.clustermap ( W_hidden["METH"].T.corr(), cmap=cmap, square=True, figsize=(size1,size1) )
  pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
  pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
  htmap3.ax_row_dendrogram.set_visible(False)
  htmap3.ax_col_dendrogram.set_visible(False)
  pp.savefig( save_dir + "/weights_meth_clustermap.png", fmt="png", bbox_inches = "tight")

  
  #pdb.set_trace()
  weighted_z = join_weights( W_hidden2z, W_hidden )
  
  #pdb.set_trace()
  Z = np.vstack( (Z_train.values, Z_val.values) )
  n_z = Z.shape[1]
  #pdb.set_trace()
  z_names = ["z_%d"%z_idx for z_idx in range(Z.shape[1])]
  Z = pd.DataFrame( Z, index = np.hstack( (Z_train.index.values, Z_val.index.values)), columns = z_names )
  
  barcodes = np.union1d( Z_train.index.values, Z_val.index.values )
  barcodes = data_store["/CLINICAL/observed"][ data_store["/CLINICAL/observed"][["RNA","miRNA","METH","DNA"]].sum(1)==4 ].index.values
  
  Z=Z.loc[barcodes]
  Z_values = Z.values
  tissues = data_store["/CLINICAL/TISSUE"].loc[barcodes]
  
  rna   = np.log(1+data_store["/RNA/RSEM"].loc[ barcodes ])
  mirna = np.log(1+data_store["/miRNA/RSEM"].loc[ barcodes ])
  meth  = np.log(0.1+data_store["/METH/METH"].loc[ barcodes ])
  dna   = data_store["/DNA/channel/0"].loc[ barcodes ]
  
  
  tissue_names = tissues.columns
  tissue_idx = np.argmax( tissues.values, 1 )
  n = len(Z)
  n_tissues = len(tissue_names)
  
  n_h = W_hidden2z.shape[0]
  
  rna_normed = rna; mirna_normed = mirna; meth_normed = meth; dna_normed=2*dna-1
  for t_idx in range(n_tissues):
    t_query = tissue_idx == t_idx
    
    X = rna[t_query]
    X -= X.mean(0)
    X /= X.std(0)
    rna_normed[t_query] = X

    X = mirna[t_query]
    X -= X.mean(0)
    X /= X.std(0)
    mirna_normed[t_query] = X

    X = meth[t_query]
    X -= X.mean(0)
    X /= X.std(0)
    meth_normed[t_query] = X
  
  pdb.set_trace()
    
  nbr = 20
  Z_keep_rna=[]
  Z_keep_mirna=[]
  Z_keep_meth=[]
  Z_keep_dna = []
  for z_idx in range(n_z):
    z_values = Z_values[:,z_idx]
    order_z = np.argsort(z_values)

    rna_w = weighted_z["RNA"][ "z_%d"%(z_idx)]
    mirna_w = weighted_z["miRNA"][ "z_%d"%(z_idx)]
    meth_w = weighted_z["METH"][ "z_%d"%(z_idx)]
    
    order_rna = np.argsort( -np.abs(rna_w.values) )
    order_mirna = np.argsort( -np.abs(mirna_w.values) )
    order_meth = np.argsort( -np.abs(meth_w.values) )
    
    rna_w_ordered = pd.Series( rna_w.values[ order_rna ], index = rna_w.index[order_rna], name="RNA")
    mirna_w_ordered = pd.Series( mirna_w.values[ order_mirna ], index = mirna_w.index[order_mirna], name="miRNA")
    meth_w_ordered = pd.Series( meth_w.values[ order_meth ], index = meth_w.index[order_meth], name="METH")
    
   
    f = pp.figure( figsize = (12,8))
    ax1 = f.add_subplot(321);ax2 = f.add_subplot(323);ax3 = f.add_subplot(325);
    ax_pie1 = f.add_subplot(133); #ax_pie3 = f.add_subplot(424); ax_pie4 = f.add_subplot(426)
    
    max_ax = np.max( np.hstack( (rna_w_ordered[:nbr].values,meth_w_ordered[:nbr].values,mirna_w_ordered[:nbr].values) ) )
    min_ax = np.min( np.hstack( (rna_w_ordered[:nbr].values,meth_w_ordered[:nbr].values,mirna_w_ordered[:nbr].values) ) )
    
    h1=rna_w_ordered[:nbr].sort_values(ascending=False).plot(kind='barh',ax=ax1,color="red",legend=False,title=None,fontsize=8); ax1.set_xlim(min_ax,max_ax); ax1.set_title(""); h1.set_xticklabels([]); ax1.legend(["RNA"])
    h2=meth_w_ordered[:nbr].sort_values(ascending=False).plot(kind='barh',ax=ax2,color="blue",legend=False,title=None,fontsize=8);ax2.set_xlim(min_ax,max_ax); ax2.set_title(""); h2.set_xticklabels([]); ax2.legend(["METH"])
    h3=mirna_w_ordered[:nbr].sort_values(ascending=False).plot(kind='barh',ax=ax3,color="black",legend=False,title=None,fontsize=8); ax3.set_xlim(min_ax,max_ax); ax3.set_title("");ax3.legend(["miRNA"])
    
    
    neg_rna = pp.find( rna_w_ordered.values<0) ; pos_rna = pp.find( rna_w_ordered.values>0)
    neg_meth = pp.find( meth_w_ordered.values<0) ; pos_meth = pp.find( meth_w_ordered.values>0) 

    rna_readable = pathway_info.CancerEnrichment(rna_w_ordered[:nbr].index, 1+0*np.abs( rna_w_ordered[:nbr].values)  )
    meth_readable = pathway_info.CancerEnrichment(meth_w_ordered[:nbr].index, 1+0*np.abs( meth_w_ordered[:nbr].values ) )
    
    
    # rna_readable_p   = pathway_info.CancerEnrichment(rna_w_ordered.index[pos_rna[:20]], 1+0*rna_w_ordered.values[pos_rna[:20]] )
    # meth_readable_p = pathway_info.CancerEnrichment(meth_w_ordered.index[pos_meth[:20]], 1+0*meth_w_ordered.values[pos_meth[:20]])
    # #
    # rna_readable_n   = pathway_info.CancerEnrichment(rna_w_ordered.index[neg_rna[:20]], -1+0*rna_w_ordered.values[neg_rna[:20]] )
    # meth_readable_n = pathway_info.CancerEnrichment(meth_w_ordered.index[neg_meth[:20]], -1+0*meth_w_ordered.values[neg_meth[:20]] )

    rna_readable.name="rna"
    meth_readable.name="meth"
    
    # rna_readable_p.name="rna_p"
    # meth_readable_p.name="meth_p"
    # rna_readable_n.name="rna_n"
    # meth_readable_n.name="meth_n"
                         
    #joined = pd.concat( [rna_readable[:20],\
    #                     meth_readable[:20]], axis=1 )
                         
    joined = pd.concat( [rna_readable,\
                         meth_readable], axis=1 )
                         
    # joined = pd.concat( [rna_readable_p,rna_readable_n,\
    #                      meth_readable_p,meth_readable_n], axis=1 )
    #
    # maxvalues = joined.index[ np.argsort( -np.abs(joined.fillna(0)).sum(1).values ) ]
    #
    # joined=joined.loc[maxvalues]
    # joined = joined[:25]

    #br = joined.plot(kind="barh",ax=ax_pie1,color=["red","red","blue","blue"],legend=False,stacked=True, sort_columns=False,fontsize=8); 
    br = joined.plot(kind="barh",ax=ax_pie1,color=["red","blue"],legend=False,stacked=True, sort_columns=False,fontsize=8); 
    
    max_ax = np.max( joined.values.flatten() )
    min_ax = np.min( joined.values.flatten() )
    max_ax = np.max( max_ax, -min_ax )
    min_ax = -max_ax
    #pdb.set_trace()
    #ax_pie1.set_xlim(min_ax,max_ax);
    #br = joined.plot(kind="barh",ax=ax_pie1,color=["red","blue"],legend=True,stacked=True, sort_columns=False); 
    pp.suptitle( "Z %d"%(z_idx))
    pp.savefig( z_dir + "/z%d_weighted.png"%(z_idx), format="png", dpi=300 )
    #pp.show()
    #pdb.set_trace()
    pp.close('all')

  for z_idx in range(n_h):
    #z_values = Z_values[:,z_idx]
    #order_z = np.argsort(z_values)

    rna_w = W_hidden["RNA"][ "h_%d"%(z_idx)]
    mirna_w = W_hidden["miRNA"][ "h_%d"%(z_idx)]
    meth_w = W_hidden["METH"][ "h_%d"%(z_idx)]
    
    order_rna = np.argsort( -np.abs(rna_w.values) )
    order_mirna = np.argsort( -np.abs(mirna_w.values) )
    order_meth = np.argsort( -np.abs(meth_w.values) )
    
    rna_w_ordered = pd.Series( rna_w.values[ order_rna ], index = rna_w.index[order_rna], name="RNA")
    mirna_w_ordered = pd.Series( mirna_w.values[ order_mirna ], index = mirna_w.index[order_mirna], name="miRNA")
    meth_w_ordered = pd.Series( meth_w.values[ order_meth ], index = meth_w.index[order_meth], name="METH")
    
   
    f = pp.figure( figsize = (12,8))
    ax1 = f.add_subplot(321);ax2 = f.add_subplot(323);ax3 = f.add_subplot(325);
    ax_pie1 = f.add_subplot(133); #ax_pie3 = f.add_subplot(424); ax_pie4 = f.add_subplot(426)
    
    max_ax = np.max( np.hstack( (rna_w_ordered[:nbr].values,meth_w_ordered[:nbr].values,mirna_w_ordered[:nbr].values) ) )
    min_ax = np.min( np.hstack( (rna_w_ordered[:nbr].values,meth_w_ordered[:nbr].values,mirna_w_ordered[:nbr].values) ) )
    
    h1=rna_w_ordered[:nbr].sort_values(ascending=False).plot(kind='barh',ax=ax1,color="red",legend=False,title=None,fontsize=8); ax1.set_xlim(min_ax,max_ax); ax1.set_title(""); h1.set_xticklabels([]); ax1.legend(["RNA"])
    h2=meth_w_ordered[:nbr].sort_values(ascending=False).plot(kind='barh',ax=ax2,color="blue",legend=False,title=None,fontsize=8);ax2.set_xlim(min_ax,max_ax); ax2.set_title(""); h2.set_xticklabels([]); ax2.legend(["METH"])
    h3=mirna_w_ordered[:nbr].sort_values(ascending=False).plot(kind='barh',ax=ax3,color="black",legend=False,title=None,fontsize=8); ax3.set_xlim(min_ax,max_ax); ax3.set_title("");ax3.legend(["miRNA"])
    
    
    neg_rna = pp.find( rna_w_ordered.values<0) ; pos_rna = pp.find( rna_w_ordered.values>0)
    neg_meth = pp.find( meth_w_ordered.values<0) ; pos_meth = pp.find( meth_w_ordered.values>0) 

    
    
    rna_readable = pathway_info.CancerEnrichment(rna_w_ordered[:nbr].index, 1+0*np.abs( rna_w_ordered[:nbr].values)  )
    meth_readable = pathway_info.CancerEnrichment(meth_w_ordered[:nbr].index, 1+0*np.abs( meth_w_ordered[:nbr].values ) )
    
    
    # rna_readable_p   = pathway_info.CancerEnrichment(rna_w_ordered.index[pos_rna[:20]], 1+0*rna_w_ordered.values[pos_rna[:20]] )
    # meth_readable_p = pathway_info.CancerEnrichment(meth_w_ordered.index[pos_meth[:20]], 1+0*meth_w_ordered.values[pos_meth[:20]])
    # #
    # rna_readable_n   = pathway_info.CancerEnrichment(rna_w_ordered.index[neg_rna[:20]], -1+0*rna_w_ordered.values[neg_rna[:20]] )
    # meth_readable_n = pathway_info.CancerEnrichment(meth_w_ordered.index[neg_meth[:20]], -1+0*meth_w_ordered.values[neg_meth[:20]] )

    rna_readable.name="rna"
    meth_readable.name="meth"
    
    # rna_readable_p.name="rna_p"
    # meth_readable_p.name="meth_p"
    # rna_readable_n.name="rna_n"
    # meth_readable_n.name="meth_n"
                         
    #joined = pd.concat( [rna_readable[:20],\
    #                     meth_readable[:20]], axis=1 )
                         
    joined = pd.concat( [rna_readable,\
                         meth_readable], axis=1 )
                         
    # joined = pd.concat( [rna_readable_p,rna_readable_n,\
    #                      meth_readable_p,meth_readable_n], axis=1 )
    #
    # maxvalues = joined.index[ np.argsort( -np.abs(joined.fillna(0)).sum(1).values ) ]
    #
    # joined=joined.loc[maxvalues]
    # joined = joined[:25]

    #br = joined.plot(kind="barh",ax=ax_pie1,color=["red","red","blue","blue"],legend=False,stacked=True, sort_columns=False,fontsize=8); 
    br = joined.plot(kind="barh",ax=ax_pie1,color=["red","blue"],legend=False,stacked=True, sort_columns=False,fontsize=8); 
    max_ax = np.max( joined.values.flatten() )
    min_ax = np.min( joined.values.flatten() )
    max_ax = np.max( max_ax, -min_ax )
    min_ax = -max_ax
    #br = joined.plot(kind="barh",ax=ax_pie1,color=["red","blue"],legend=True,stacked=True, sort_columns=False); 
    pp.suptitle( "H %d"%(z_idx))
    pp.savefig( h_dir + "/h%d_weighted.png"%(z_idx), format="png", dpi=300 )
    #pp.show()
    #pdb.set_trace()
    pp.close('all')
        

  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )