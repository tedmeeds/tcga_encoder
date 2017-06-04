from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.data.pathway_data import Pathways
from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
#from tcga_encoder.algorithms import *
import seaborn as sns
from sklearn.manifold import TSNE, locally_linear_embedding
from scipy import stats

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
  
  save_dir = os.path.join( results_path, "input_clustering" )
  check_and_mkdir(save_dir)
  z_dir = os.path.join( save_dir, "z_pics" )
  check_and_mkdir(z_dir)
  print "HOME_DIR: ", HOME_DIR
  print "data_filename: ", data_filename
  print "fill_filename: ", fill_filename
  
  print "LOADING stores"
  data_store = pd.HDFStore( data_filename, "r" )
  fill_store = pd.HDFStore( fill_filename, "r" )
  
  Z_train = fill_store["/Z/TRAIN/Z/mu"]
  Z_val = fill_store["/Z/VAL/Z/mu"]
  
  
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
  mirna = np.log(1+data_store["/miRNA/READS"].loc[ barcodes ])
  meth  = np.log(0.1+data_store["/METH/METH"].loc[ barcodes ])
  dna   = data_store["/DNA/channel/0"].loc[ barcodes ]
  
  
  tissue_names = tissues.columns
  tissue_idx = np.argmax( tissues.values, 1 )
  n = len(Z)
  n_tissues = len(tissue_names)
  
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
    
  nbr = 15
  Z_keep_rna=[]
  Z_keep_mirna=[]
  Z_keep_meth=[]
  Z_keep_dna = []
  for z_idx in range(n_z):
    z_values = Z_values[:,z_idx]
    order_z = np.argsort(z_values)
    rna_sorted = pd.DataFrame( rna_normed.values[order_z,:], index = barcodes[order_z], columns = rna.columns )
    mirna_sorted = pd.DataFrame( mirna_normed.values[order_z,:], index = barcodes[order_z], columns = mirna.columns )
    meth_sorted = pd.DataFrame( meth_normed.values[order_z,:], index = barcodes[order_z], columns = meth.columns )
    dna_sorted = pd.DataFrame( dna_normed.values[order_z,:], index = barcodes[order_z], columns = dna.columns )
    #
    # inner_rna = pd.Series( np.dot( z_values, rna_normed ), index = rna_normed.columns, name="rna" )
    # inner_rna.sort_values(inplace=True)
    # inner_rna = inner_rna / np.max(np.abs(inner_rna))
    # sign_rna = np.sign( inner_rna )
    # abs_rna = np.abs( inner_rna )
    # ordered_rna = np.argsort( -abs_rna.values )
    # ordered_rna = pd.Series( inner_rna.values[ordered_rna], index =inner_rna.index[ordered_rna],name="rna" )
    #
    # keep_rna = ordered_rna[:nbr]
    # keep_rna = keep_rna.sort_values()
    keep_rna,keep_mirna,keep_meth,keep_dna = find_keepers_over_groups( z_values, [rna_normed,mirna_normed,meth_normed,dna_normed], "z_%d"%(z_idx), nbr, stats2use=["pearson","pearson","pearson","auc"])
    
    # keep_rna = find_keepers( z_values, rna_normed, "z_%d"%(z_idx), nbr )
    # keep_mirna = find_keepers( z_values, mirna_normed, "z_%d"%(z_idx), nbr )
    # keep_meth = find_keepers( z_values, meth_normed, "z_%d"%(z_idx), nbr )
    
    keep_rna_big,keep_mirna_big,keep_meth_big,keep_dna_big = find_keepers_over_groups( z_values, [rna_normed,mirna_normed,meth_normed,dna_normed], "z_%d"%(z_idx), 2*nbr, stats2use=["pearson","pearson","pearson","auc"])
    
    # keep_rna_big = find_keepers( z_values, rna_normed, "z_%d"%(z_idx), 3*nbr )
    # keep_mirna_big = find_keepers( z_values, mirna_normed, "z_%d"%(z_idx), 3*nbr )
    # keep_meth_big = find_keepers( z_values, meth_normed, "z_%d"%(z_idx), 3*nbr )
    
    Z_keep_rna.append( keep_rna )
    Z_keep_mirna.append( keep_mirna )
    Z_keep_meth.append( keep_meth )
    Z_keep_dna.append( keep_dna )
    
    f = pp.figure( figsize = (12,8))
    ax1 = f.add_subplot(421);ax2 = f.add_subplot(423);ax3 = f.add_subplot(425);ax4 = f.add_subplot(427)
    #ax_pie1 = f.add_subplot(422); ax_pie3 = f.add_subplot(424); ax_pie4 = f.add_subplot(426)
    ax_pie1 = f.add_subplot(222); #ax_pie3 = f.add_subplot(424); ax_pie4 = f.add_subplot(426)
    
    h1=keep_rna[["r"]].plot(kind='barh',ax=ax1,color="red",legend=False,title=None,fontsize=8); h1.set_xlim(-0.5,0.5); ax1.set_title(""); h1.set_xticklabels([]); ax1.legend(["RNA"])
    h2=keep_mirna[["r"]].plot(kind='barh',ax=ax4,color="black",legend=False,title=None,fontsize=8);h2.set_xlim(-0.5,0.5);ax4.set_title(""); ax4.legend(["miRNA"])
    h3=keep_meth[["r"]].plot(kind='barh',ax=ax3,color="blue",legend=False,title=None,fontsize=8);h3.set_xlim(-0.5,0.5);ax3.set_title(""); h3.set_xticklabels([]); ax3.legend(["METH"])
    h4=keep_dna[["r"]].plot(kind='barh',ax=ax2,color="green",legend=False,title=None,fontsize=8);h4.set_xlim(-0.5,0.5);ax2.set_title(""); h4.set_xticklabels([]); ax2.legend(["DNA"])
    
    neg_dna = pp.find( keep_dna_big.values[:,0]<0) ; pos_dna = pp.find( keep_dna_big.values[:,0]>0)
    neg_rna = pp.find( keep_rna_big.values[:,0]<0) ; pos_rna = pp.find( keep_rna_big.values[:,0]>0)
    neg_meth = pp.find( keep_meth_big.values[:,0]<0) ; pos_meth = pp.find( keep_meth_big.values[:,0]>0) 
    
    #dna_kegg,dna_readable = pathway_info.CancerEnrichment(keep_dna_big.index, np.abs( np.log2(keep_dna_big.values[:,1]) ) )
    #rna_kegg,rna_readable = pathway_info.CancerEnrichment(keep_rna_big.index, np.abs( np.log2(keep_rna_big.values[:,1]) ) )
    #meth_kegg,meth_readable = pathway_info.CancerEnrichment(keep_meth_big.index, np.abs( np.log2(keep_meth_big.values[:,1]) ) )
    
    dna_kegg,dna_readable = pathway_info.CancerEnrichment(keep_dna_big.index, np.abs(keep_dna_big.values[:,0])  )
    rna_kegg,rna_readable = pathway_info.CancerEnrichment(keep_rna_big.index, np.abs( keep_rna_big.values[:,0])  )
    meth_kegg,meth_readable = pathway_info.CancerEnrichment(keep_meth_big.index, np.abs( keep_meth_big.values[:,0] ) )
    
    # dna_kegg_p,dna_readable_p   = pathway_info.CancerEnrichment(keep_dna_big.index[pos_dna], (np.abs( np.log2(keep_dna_big.values[pos_dna,1]) )>-np.log2(0.01)).astype(float) )
    # rna_kegg_p,rna_readable_p   = pathway_info.CancerEnrichment(keep_rna_big.index[pos_rna], (np.abs( np.log2(keep_rna_big.values[pos_rna,1]) )>-np.log2(0.01)).astype(float) )
    # meth_kegg_p,meth_readable_p = pathway_info.CancerEnrichment(keep_meth_big.index[pos_meth], (np.abs( np.log2(keep_meth_big.values[pos_meth,1]) )>-np.log2(0.01)).astype(float) )
    #
    # dna_kegg_n,dna_readable_n   = pathway_info.CancerEnrichment(keep_dna_big.index[neg_dna], (np.abs( np.log2(keep_dna_big.values[neg_dna,1]) )>-np.log2(0.01)).astype(float) )
    # rna_kegg_n,rna_readable_n   = pathway_info.CancerEnrichment(keep_rna_big.index[neg_rna], (np.abs( np.log2(keep_rna_big.values[neg_rna,1]) )>-np.log2(0.01)).astype(float) )
    # meth_kegg_n,meth_readable_n = pathway_info.CancerEnrichment(keep_meth_big.index[neg_meth], (np.abs( np.log2(keep_meth_big.values[neg_meth,1]) )>-np.log2(0.01)).astype(float) )

    # dna_kegg_p,dna_readable_p   = pathway_info.CancerEnrichment(keep_dna_big.index[pos_dna], 1.0-keep_dna_big.values[pos_dna,1] )
    # rna_kegg_p,rna_readable_p   = pathway_info.CancerEnrichment(keep_rna_big.index[pos_rna], 1.0-keep_rna_big.values[pos_rna,1] )
    # meth_kegg_p,meth_readable_p = pathway_info.CancerEnrichment(keep_meth_big.index[pos_meth], 1.0-keep_meth_big.values[pos_meth,1])
    #
    # dna_kegg_n,dna_readable_n   = pathway_info.CancerEnrichment(keep_dna_big.index[neg_dna], 1.0-keep_dna_big.values[neg_dna,1] )
    # rna_kegg_n,rna_readable_n   = pathway_info.CancerEnrichment(keep_rna_big.index[neg_rna], 1.0-keep_rna_big.values[neg_rna,1] )
    # meth_kegg_n,meth_readable_n = pathway_info.CancerEnrichment(keep_meth_big.index[neg_meth], 1.0-keep_meth_big.values[neg_meth,1] )


    dna_kegg_p,dna_readable_p   = pathway_info.CancerEnrichment(keep_dna_big.index[pos_dna], np.abs( keep_dna_big.values[pos_dna,0] ) )
    rna_kegg_p,rna_readable_p   = pathway_info.CancerEnrichment(keep_rna_big.index[pos_rna], np.abs( keep_rna_big.values[pos_rna,0]) ) 
    meth_kegg_p,meth_readable_p = pathway_info.CancerEnrichment(keep_meth_big.index[pos_meth], np.abs( keep_meth_big.values[pos_meth,0]))

    dna_kegg_n,dna_readable_n   = pathway_info.CancerEnrichment(keep_dna_big.index[neg_dna], np.abs( keep_dna_big.values[neg_dna,0] ) )
    rna_kegg_n,rna_readable_n   = pathway_info.CancerEnrichment(keep_rna_big.index[neg_rna], np.abs( keep_rna_big.values[neg_rna,0] ) )
    meth_kegg_n,meth_readable_n = pathway_info.CancerEnrichment(keep_meth_big.index[neg_meth], np.abs( keep_meth_big.values[neg_meth,0]) )

    
    # dna_readable_n=-dna_readable_n
    # rna_readable_n=-rna_readable_n
    # meth_readable_n=-meth_readable_n
    rna_readable.name="rna"
    meth_readable.name="meth"
    dna_readable.name="dna"    
    
    rna_readable_p.name="rna_p"
    meth_readable_p.name="meth_p"
    dna_readable_p.name="dna_p"
    rna_readable_n.name="rna_n"
    meth_readable_n.name="meth_n"
    dna_readable_n.name="dna_n"
    # joined = pd.concat( [rna_readable_p[:20],rna_readable_n[:20],\
    #                      dna_readable_p[:20],dna_readable_n[:20],\
    #                      meth_readable_n[:20],meth_readable_p[:20]], axis=1 )
                         
    joined = pd.concat( [rna_readable[:20],\
                         dna_readable[:20],\
                         meth_readable[:20]], axis=1 )
    
    maxvalues = joined.index[ np.argsort( -np.abs(joined.fillna(0)).sum(1).values ) ]
    joined=joined.loc[maxvalues]
    joined = joined[:20]
    
    pathways = joined.index.values
    pathways = pathways[ np.argsort(pathways)]
    joined=joined.loc[pathways]
    #br = joined[["rna_p","rna_n"]].plot(kind="bar",ax=ax_pie1,color=["blue","red"],legend=False,stacked=True); br.set_xticklabels([]); ax_pie1.set_ylabel("RNA")
    #br = joined[["meth_p","meth_n"]].plot(kind="bar",ax=ax_pie4,color=["blue","red"],legend=False,stacked=True);  ax_pie4.set_ylabel("METH")
    #br = joined[["dna_p","dna_n"]].plot(kind="bar",ax=ax_pie3,color=["blue","red"],legend=False,stacked=True); br.set_xticklabels([]);  ax_pie3.set_ylabel("DNA")
    #pdb.set_trace()
    
    br = joined.plot(kind="bar",ax=ax_pie1,color=["red","green","blue"],legend=True,stacked=True, sort_columns=False); # ax_pie1.legend(["RNA","DNA","METH"])
    #br = joined[["meth_p","meth_n"]].plot(kind="bar",ax=ax_pie4,color=["blue","red"],legend=False,stacked=True);  ax_pie4.set_ylabel("METH")
    #br = joined[["dna_p","dna_n"]].plot(kind="bar",ax=ax_pie3,color=["blue","red"],legend=False,stacked=True); br.set_xticklabels([]);  ax_pie3.set_ylabel("DNA")
    
    #joined[["rna_n","meth_n","dna_n"]].plot(kind="bar",ax=ax_pie1,color="red")
    # if len(rna_readable_p)>0:
    #   rna_readable_p[:12].plot( kind="barh",ax=ax_pie1, fontsize=8, color="green" )
    # if len(rna_readable_n)>0:
    #   rna_readable_n[:12].plot( kind="barh",ax=ax_pie1, fontsize=8, color="red" )
    #
    # if len(meth_readable_p)>0:
    #   meth_readable_p[:12].plot( kind="barh",ax=ax_pie1, fontsize=8, color="blue" )
    # if len(meth_readable_n)>0:
    #   meth_readable_n[:12].plot( kind="barh",ax=ax_pie1, fontsize=8, color="purple" )
    #
    # if len(dna_readable_p)>0:
    #   dna_readable_p[:12].plot( kind="barh",ax=ax_pie1, fontsize=8, color="yellow" )
    # if len(dna_readable_n)>0:
    #   dna_readable_n[:12].plot( kind="barh",ax=ax_pie1, fontsize=8, color="black" )
    #
    #   #rna_readable[:12].plot.pie( ax=ax_pie1, fontsize=8 )
    # if len(meth_readable)>0:
    #   #meth_readable[:12].plot.pie( ax=ax_pie3, fontsize =8 )
    # if len(dna_readable)>0:
    #   3dna_readable[:12].plot.pie( ax=ax_pie4, fontsize=8 )
    #pp.show()
    #pdb.set_trace()
    #assert False
    #print "normaize over meth and rna anr mirna"
    #print "include dna"
    #print "up and down pie charts"
    #print "add other pathways if not in cancer"
    #print "put counts in pies"
    print "survival: best per cohort, also double sided on third, go to fifth if enough events"
    
    #pp.show()
    #pdb.set_trace()
    
    #f.suptitle( "z %d"%(z_idx) ); 
    #f.subplots_adjust(bottom=0.25);
    pp.savefig( z_dir + "/z%d.png"%(z_idx), format="png", dpi=300 )
    #print h
    pp.close('all')
    #pdb.set_trace()
    #kept_rna = pd.DataFrame( rna_sorted[keep_rna.index], index=rna_sorted.index, columns = keep_rna.index )
    #kept_mirna = pd.DataFrame( mirna_sorted[keep_mirna.index], index=mirna_sorted.index, columns = keep_mirna.index )
    #kept_meth = pd.DataFrame( meth_sorted[keep_meth.index], index=meth_sorted.index, columns = keep_meth.index )
  merged_rna   = pd.concat(Z_keep_rna,axis=1) 
  merged_mirna = pd.concat(Z_keep_mirna,axis=1) 
  merged_meth  = pd.concat(Z_keep_meth,axis=1)  
  
  merged_rna.to_csv( save_dir + "/z_to_rna.csv" )
  merged_mirna.to_csv( save_dir + "/z_to_mirna.csv" )
  merged_meth.to_csv( save_dir + "/z_to_meth.csv" )
  
  f = sns.clustermap(merged_rna.fillna(0), figsize=(8,6))
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), fontsize=8)
  pp.savefig( save_dir + "/clustermap_z_to_rna.png", format="png", dpi=300 )
  f = sns.clustermap(merged_mirna.fillna(0), figsize=(8,6))
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), fontsize=8)
  pp.savefig( save_dir + "/clustermap_z_to_mirna.png", format="png", dpi=300 )
  
  f = sns.clustermap(merged_meth.fillna(0), figsize=(8,6))
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
  pp.setp(f.ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
  pp.setp(f.ax_heatmap.xaxis.get_majorticklabels(), fontsize=8)
  pp.savefig( save_dir + "/clustermap_z_to_meth.png", format="png", dpi=300 )
  
  
  #pdb.set_trace()
    #pdb.set_trace()
  #
  # binses = [20,50,100,500]
  # for bins in binses:
  #   pp.figure()
  #   pp.hist( aucs_true.values.flatten(), bins, range=(0,1), normed=True, histtype="step", lw=3, label="True" )
  #   pp.hist( aucs_random.values.flatten(), bins, color="red",range=(0,1), normed=True, histtype="step", lw=3, label="Random" )
  #   #pp.plot( [0,1.0],[0.5,0.5], 'r-', lw=3)
  #   pp.legend()
  #   pp.xlabel("Area Under the ROC")
  #   pp.ylabel("Pr(AUC)")
  #   pp.title("Comparison between AUC using latent space and random")
  #   pp.savefig( tissue_dir + "/auc_comparison_%dbins.png"%(bins), format='png', dpi=300 )
  #
  # pp.close('all')
  #pdb.set_trace()
  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )