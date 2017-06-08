from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
#from tcga_encoder.algorithms import *
import seaborn as sns
from sklearn.manifold import TSNE, locally_linear_embedding
#import scipy.spatial.distance.pdist
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


# cloudy blue  #acc2d9
# dark pastel green  #56ae57
# dust  #b2996e
# electric lime  #a8ff04
# fresh green  #69d84f
# light eggplant  #894585
# nasty green  #70b23f
# really light blue  #d4ffff
# tea  #65ab7c
# warm purple  #952e8f
# yellowish tan  #fcfc81
# cement  #a5a391
# dark grass green  #388004
# dusty teal  #4c9085
# grey teal  #5e9b8a
# macaroni and cheese  #efb435
# pinkish tan  #d99b82
# spruce  #0a5f38
# strong blue  #0c06f7
# toxic green  #61de2a
# windows blue  #3778bf
# blue blue  #2242c7
# blue with a hint of purple  #533cc6
# booger  #9bb53c
# bright sea green  #05ffa6
# dark green blue  #1f6357
# deep turquoise  #017374
# green teal  #0cb577
# strong pink  #ff0789
# bland  #afa88b
# deep aqua  #08787f
# lavender pink  #dd85d7
# light moss green  #a6c875
# light seafoam green  #a7ffb5
# olive yellow  #c2b709
# pig pink  #e78ea5
# deep lilac  #966ebd
# desert  #ccad60
# dusty lavender  #ac86a8
# purpley grey  #947e94
# purply  #983fb2
# candy pink  #ff63e9
# light pastel green  #b2fba5
# boring green  #63b365
# kiwi green  #8ee53f
# light grey green  #b7e1a1
# orange pink  #ff6f52
# tea green  #bdf8a3
# very light brown  #d3b683
# egg shell  #fffcc4
# eggplant purple  #430541
# powder pink  #ffb2d0
# reddish grey  #997570
# baby shit brown  #ad900d
# liliac  #c48efd
# stormy blue  #507b9c
# ugly brown  #7d7103
# custard  #fffd78
# darkish pink  #da467d

 

tissue_color_names = ["windows blue", "amber", "greyish", "faded green", "dusty purple",\
                "nice blue","rosy pink","sand brown","baby purple",\
                "fern","creme","ugly blue","washed out green","squash",\
                "cinnamon","radioactive green","cocoa","charcoal grey","indian red",\
                "light lavendar","toupe","dark cream" ,"burple","tan green",\
                "azul","bruise", "sunny yellow","deep brown","off blue",\
                "custard","powder pink","deep lilac","kiwi green","orange pink"]

def main( data_location, results_location ):
  data_path    = os.path.join( HOME_DIR ,data_location ) #, "data.h5" )
  results_path = os.path.join( HOME_DIR, results_location )
  
  data_filename = os.path.join( data_path, "data.h5")
  fill_filename = os.path.join( results_path, "full_vae_fill.h5" )
  
  save_dir = os.path.join( results_path, "kmeans_with_z" )
  check_and_mkdir(save_dir)
  size_per_unit = 0.25
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
  quantiles = (len(Z)*np.array( [0,0.33, 0.66, 1.0] )).astype(int)
  #quantiles = (len(Z)*np.array( [0,0.2, 0.4,0.6,0.8,1.0] )).astype(int)
  #quantiles = (len(Z)*np.array( [0,0.1, 0.2,0.3,0.4,0.6,0.7,0.8,0.9,1.0] )).astype(int)
  n_quantiles = len(quantiles)-1
  start_q_id = -(n_quantiles-1)/2
  Z=Z.loc[barcodes]
  Z_values = Z.values
  
  argsort_Z = np.argsort( Z_values, 0 )
  
  Z_quantized = np.zeros( Z_values.shape, dtype=int )
  for start_q, end_q in zip( quantiles[:-1], quantiles[1:] ):
    for z_idx in range(n_z):
      z_idx_order = argsort_Z[:,z_idx] 
      Z_quantized[ z_idx_order[start_q:end_q], z_idx] = start_q_id
    start_q_id+=1
    
  Z_quantized = pd.DataFrame(Z_quantized, index=barcodes, columns=z_names )
  Z_quantized.to_csv( save_dir + "/Z_quantized.csv")
  
  sub_bcs = np.array([ x+"_"+y for x,y in np.array(data_store["/CLINICAL/data"]["patient.stage_event.pathologic_stage"].index.tolist(),dtype=str)] )
  sub_values = np.array( data_store["/CLINICAL/data"]["patient.stage_event.pathologic_stage"].values, dtype=str )
  subtypes = pd.Series( sub_values, index = sub_bcs, name="subtypes")
  
  from sklearn.cluster import MiniBatchKMeans
  # print "running kmeans"
  # kmeans_patients = MiniBatchKMeans(n_clusters=10, random_state=0).fit(Z_quantized.values)
  # kmeans_patients_labels = kmeans_patients.labels_
  #
  # kmeans_z = MiniBatchKMeans(n_clusters=10, random_state=0).fit(Z_quantized.values.T)
  # kmeans_z_labels = kmeans_z.labels_
  #
  #
  # order_labels = np.argsort(kmeans_patients_labels)
  # order_labels_z = np.argsort(kmeans_z_labels)
  # sorted_Z = pd.DataFrame( Z_quantized.values[order_labels,:], index=Z_quantized.index[order_labels], columns=Z_quantized.columns)
  # sorted_Z = pd.DataFrame( sorted_Z.values[:,order_labels_z], index=sorted_Z.index, columns = sorted_Z.columns[order_labels_z] )
  
  tissues = data_store["/CLINICAL/TISSUE"].loc[barcodes]
  
  tissue_names = tissues.columns
  tissue_idx = np.argmax( tissues.values, 1 )

  n = len(Z)
  n_tissues = len(tissue_names)
  K_p = 5
  K_z = 10
  for t_idx in range(n_tissues):
    tissue_name = tissue_names[t_idx]
    
    print "working %s"%(tissue_name)
    
    t_ids_cohort = tissue_idx == t_idx
    n_tissue = np.sum(t_ids_cohort)
 
    if n_tissue < 1:
      continue
    
    Z_cohort = Z_quantized[ t_ids_cohort ]
    
    bcs = barcodes[t_ids_cohort]
    
    kmeans_patients = MiniBatchKMeans(n_clusters=K_p, random_state=0).fit(Z_cohort.values)
    kmeans_patients_labels = kmeans_patients.labels_

    kmeans_z = MiniBatchKMeans(n_clusters=K_z, random_state=0).fit(Z_cohort.values.T)
    kmeans_z_labels = kmeans_z.labels_

  
    order_labels = np.argsort(kmeans_patients_labels)
    order_labels_z = np.argsort(kmeans_z_labels)
    sorted_Z = pd.DataFrame( Z_cohort.values[order_labels,:], index=Z_cohort.index[order_labels], columns=Z_cohort.columns)
    sorted_Z = pd.DataFrame( sorted_Z.values[:,order_labels_z], index=sorted_Z.index, columns = sorted_Z.columns[order_labels_z] )

    cohort_subtypes = subtypes.loc[bcs]
    subtype_names = np.unique(cohort_subtypes.values)
    subtype2colors = OrderedDict( zip(subtype_names,sns.color_palette("Blues", len(subtype_names))) )
    subtype_colors = np.array( [subtype2colors[subtype] for subtype in cohort_subtypes.values] )

    
    f = pp.figure(figsize=(12,12))
    ax=f.add_subplot(111)
    h = sns.heatmap( sorted_Z, ax=ax )
    #pdb.set_trace()
    pp.setp(h.yaxis.get_majorticklabels(), rotation=0)
    pp.setp(h.xaxis.get_majorticklabels(), rotation=90)
    pp.setp(h.yaxis.get_majorticklabels(), fontsize=12)
    pp.setp(h.xaxis.get_majorticklabels(), fontsize=12)
    #h.ax_row_dendrogram.set_visible(False)
    #h.ax_col_dendrogram.set_visible(False)
    #h.cax.set_visible(False)
    pp.savefig( save_dir + "/Z_kmeans_%s.png"%(tissue_name), fmt="png", dpi=300, bbox_inches='tight')
    pp.close('all')
    #pdb.set_trace()
    
    
            
  #pdb.set_trace()

    #d_mat = pdist( Z_cohort.values )
    #s_form = squareform(d_mat)
    #csr = csr_matrix(np.triu(s_form))
    #Tcsr = minimum_spanning_tree(csr)
    #as_mat = Tcsr.toarray()
    
    #pp.figure(figsize=(16,16))
    
    # i=0
    # for x in Tcsr:
    #   indices = x.indices
    #   weights = x.data
    #
    #   for j,w in zip(indices,weights):
    #     G.add_edge(bcs[i][-7:], bcs[j][-7:], weight=w)
    #   i+=1
    # layout=nx.spring_layout
    # #layout=nx.spectral_layout
    # pos=layout(G)
    # nx.draw(G,pos,
    #             with_labels=True,
    #             node_size=1000, hold=False, node_color='b'
    #             )
    # G.clear()
    # pp.title("%s"%(tissue_name))
    # pp.savefig(save_dir + "/%s_mwst.png"%(tissue_name), fmt='png',dpi=300)
    # pp.close('all')
    # #pdb.set_trace()
    # f = pp.figure()
    # ax = f.add_subplot(111)
    #
    # size1 = max( int( n_z*size_per_unit ), 20 )
    # size2 = min( max( int( n_tissue*size_per_unit ), 12 ), 20 )
    #
    # #
    # # if len(subtype_names)>1:
    # #   h = sns.clustermap( Z_cohort, square=False, figsize=(size1,size2), row_colors = subtype_colors  )
    # # else:
    # #   h = sns.clustermap( Z_cohort, square=False, figsize=(size1,size2)  )
    # # pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    # # pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    # # pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
    # # pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
    # # h.ax_row_dendrogram.set_visible(False)
    # # h.ax_col_dendrogram.set_visible(False)
    # # h.cax.set_visible(False)
    #
    #
    # pp.savefig( save_dir + "/Z_clustermap_%s.png"%(tissue_name), fmt="png", dpi=300, bbox_inches='tight')
    # pp.close('all')


  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  main( data_location, results_location )