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
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from tcga_encoder.analyses.survival_functions import *
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

# def get_global_cost( data, w, K, lambda_l1, lambda_l2, idx ):
#   cost = 0.0 #lambda_l1*np.sum( np.abs(w) ) + lambda_l2*np.sum(w*w)
#   d = data[ idx]
#   times = d["times"]
#   events = d["events"]
#   z = d["z"]
#   n = len(times)
#   cost = get_cost( times, events, z, w, K, lambda_l1, lambda_l2 )/n
#   return cost
  
def get_global_cost( data, w, K, lambda_l1, lambda_l2, idx ):
  cost = 0.0 #lambda_l1*np.sum( np.abs(w) ) + lambda_l2*np.sum(w*w)

  for d in data:
    times = d["times"]
    events = d["events"]
    z = d["z"]
    cost += get_cost( times, events, z, w, K, lambda_l1, lambda_l2 )
  return cost
  
# get_cost( times, events, z_train, w_delta_plus, K_p, lambda_l1, lambda_l2 )
def get_cost( times, events, z, w, K, lambda_l1, lambda_l2 ):
  cost = 0 #lambda_l1*np.sum( np.abs(w) ) + lambda_l2*np.sum(w*w)
  
  
  ok = pp.find( pp.isnan( times.values) == False  )
  
  y = np.dot( z[ok], w )
  e = events.values[ok]
  t = times.values[ok]
  ids = e==1
  n=len(t)
  K = min( K, int(e.sum() ))
  #pdb.set_trace()
  
  
  #results = stats.spearmanr( y, t )
  results = stats.spearmanr( y[ids], t[ids] )
  if np.isnan( results.pvalue ):
    pdb.set_trace()
  #print results
  #return cost + np.log( results.pvalue+1e-12 )
  #results = stats.spearmanr( y[ids], times.values[ids] )
  #return cost + np.sign(results.correlation)*np.log( results.pvalue+1e-12 )
  I_splits = survival_splits( e, np.argsort(y), 2 )
  bad_order=False
  z_score = 0
  # for k1 in range(K-1):
  #   g1 = I_splits[k1]
  #   g2 = I_splits[k1+1]
  #   z_score -= logrank_test( t[g1], t[g2], e[g1], e[g2] ).test_statistic
  z_score +=  np.log( logrank_test( t[I_splits[0]], t[I_splits[-1]], e[I_splits[0]], e[I_splits[-1]] ).p_value)
  #z_score +=  np.log( logrank_test( t[I_splits[1]], t[I_splits[-2]], e[I_splits[1]], e[I_splits[-2]] ).p_value)
  #z_score -= logrank_test( t[I_splits[2]], t[I_splits[-3]], e[I_splits[2]], e[I_splits[-3]] ).test_statistic
  #z_score -= logrank_test( t[g1], t[g2], e[g1], e[g2] ).test_statistic

  return cost + (z_score + np.log( results.pvalue+1e-12 ))
      
  # groups = groups_by_splits( len(z), I_splits )
  #
  #       cost_delta_plus = np.log( \
  #                         multivariate_logrank_test( \
  #                              times, \
  #                              groups=groups_by_splits( \
  #                                             n_tissue, \
  #                                             survival_splits( events, np.argsort(np.dot( z_train, w_delta_plus )), \
  #                                             K_p ) ), event_observed=events ).p_value + 1e-12 ) \
  #                        +lambda_l2*np.sum( np.abs(w_delta_plus))



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
  
  save_dir = os.path.join( results_path, "kmeans_with_z_global_learn_survival7" )
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
  barcodes = Z.index.values
  #barcodes = np.union1d( Z_train.index.values, Z_val.index.values )
  quantiles = (len(Z)*np.array( [0,0.33, 0.66, 1.0] )).astype(int)
  quantiles = (len(Z)*np.array( [0,0.2, 0.4,0.6,0.8,1.0] )).astype(int)
  #quantiles = (len(Z)*np.linspace(0,1,61)).astype(int)
  n_quantiles = len(quantiles)-1
  start_q_id = -(n_quantiles-1)/2
  #Z=Z.loc[barcodes]
  
  std_z = Z.values.std(0)
  
  keep_z = pp.find( std_z > 0.0 )
  z_names = ["z_%d"%(z_idx) for z_idx in keep_z]
  Z = Z[z_names]
  n_z = len(z_names)
  z_names =  ["z_%d"%z_idx for z_idx in range(Z.shape[1])]
  Z.columns = z_names
  #return Z
  #pdb.set_trace()
  #Z = pd.DataFrame( Z.values / Z.std(1).values[:,np.newaxis], index=Z.index, columns=Z.columns)
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
  Z_quantized=Z
  sub_bcs = np.array([ x+"_"+y for x,y in np.array(data_store["/CLINICAL/data"]["patient.stage_event.pathologic_stage"].index.tolist(),dtype=str)] )
  sub_values = np.array( data_store["/CLINICAL/data"]["patient.stage_event.pathologic_stage"].values, dtype=str )
  subtypes = pd.Series( sub_values, index = sub_bcs, name="subtypes")
  tissues = data_store["/CLINICAL/TISSUE"].loc[barcodes]
  
  tissue_names = tissues.columns
  tissue_idx = np.argmax( tissues.values, 1 )
  
  # -----------------------------
  # -----------------------------
  ALL_SURVIVAL = data_store["/CLINICAL/data"][["patient.days_to_last_followup","patient.days_to_death","patient.days_to_birth"]]
  tissue_barcodes = np.array( ALL_SURVIVAL.index.tolist(), dtype=str )
  surv_barcodes = np.array([ x+"_"+y for x,y in tissue_barcodes])
  NEW_SURVIVAL = pd.DataFrame( ALL_SURVIVAL.values, index =surv_barcodes, columns = ALL_SURVIVAL.columns )
  NEW_SURVIVAL = NEW_SURVIVAL.loc[barcodes]
  #clinical = data_store["/CLINICAL/data"].loc[barcodes]

  Age = NEW_SURVIVAL[ "patient.days_to_birth" ].values.astype(int)
  Times = NEW_SURVIVAL[ "patient.days_to_last_followup" ].fillna(0).values.astype(int)+NEW_SURVIVAL[ "patient.days_to_death" ].fillna(0).values.astype(int)
  Events = (1-np.isnan( NEW_SURVIVAL[ "patient.days_to_death" ].astype(float)) ).astype(int)

  ok_age_query = Age<-10
  ok_age = pp.find(ok_age_query )
  tissues = tissues[ ok_age_query ]
  #pdb.set_trace()
  Age=-Age[ok_age]
  Times = Times[ok_age]
  Events = Events[ok_age]
  s_barcodes = barcodes[ok_age]
  NEW_SURVIVAL = NEW_SURVIVAL.loc[s_barcodes]

  #ok_followup_query = NEW_SURVIVAL[ "patient.days_to_last_followup" ].fillna(0).values>=0
  #ok_followup = pp.find( ok_followup_query )

  bad_followup_query = NEW_SURVIVAL[ "patient.days_to_last_followup" ].fillna(0).values.astype(int)<0
  bad_followup = pp.find( bad_followup_query )

  ok_followup_query = 1-bad_followup_query
  ok_followup = pp.find( ok_followup_query )

  bad_death_query = NEW_SURVIVAL[ "patient.days_to_death" ].fillna(0).values.astype(int)<0
  bad_death = pp.find( bad_death_query )

  #pdb.set_trace()
  Age=Age[ok_followup]
  Times = Times[ok_followup]
  Events = Events[ok_followup]
  s_barcodes = s_barcodes[ok_followup]
  NEW_SURVIVAL = NEW_SURVIVAL.loc[s_barcodes]

  fill_store.close()
  data_store.close()
  # S = Z.loc[s_barcodes]
  # S["E"] = Events
  # S["T"] = Times
  # S["Age"] = np.log(Age)
  
  S = pd.DataFrame( np.vstack((Events,Times)).T, index = s_barcodes, columns=["E","T"])

  # -----------------------------
  # -----------------------------

  
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
  


  n = len(Z)
  n_tissues = len(tissue_names)
  K_p = 4
  K_z = 10
  k_pallette = sns.hls_palette(K_p)
  data = []
  
  for t_idx in range(n_tissues):
    #t_idx=1
    tissue_name = tissue_names[t_idx]
    print "working %s"%(tissue_name)
  
    t_ids_cohort = tissue_idx == t_idx
    n_tissue = np.sum(t_ids_cohort)
    
    if n_tissue < 1:
      print "skipping ",tissue_name
      continue
    Z_cohort = Z_quantized[ t_ids_cohort ]
    
    bcs = barcodes[t_ids_cohort]
    S_cohort = S.loc[bcs]
    events = S_cohort["E"]
    times  = S_cohort["T"]
    
    ok = np.isnan(times.values)==False
    bcs = S_cohort.index.values[ok]
    
    Z_cohort = Z_cohort.loc[bcs]
    S_cohort = S_cohort.loc[bcs]
    events = S_cohort["E"]
    times  = S_cohort["T"]
    z_train = Z_cohort.values
    
    data.append( {"tissue":tissue_name, "barcodes":bcs,"z":z_train,"events":events,"times":times})
    
  dims = len(z_names)
  w = 0.001*np.random.randn( dims )

  epsilon = 0.001
  learning_rate = 0.001
  mom = 0*w
  alpha=0.95
  lambda_l1=0.0
  lambda_l2=0.0
  cost = get_global_cost( data, w, K_p, lambda_l1, lambda_l2, 0 )
  print "prelim cost ", -1, cost 
  min_cost = cost
  for i in range(0):
    xw = 0.0001*np.random.randn( dims )
    cost = get_global_cost( data, xw, K_p, lambda_l1, lambda_l2 )
    print "prelim cost ", i, cost 
    if cost < min_cost:
      min_cost = cost
      w = xw
  cost=min_cost
  repeats = range(2)
  print -1, cost
  #dX = mc*dXprev + lr*(1-mc)*dperf/dX
  old_dw=0.0
  costs=[]
  all_costs=[]
  pp.close('all')
  # f=pp.figure()
  # pp.show()
  # pp.ion()
  # pp.plot( [-1], [cost], 'ro')
  for step in range(500):
    grad = np.zeros(dims)
    random_off = [] #np.random.permutation(dims)[:dims-10]
    
    for r in repeats:
      idx = np.random.randint(len(data))
      bernoulli = 2*stats.bernoulli( 0.5 ).rvs(dims) - 1
      delta_w = epsilon*bernoulli #np.random.randn(dims)
      #random_off = np.random.permutation(dims)[:dims/3]
      delta_w[random_off]=0
      bernoulli[random_off]=0
      w_delta_plus = w + delta_w
      w_delta_neg = w - delta_w


      cost_delta_plus = get_global_cost( data, w_delta_plus, K_p, lambda_l1, lambda_l2, idx )
      cost_delta_neg  = get_global_cost( data, w_delta_neg, K_p, lambda_l1, lambda_l2, idx )
      
      grad += bernoulli*(cost_delta_plus-cost_delta_neg)/(2*epsilon)
    grad /= len(repeats)  
    grad += lambda_l2*w + lambda_l1*np.sign(w)
    
    grad = grad / np.linalg.norm(grad)
    #grad[random_off] = 0
    
    if step==0:
      dw = learning_rate*grad
    else:  
      dw = alpha*old_dw + learning_rate*grad
    #w -= learning_rate*grad
    dw =learning_rate*grad 
    w -= dw + 0*learning_rate*np.random.randn(dims)
    old_dw = dw
    #epsilon *= 0.995
    if np.random.rand()<0.1:
      print "train cost_delta_plus ", step, cost_delta_plus, cost_delta_neg 
    if cost_delta_plus < cost:
      #w = w_delta_plus
      cost = cost_delta_plus
      learning_rate *= 1.0
      dw = learning_rate*grad
      print "***", step, cost, cost_delta_plus, np.sum(np.abs(w))
    else:
      learning_rate /= 1
      
    costs.append(cost)
    all_costs.append(cost_delta_plus)
    #
    # pp.plot( [step], [cost], 'ro')
    # pp.plot( [step], [cost_delta_plus], 'b.')
    # pp.plot( [step], [cost_delta_neg], 'b.')
    
    # if np.random.rand()<0.01:
    #   pp.draw()
  # pp.ioff()
  # pdb.set_trace()
  pp.figure()
  pp.plot(all_costs,'o-')
  pp.plot(costs,'o-')
  pp.savefig( save_dir + "/costs.png", fmt="png" )
  #pp.show()
  print step, cost, cost_delta_plus, np.sum(np.abs(w))


  for tissue_data in data:
    
    z_train = tissue_data["z"]
    events = tissue_data["events"]
    times = tissue_data["times"]
    bcs = tissue_data["barcodes"]
    n_tissue = len(events)
    tissue_name = tissue_data["tissue"]
    print "plotting ",tissue_name
    Z_cohort = pd.DataFrame( z_train, index = bcs, columns=z_names )
    
    y = np.dot( z_train, w )
    I = np.argsort(y)
    n_tissue=len(y)
    I_splits = survival_splits( events, I, min(K_p,int(events.sum())) )
    groups = groups_by_splits( n_tissue, I_splits )
    
    results = multivariate_logrank_test(times, groups=groups, event_observed=events )
    p_value = results.p_value
    
    size1 = max( min( int( n_z*size_per_unit ), 12), 16 )
    size2 = max( min( int( n_tissue*size_per_unit), 12), 16)
    
    z_order = np.argsort( -np.abs(w) )
    patient_order = np.argsort(y)
    m_times_0 = times.values[ I_splits[0]][ events[I_splits[0]].values==1].mean()
    m_times_1 = times.values[ I_splits[-1]][ events[I_splits[-1]].values==1].mean()
    
    k_pallette = sns.hls_palette(K_p)
    k_pallette = sns.color_palette("rainbow", K_p)
    k_pallette.reverse()
    if m_times_1 < m_times_0:
      # reverse pallette
      k_pallette.reverse()
    
    
    k_colors = np.array([k_pallette[int(i)] for i in groups[patient_order]] )
    #pdb.set_trace()
    sorted_Z = Z_cohort.values
    sorted_Z = sorted_Z[patient_order,:]
    sorted_Z = sorted_Z[:,z_order]
    
    
    #so
    sorted_Z = pd.DataFrame( sorted_Z, index = Z_cohort.index.values[patient_order], columns=Z_cohort.columns[z_order] )
    #pdb.set_trace()
    h = sns.clustermap( sorted_Z, row_colors=k_colors, row_cluster=False, col_cluster=False, figsize=(size1,size2) )
    pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    pp.setp(h.ax_heatmap.yaxis.get_majorticklabels(), fontsize=12)
    pp.setp(h.ax_heatmap.xaxis.get_majorticklabels(), fontsize=12)
    h.ax_row_dendrogram.set_visible(False)
    h.ax_col_dendrogram.set_visible(False)
    h.cax.set_visible(False)
    h.ax_heatmap.hlines(n_tissue-pp.find(np.diff(groups[patient_order]))-1, *h.ax_heatmap.get_xlim(), color="black", lw=5)
    pp.savefig( save_dir + "/%s_learned.png"%(tissue_name), fmt="png" )#, dpi=300, bbox_inches='tight')
    pp.close('all')
    
    
    f = pp.figure()
    ax= f.add_subplot(111)
    
   
    kp = 0
    kmf = KaplanMeierFitter()
    for i_split in I_splits:
      k_bcs = bcs[ i_split ]
      if len(k_bcs) > 1:
        kmf.fit(times[i_split], event_observed=events[i_split], label="k%d"%(kp)  )
        ax=kmf.plot(ax=ax,at_risk_counts=False,show_censors=True, color=k_pallette[kp],ci_show=False,lw=4)
      kp += 1
    pp.ylim(0,1)
    pp.title("%s p-value = %0.5f"%(tissue_name,p_value))
    pp.savefig( save_dir + "/%s_survival.png"%(tissue_name), format="png" )#, dpi=300)
  return Z

  
if __name__ == "__main__":
  
  data_location = sys.argv[1]
  results_location = sys.argv[2]
  
  kmf = main( data_location, results_location )