from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
from tcga_encoder.algorithms import *
import seaborn as sns

from scipy import special
sns.set_style("whitegrid")
sns.set_context("talk")

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def keep_only( bcs, list_of_tissues ):
  tissues = np.array( [s.split("_")[0] for s in bcs ], dtype=str )
  
  query = np.zeros( len(bcs), dtype=bool)
  for tissue in list_of_tissues:
    query |= tissues == tissue
    
  return bcs[ query ]
  
if __name__ == "__main__":
  auc_threshold = 0.6
  data_location =  os.path.join( HOME_DIR,  "data/broad_processed_april_2017/20160128/pan_small_rna_dna_set_dna100"  )
  #data_location =  os.path.join( HOME_DIR,  "data/broad_processed_april_2017/20160128/pan_medium_multi_set_dna100"  )
  
  #data_location =  os.path.join( HOME_DIR,  "data/broad_processed_april_2017/20160128/pan_tiny_multi_set_dna100"  )
  fill_location =  os.path.join( HOME_DIR,  "results/tcga_vae_post_recomb3/medium/adversarial/anti_0_brca"  )
  #fill_location =  os.path.join( HOME_DIR,  "results/tcga_vae_post_recomb3/medium/adversarial/anti_weight_1000_H500_Z100_brca_obs_inputs"  )
  save_location = fill_location
  
  
  data_store = pd.HDFStore( data_location + "/data.h5", "r")
  fill_store = pd.HDFStore( fill_location + "/full_vae_fill.h5", "r")
  
  observed = data_store["CLINICAL/observed"]
  
  query = (observed["DNA"] * observed["RNA"] > 0 )
  
  bcs = query.index[ query.values ].values
  
  R = data_store["/RNA/RSEM"].loc[ bcs ]
  D = data_store["/DNA/channel/0"].loc[ bcs ]
  
  train_bcs = np.intersect1d( fill_store["/Z/TRAIN/Z/mu"].index.values, bcs )
  
  train_bcs = keep_only( train_bcs, ["coad", "read"] )
  
  val_bcs   = np.intersect1d( fill_store["/Z/VAL/Z/mu"].index.values, bcs )
  
  Z_mu_train  = fill_store["/Z/TRAIN/Z/mu"].loc[ train_bcs ]
  Z_var_train = fill_store["/Z/TRAIN/Z/var"].loc[ train_bcs ]
  Z_mu_val    = fill_store["/Z/VAL/Z/mu"].loc[ val_bcs ]
  Z_var_val   = fill_store["/Z/VAL/Z/var"].loc[ val_bcs ]
  
  n_z = len(Z_mu_train.columns)
  dna2use = ["APC","BRAF","TP53", "PTEN"]
  rna2use = R.columns #
  #rna2use = ["MUC2","DES","PFN2","MMP1","MYH11"]
  
  R_train = np.log2( 2.0 + R.loc[train_bcs][ rna2use ] ) - 1.0
  R_val   = np.log2( 2.0 + R.loc[val_bcs][ rna2use ] ) - 1.0
  #R_train = R.loc[train_bcs][ rna2use ]
  #R_val   = R.loc[val_bcs][ rna2use ]
  for dna_gene in dna2use:
    dna_train = D[dna_gene].loc[ train_bcs ]
    dna_val   = D[dna_gene].loc[ val_bcs ]
    
    mut_ids_1_train = dna_train[ dna_train > 0 ]
    mut_ids_0_train = dna_train[ dna_train < 1 ]
    
    train_1_bcs = mut_ids_1_train.index.values
    train_0_bcs = mut_ids_0_train.index.values
    
    train_ids_1 = pp.find( mut_ids_1_train.values==1 )
    train_ids_0 = pp.find( mut_ids_0_train.values==0 )
    
    mut_ids_1_val = dna_val[ dna_val > 0 ]
    mut_ids_0_val = dna_val[ dna_val == 0 ]
    
    val_1_bcs = mut_ids_1_val.index.values
    val_0_bcs = mut_ids_0_val.index.values
    
    val_ids_1 = pp.find( mut_ids_1_train.values==1 )
    val_ids_0 = pp.find( mut_ids_0_train.values==0 )
    
    mu_1 = Z_mu_train.loc[train_1_bcs].mean()
    mu_0 = Z_mu_train.loc[train_0_bcs].mean()
    var_1 = Z_mu_train.loc[train_1_bcs].var()
    var_0 = Z_mu_train.loc[train_0_bcs].var()
    
    cov = Z_mu_train.cov()+0.1*np.eye(n_z)
    icov = np.linalg.inv(cov)
    #cov_1 = Z_mu_train.loc[train_1_bcs].cov()
    #cov_0 = Z_mu_train.loc[train_0_bcs].cov()
    
    pi_1 = float(len(train_1_bcs)) / (float(len(Z_mu_train)))
    pi_0 = 1.0 - pi_1 
    
    log_ratio = np.log(pi_1) - np.log(pi_0)
    common_b = np.log(pi_1) - np.log(pi_0)
    
    w = np.dot( icov,  mu_1 - mu_0 )
    w0 = -0.5*np.dot( np.dot( mu_1.T, icov ), mu_1 )+0.5*np.dot( np.dot( mu_0.T, icov ), mu_0 ) + log_ratio
    
    activations_train = np.dot(Z_mu_train, w ) + w0 
    activations_val  = np.dot( Z_mu_val, w ) + w0 
    
    predictions_train = 1.0 / (1.0 + np.exp(-activations_train) )
    predictions_val = 1.0 / (1.0 + np.exp(-activations_val) )
    
    auc_train = roc_auc_score( dna_train.values, predictions_train )
    auc_val   = roc_auc_score( dna_val.values, predictions_val )
    
    print "======================"
    print "Z -> %s --- auc train = %0.3f  val %0.3f"%(dna_gene,auc_train,auc_val)
    print "----------------------"
    # for z_idx in range(mu_1.values.shape[0]):
    #   mu_1_z = mu_1.loc[ "z%d"%(z_idx)]
    #   mu_0_z = mu_0.loc[ "z%d"%(z_idx)]
    #   var_1_z = var_1.loc[ "z%d"%(z_idx)]
    #   var_0_z = var_0.loc[ "z%d"%(z_idx)]
    #   v = cov.values[ z_idx,: ][ z_idx ]
    #
    #   w_z = (mu_1_z - mu_0_z )/v
    #   w0_z = -0.5*(mu_1_z*mu_1_z/v )+0.5*mu_0_z*mu_0_z/v + log_ratio
    #
    #   activations_train_z = np.dot(Z_mu_train.values[:,z_idx], w_z ) + w0_z
    #   activations_val_z  = np.dot(Z_mu_val.values[:,z_idx], w_z ) + w0_z
    #
    #   predictions_train_z = 1.0 / (1.0 + np.exp(-activations_train_z) )
    #   predictions_val_z = 1.0 / (1.0 + np.exp(-activations_val_z) )
    #
    #   auc_train_z = roc_auc_score( dna_train.values, predictions_train_z )
    #   auc_val_z   = roc_auc_score( dna_val.values, predictions_val_z )
    #
    #   #print "Z%d -> %s --- auc train = %0.3f  val %0.3f"%(z_idx,dna_gene,auc_train_z,auc_val_z)
    # #print "======================"
    #assert False
    #pdb.set_trace()  
    r_mu_1 = R_train.loc[train_1_bcs].mean()
    r_mu_0 = R_train.loc[train_0_bcs].mean()
    r_var_1 = R_train.loc[train_1_bcs].var()
    r_var_0 = R_train.loc[train_0_bcs].var()
    
    #prob_success_1 = (r_var_1 - r_mu_1)/r_var_1
    #prob_success_0 = (r_var_0 - r_mu_0)/r_var_0
    #prob_success_1 = r_mu_1/r_var_1
    #prob_success_0 = r_mu_0/r_var_0
    
    # failure_1 = r_mu_1*r_mu_1/(r_var_1 - r_mu_1)
    # failure_0 = r_mu_0*r_mu_0/(r_var_0 - r_mu_0)
    #
    # alpha_1 = r_mu_1*( r_mu_1*(1-r_mu_1)/r_var_1 - 1.0 )
    # alpha_0 = r_mu_0*( r_mu_0*(1-r_mu_0)/r_var_0 - 1.0 )
    #
    # beta_1 = (1.0-r_mu_1)*( r_mu_1*(1-r_mu_1)/r_var_1 - 1.0 )
    # beta_0 = (1.0-r_mu_0)*( r_mu_0*(1-r_mu_0)/r_var_0 - 1.0 )
    
    aucs_val = []; aucs_train = []
    activations_train_mean = np.zeros(len(R_train))
    activations_val_mean = np.zeros(len(R_val))
    nbr_ok = 1
    for rna_gene in rna2use:
      rate_0 = r_mu_0.loc[rna_gene]; rate_1 = r_mu_1.loc[rna_gene]
      
      w = np.array([ np.log(rate_1) - np.log(rate_0)])
                     
                    
      z_train = R_train[rna_gene].values 
      z_val   = R_val[rna_gene].values 

      b = rate_0 - rate_1
      
      activations_train = z_train*w + b 
      activations_val   = z_val*w + b 
      
      
      predictions_train = 1.0 / (1.0 + np.exp(-activations_train-common_b) )
      predictions_val = 1.0 / (1.0 + np.exp(-activations_val-common_b) )
    
      auc_train = roc_auc_score( dna_train.values, predictions_train )
      auc_val   = roc_auc_score( dna_val.values, predictions_val )
      
      #ws.append( w )
      
      if auc_val > auc_threshold and auc_train > auc_threshold:
        activations_train_mean += activations_train
        activations_val_mean += activations_val
        nbr_ok +=1
      #print "%s -> %s --- auc train = %0.3f  val %0.3f"%(rna_gene, dna_gene,auc_train,auc_val)
      #
      # pp.figure()
      # pp.hist( R_train.loc[train_1_bcs][rna_gene], 20, normed=True, color='green', alpha=0.5 )
      # pp.hist( R_train.loc[train_0_bcs][rna_gene], 20, normed=True, color='red', alpha=0.5 )
      #
      # pp.hist( R_val.loc[val_1_bcs][rna_gene], 10, histtype="step", normed=True, color='green', lw=3 )
      # pp.hist( R_val.loc[val_0_bcs][rna_gene], 10, histtype="step", normed=True, color='red', lw=3 )
      #
      # pp.title( "%s"%(rna_gene))
      
      aucs_val.append( auc_val)
      aucs_train.append( auc_train )
    
    aucs_val = np.array( aucs_val )
    aucs_train = np.array( aucs_train )

    predictions_train_mean = 1.0 / (1.0 + np.exp(-activations_train_mean-common_b) )
    predictions_val_mean = 1.0 / (1.0 + np.exp(-activations_val_mean-common_b) )
  
    auc_train_mean = roc_auc_score( dna_train.values, predictions_train_mean )
    auc_val_mean   = roc_auc_score( dna_val.values, predictions_val_mean )

    
    I = pp.find( aucs_val > auc_threshold ); J = pp.find( aucs_train > auc_threshold );
    
    ids_to_use = np.intersect1d( I, J )
    
    order = np.argsort( 1-aucs_val[ ids_to_use ] )

    print "======================"
    print "BEST RNA PREDICTORS for %s: "%(dna_gene)
    print "---------------------"
    for idx in order[:10]:
      i = ids_to_use[idx]
      
      print "%s -> %s train = %0.3f  val = %0.3f"%(rna2use[i], dna_gene, aucs_train[i],aucs_val[i])
      rna_gene = rna2use[i]
      pp.figure()
      pp.hist( R_train.loc[train_1_bcs][rna_gene], 20, normed=True, color='green', alpha=0.5 )
      pp.hist( R_train.loc[train_0_bcs][rna_gene], 20, normed=True, color='red', alpha=0.5 )

      pp.hist( R_val.loc[val_1_bcs][rna_gene], 10, histtype="step", normed=True, color='green', lw=3 )
      pp.hist( R_val.loc[val_0_bcs][rna_gene], 10, histtype="step", normed=True, color='red', lw=3 )

      pp.title( "%s to %s"%(rna_gene, dna_gene))
      
    print "---------------------"  
    print "MEAN -> %s train = %0.3f  val = %0.3f"%( dna_gene, auc_train_mean,auc_val_mean)
    
    
      
      
      
    
    
    
    
    
    
    
    
    
    
    
    
  
  