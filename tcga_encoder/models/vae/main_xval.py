from tcga_encoder.utils.helpers import *
from tcga_encoder.definitions.locations import *
import pdb
import sys
import tcga_encoder.models.vae.main as runner

def replace_template(template, fold_idx ):
  
  y = load_yaml( template )
  y['algorithm']['xval_fold'] = fold_idx

  this_yaml = template.rstrip(".yaml") + "_runner.yaml"
  
  y['logging']['experiment'] += "/fold_%d_of_%d"%(fold_idx,y['algorithm']['n_xval_folds'] )
  
  with open(this_yaml, 'w') as yaml_file:
    yaml.dump(y, yaml_file, default_flow_style=False)
      
  return this_yaml


def run_dna_test( observations, Z ):
  
  z_names = Z.columns
  n_z = len(z_names)
  
  bcs = Z.index.values
  cohorts = np.array(  [s.split("_")[0] for s in bcs] )
  u_cohorts = np.unique( cohorts )
  n_cohorts = len(u_cohorts)
  
  Z_auc = np.zeros( (n_cohorts,n_z), dtype=float )
  good_cohorts = []
  for i_cohort, cohort in zip( range(n_cohorts), u_cohorts ):
    I = cohorts == cohort
    
    bcs_i = bcs[I]
    z_i = Z.loc[bcs_i]
    try:
      y_i = observations.loc[bcs_i].values
    except:
      print "skipping ",cohort
      continue
    #print cohort, np.sum(y_i), y_i
    oks = pp.find( np.isnan(y_i) == False )
    
    y_i = y_i[oks]
    bcs_i = bcs_i[oks]
    z_i = z_i.loc[ bcs_i]
    
    if np.sum( y_i ) > 0:
      good_cohorts.append( cohort )
      for z_idx in range(n_z):
        auc = roc_auc_score( y_i, z_i.values[:,z_idx] )
        
        Z_auc[i_cohort,z_idx] = auc
        
  
  df =  pd.DataFrame( Z_auc, index = u_cohorts, columns = z_names ).loc[good_cohorts]
  return df

def run_dna_test_full( observations, Z ):
  
  z_names = Z.columns
  n_z = len(z_names)
  
  bcs = Z.index.values
  # cohorts = np.array(  [s.split("_")[0] for s in bcs] )
  # u_cohorts = np.unique( cohorts )
  # n_cohorts = len(u_cohorts)
  
  Z_auc = np.zeros( (1,n_z), dtype=float )
  good_cohorts = []
  
  y_i = observations.values.sum(1)
  oks = pp.find( np.isnan(y_i) == False )
    
  y_i = y_i[oks]
  bcs_i = bcs[oks]
  z_i = Z.loc[ bcs_i]
  y_i = np.minimum(y_i,1)
  if np.sum( y_i ) > 0:
    #good_cohorts.append( cohort )
    for z_idx in range(n_z):
      auc = roc_auc_score( y_i, z_i.values[:,z_idx] )
      
      Z_auc[0,z_idx] = auc
        
  
  df =  pd.DataFrame( Z_auc, index = ["ALL"], columns = z_names )
  return df
        
  
if __name__ == "__main__":
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_template_file = sys.argv[1]
  run = False
  if len(sys.argv) == 3:
    run = bool(int( sys.argv[2]))
  print "Running template: ",yaml_template_file
  
  template_yaml = load_yaml( yaml_template_file )
  n_xval_folds = template_yaml["algorithm"]["n_xval_folds"]
  
  if template_yaml["algorithm"].has_key("runner"):
    runner = template_yaml["algorithm"]["runner"]
  else:
    runner = runner.main
  for fold in range(n_xval_folds):
    print "Running XVAL = %d"%(fold)
    
    yaml_file = replace_template( yaml_template_file, fold+1 )
    if run is True:
      runner( yaml_file )
      
      #s = "python tcga_encoder/models/dna/naive_bayes_by_gene.py %s %s %s %s %s %d %d %d 1 %s"%(data_file,results_location,dna_gene,source,method,n_folds,n_xval_repeats,n_permutations,diseases)
      #os.system(s)
      
      s = "python tcga_encoder/models/vae/main.py %s "%(yaml_file)
      os.system(s)
    else:
      print "Run is OFF by default, pass argument 1 to turn on"
    
  print "RUNNING XVAL COLLECTOR..."
  fill_dna, loglik_dna, dna, results, weights, train_dna_and_z = template_yaml["algorithm"]["xval_collector"]( yaml_template_file )
  fold_weights = weights[0]
  mean_weights = weights[1]
  
  train_dna_predictions = train_dna_and_z[0]
  train_z_mu = train_dna_and_z[1]
  
  train_predictions_as_ints = pd.DataFrame( (train_dna_predictions>0.75).values.astype(int), index=train_dna_predictions.index, columns=train_dna_predictions.columns)
  
  data = pd.HDFStore( os.path.join( HOME_DIR, template_yaml["data"]["location"] + "/data.h5" ) )
  
  dna_data = data["/DNA/channel/0"]
  
  dna_gene="TP53"
  df_true = run_dna_test( dna_data[dna_gene], train_z_mu )
  df_est  = run_dna_test( train_predictions_as_ints[dna_gene], train_z_mu )