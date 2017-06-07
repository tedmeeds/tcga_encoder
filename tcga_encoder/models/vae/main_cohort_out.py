from tcga_encoder.utils.helpers import *
from tcga_encoder.definitions.locations import *
import pdb
import sys
import tcga_encoder.models.vae.main as runner

def replace_template(template, cohorts_to_leave_out, move2train = 0.0 ):
  
  y = load_yaml( template )
  
  this_yaml = template.rstrip(".yaml") + "_runner.yaml"
  
  print "setting move2train ", move2train
  y['architecture']['variables']['move2train'] = move2train
  if y['algorithm'].has_key('xval_fold'):
    print "popping xval_fold"
    y['algorithm'].pop('xval_fold')
    
  print "setting validation_tissues", cohorts_to_leave_out
  y['data']['validation_tissues'] = cohorts_to_leave_out
  #y['logging']['experiment'] += "/fold_%d_of_%d"%(fold_idx,y['algorithm']['n_xval_folds'] )
  s=""
  y['logging']['experiment'] += "/%s"%( s.join(cohorts_to_leave_out) )
  
  with open(this_yaml, 'w') as yaml_file:
    yaml.dump(y, yaml_file, default_flow_style=False)
      
  return this_yaml


# def run_dna_test( observations, Z ):
#
#   z_names = Z.columns
#   n_z = len(z_names)
#
#   bcs = Z.index.values
#   cohorts = np.array(  [s.split("_")[0] for s in bcs] )
#   u_cohorts = np.unique( cohorts )
#   n_cohorts = len(u_cohorts)
#
#   Z_auc = np.zeros( (n_cohorts,n_z), dtype=float )
#   good_cohorts = []
#   for i_cohort, cohort in zip( range(n_cohorts), u_cohorts ):
#     I = cohorts == cohort
#
#     bcs_i = bcs[I]
#     z_i = Z.loc[bcs_i]
#     try:
#       y_i = observations.loc[bcs_i].values
#     except:
#       print "skipping ",cohort
#       continue
#     #print cohort, np.sum(y_i), y_i
#     oks = pp.find( np.isnan(y_i) == False )
#
#     y_i = y_i[oks]
#     bcs_i = bcs_i[oks]
#     z_i = z_i.loc[ bcs_i]
#
#     if np.sum( y_i ) > 0:
#       good_cohorts.append( cohort )
#       for z_idx in range(n_z):
#         auc = roc_auc_score( y_i, z_i.values[:,z_idx] )
#
#         Z_auc[i_cohort,z_idx] = auc
#
#
#   df =  pd.DataFrame( Z_auc, index = u_cohorts, columns = z_names ).loc[good_cohorts]
#   return df

# def run_dna_test_full( observations, Z ):
#
#   z_names = Z.columns
#   n_z = len(z_names)
#
#   bcs = Z.index.values
#   # cohorts = np.array(  [s.split("_")[0] for s in bcs] )
#   # u_cohorts = np.unique( cohorts )
#   # n_cohorts = len(u_cohorts)
#
#   Z_auc = np.zeros( (1,n_z), dtype=float )
#   good_cohorts = []
#
#   y_i = observations.values.sum(1)
#   oks = pp.find( np.isnan(y_i) == False )
#
#   y_i = y_i[oks]
#   bcs_i = bcs[oks]
#   z_i = Z.loc[ bcs_i]
#   y_i = np.minimum(y_i,1)
#   if np.sum( y_i ) > 0:
#     #good_cohorts.append( cohort )
#     for z_idx in range(n_z):
#       auc = roc_auc_score( y_i, z_i.values[:,z_idx] )
#
#       Z_auc[0,z_idx] = auc
#
#
#   df =  pd.DataFrame( Z_auc, index = ["ALL"], columns = z_names )
#   return df
        
  
if __name__ == "__main__":
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_template_file = sys.argv[1]
  print sys.argv
  cohorts_to_leave_out = []
  i = 2
  while i < len(sys.argv):
    cohorts_to_leave_out.append( sys.argv[i] )
    i+=1
    
  print "Running template: ",yaml_template_file
  print "Leaving out: ", cohorts_to_leave_out
  template_yaml = load_yaml( yaml_template_file )

    
  yaml_file = replace_template( yaml_template_file, cohorts_to_leave_out )
  s = "python tcga_encoder/models/vae/main.py %s "%(yaml_file)
  os.system(s)
