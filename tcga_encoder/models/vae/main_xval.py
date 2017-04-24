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
    else:
      print "Run is OFF by default, pass argument 1 to turn on"
    
  print "RUNNING XVAL COLLECTOR..."
  fill_dna, loglik_dna, dna, results, weights = template_yaml["algorithm"]["xval_collector"]( yaml_template_file )
  fold_weights = weights[0]
  mean_weights = weights[1]