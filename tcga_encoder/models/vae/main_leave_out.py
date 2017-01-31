from tcga_encoder.utils.helpers import *
#from tcga_encoder.data.data import *
#from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
import pdb
import sys
import tcga_encoder.models.vae.main as runner

def replace_template(template, disease_group ):
  
  disease_group = disease_group.rstrip('\n')
  disease_list = disease_group.split(" ")
  
  y = load_yaml( template )
  
  y['data']['validation_tissues'] = []
  if disease_list.__class__ == list:
    y['data']['validation_tissues'] = disease_list
    diseases = ""
    if len(disease_list)>1:
      for d in disease_list:
        diseases += "%s_"%(d)
    else:
      diseases = disease_list[0]   
      
  else:
    y['data']['validation_tissues'] = [disease_list]
    diseases = disease_list
  
  y['logging']['experiment'] += diseases
  
  this_yaml = template.rstrip(".yaml") + "_runner.yaml"
  #template
  with open(this_yaml, 'w') as yaml_file:
    yaml.dump(y, yaml_file, default_flow_style=False)
      
  #pdb.set_trace()
  return this_yaml


if __name__ == "__main__":
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_template_file = sys.argv[1]
  disease_list_file = sys.argv[2]
  print "Running template: ",yaml_template_file
  
  disease_list = open( disease_list_file, "r" ).readlines()
  for disease_group in disease_list:
    print "Running disease_group = %s"%(disease_group)
    
    yaml_file = replace_template( yaml_template_file, disease_group )
    runner.main( yaml_file )
    #break

  
  
  