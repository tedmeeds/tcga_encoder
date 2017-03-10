from tcga_encoder.utils.helpers import *
#from tcga_encoder.data.data import *
#from tcga_encoder.definitions.tcga import *
#from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
import pdb
import sys
import tcga_encoder.models.survival.main_xval_x_to_z_model_selection as runner

def replace_template(template, disease_group ):
  
  disease_group = disease_group.rstrip('\n')
  disease_list = disease_group.split(" ")
  
  y = load_yaml( template )
  
  y['data']['validation_tissues'] = []
  if disease_list.__class__ == list:
    y['data']['validation_tissues'] = disease_list
    diseases = disease_list[0] 
    for d in disease_list[1:]:
      diseases += "_%s"%(d) 
      
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
  weights_matrix = []

  diseases = []
  for disease_group in disease_list:
    print "Running disease_group = %s"%(disease_group)
    diseases.append( disease_group.rstrip("\n"))
    yaml_file = replace_template( yaml_template_file, disease_group )
    runner.main( yaml_file, weights_matrix )
    #weights.append( yaml_file["weights"] )
  
  #pdb.set_trace()
  # weights_matrix = np.array(weights_matrix)
  # columns = ["z%d"%z for z in range(weights_matrix.shape[1])]
  # df = pd.DataFrame( weights_matrix, columns = columns, index = diseases )
  # y = load_yaml( yaml_template_file )
  # logging_dict   = y[LOGGING]
  # logging_dict[SAVEDIR] = os.path.join( HOME_DIR, logging_dict[LOCATION]  )
  # w_location = os.path.join( logging_dict[SAVEDIR], "weights.csv" )
  # df.to_csv( w_location)
  # print df
    #break

  
  
  