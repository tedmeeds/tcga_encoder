from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
from tcga_encoder.algorithms import *
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("talk")

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#import tensorflow as tf

#load_data_from_dict

#from tensorflow import *



#from models.layers import *
#from models.regularizers import *
#from models.algorithms import *

#from models.vae.tcga_models import *
#from utils.utils import *
#from data.load_datasets_from_broad import load_sources

#from utils.image_utils import *


def add_variables( var_dict, data_dict ):
  # add very specific numbers:
  var_dict["dna_dim"]    = data_dict['dataset'].GetDimension("DNA")
  var_dict["meth_dim"]   = data_dict['dataset'].GetDimension("METH")
  var_dict["rna_dim"]    = data_dict['dataset'].GetDimension("RNA")
  var_dict["mirna_dim"]    = data_dict['dataset'].GetDimension("miRNA")
  var_dict["tissue_dim"] = data_dict['dataset'].GetDimension("TISSUE")
  
def load_architecture( arch_dict, data_dict ):
  add_variables( arch_dict[VARIABLES], data_dict )
  return arch_dict[NETWORK]( arch_dict, data_dict)
  
# def load_architectures( arches, data ):
#   networks = OrderedDict()
#   for arch in arches:
#     networks[ arch[NAME] ] = load_architecture( arch, data )
#   return networks

def main(yaml_file):
  y = load_yaml( yaml_file)
  
  logging_dict = {}
  #print "Loading data"
  load_data_from_dict( y[DATA] )
  algo_dict = y[ALGORITHM]
  arch_dict = y[ARCHITECTURE]
  data_dict = y[DATA] #{N_TRAIN:4000}
  logging_dict = y[LOGGING]
  logging_dict[SAVEDIR] = os.path.join( HOME_DIR, os.path.join( logging_dict[LOCATION], logging_dict[EXPERIMENT] ) )
  # #networks = load_architectures( y[ARCHITECTURES], y[DATA] )
  #add_variables( arch_dict[VARIABLES], data_dict )
  network = load_architecture( arch_dict, data_dict )
  network_name = arch_dict[NAME]

  #
  
  # make BATCHER and reassign it to dict
  algo_dict[BATCHER] = algo_dict[BATCHER]( network_name, None, data_dict, algo_dict, arch_dict, logging_dict )


  algo_dict[BATCHER].network_name   = network_name
  algo_dict[BATCHER].network        = network
  
  sess = tf.InteractiveSession()
  
  results_dict = {}
  train( sess, network, algo_dict, data_dict, logging_dict, results_dict )
  batcher = algo_dict[BATCHER]
  batcher.CloseAll()
  return batcher
  
  
######################################################################################################
if __name__ == "__main__":
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_file = sys.argv[1]
  print "Running: ",yaml_file
  
  
    
  batcher = main( yaml_file )
  try:
    fill = batcher.fill_source_store
  except:
    fill = batcher.fill_store
  fill.open()
  Z_train = fill["/Z/TRAIN/Z/mu"].loc[ batcher.train_barcodes]
  Z_val = fill["/Z/VAL/Z/mu"].loc[ batcher.validation_barcodes]
  
  print Z_train
  fill.close()

  
  
  