from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
#from tcga_encoder.algorithms import *
import seaborn as sns



def add_variables( var_dict, data_dict ):
  # add very specific numbers:
  var_dict["dna_dim"]    = data_dict['dataset'].GetDimension("DNA")
  var_dict["meth_dim"]   = data_dict['dataset'].GetDimension("METH")
  var_dict["rna_dim"]    = data_dict['dataset'].GetDimension("RNA")
  var_dict["mirna_dim"]    = data_dict['dataset'].GetDimension("miRNA")
  var_dict["tissue_dim"] = data_dict['dataset'].GetDimension("TISSUE")
  
  if data_dict.has_key( "dna_genes"):
    var_dict["dna_dim"]  = len(data_dict["dna_genes"])
    
  
def load_architecture( arch_dict, data_dict ):
  add_variables( arch_dict[VARIABLES], data_dict )
  return arch_dict[NETWORK]( arch_dict, data_dict)
  

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
  #algo_dict[BATCHER] = algo_dict[BATCHER]( network_name, None, data_dict, algo_dict, arch_dict, logging_dict )


  #algo_dict[BATCHER].network_name   = network_name
  #algo_dict[BATCHER].network        = network
  
  #sess = tf.InteractiveSession()
  
  results_dict = {}
  network.train( algo_dict, logging_dict )
  #batcher = algo_dict[BATCHER]
  #batcher.CloseAll()
  return network