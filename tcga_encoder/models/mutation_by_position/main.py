import sys, os
#print sys.path 

sys.path.insert(0, os.getcwd())
print sys.path 
from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
#from tcga_encoder.algorithms import *

from tcga_encoder.data.positions.process_gene_mutation_sequence import main as position_view
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("talk")

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == "__main__":
  
  print sys.path
  gene = sys.argv[1]
  assembly = int(sys.argv[2])
  if len(sys.argv)>3:
    tissue = sys.argv[3]
    #print "*** Filtering tissue %s"%(tissue)
  else:
    tissue = None  
  
  data_location = "data/broad_firehose/stddata__2016_01_28_processed_new/20160128/DNA_by_gene_small"
  #data_location = "data/broad_firehose/stddata__2016_01_28_processed_new/20160128/DNA_by_gene"
  save_location = os.path.join( HOME_DIR,  "results/tcga_position_mutations"  )
  check_and_mkdir(save_location)
  
  a,b,d,s,ms = position_view( gene, assembly, tissue, save_location = save_location, data_location = data_location)
  
  