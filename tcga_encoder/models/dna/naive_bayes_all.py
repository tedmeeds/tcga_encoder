from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from tcga_encoder.models.dna.models import *
#from tcga_encoder.models.dna.naive_bayes_by_gene import main
import os 

disease_groups = [
  ["acc"],
  ["blca"],
  ["brca"],
  ["cesc"],
  ["chol"],
  ["coad" ,"read"],
  ["dlbc"],
  ["esca"],
  ["gbm","lgg"],
  ["hnsc"],
  ["kich","kirc","kirp"],
  ["lihc"],
  ["luad"],
  ["lusc"],
  ["ov"],
  ["paad"],
  ["pcpg"],
  ["prad"],
  ["sarc"],
  ["skcm"],
  ["stad"],
  ["tgct"],
  ["thca"],
  ["thym"],
  ["ucec"],
  ["ucs"],
  ["uvm"]
]

if __name__ == "__main__":
  #assert len(sys.argv) >= 2, "Must pass yaml file."
  data_file        = sys.argv[1]
  results_location = sys.argv[2]
  #dna_gene         = sys.argv[3]
  source           = sys.argv[3]
  method           = sys.argv[4]
  
  
  n_folds        = 4   # nbr of folds per xval repeat
  n_xval_repeats = 5   # nbr of xval permutations/repeats to try
  n_permutations = 10  # nbr of random label assignments to try
  
  if len(sys.argv) >= 8:
    n_folds        = int( sys.argv[5] )
    n_xval_repeats = int( sys.argv[6] )
    n_permutations = int( sys.argv[7] )
    
  #train = False
  #if len(sys.argv) == 10:
  train = True #bool(int( sys.argv[9]))
    
  data_store = pd.HDFStore( data_file, "r" )
  sources = [DNA, source]
  observed = data_store["/CLINICAL/observed"][ sources ] 
  barcodes = observed[ observed.sum(1)==len(sources) ].index.values
  
  variant = "Missense_Mutation"
  dna_data    = data_store["/DNA/channel/0"].loc[ barcodes ]
  data_store.close()
  dna_genes = dna_data.columns
  # restricted_diseases = []
  # idx = 10
  # while len(sys.argv) > idx:
  #   restricted_diseases.append( sys.argv[idx] )
  #   idx += 1
  
  #dna_gene = "APC"
  for dna_gene in dna_genes[15:20]: #100]:
    for restricted_diseases in disease_groups[:2]:
      print "-----------------------------------------------------------------"
      print "running ", dna_gene, " on ", restricted_diseases 
      print "-----------------------------------------------------------------"
      diseases = restricted_diseases[0]
      for d in restricted_diseases[1:]:
        diseases += " %s"%(d)
      #try:
      s = "python tcga_encoder/models/dna/naive_bayes_by_gene.py %s %s %s %s %s %d %d %d 1 %s"%(data_file,results_location,dna_gene,source,method,n_folds,n_xval_repeats,n_permutations,diseases)
      os.system(s)
      #  #main( data_file, results_location, dna_gene, source, method, n_folds, n_xval_repeats, n_permutations, train, restricted_diseases )
      #except:
      #  print "PROBLEM, skipping..."
    #main( data_file, results_location, dna_gene, source, method, n_folds, n_xval_repeats, n_permutations, train, restricted_diseases )
  #pdb.set_trace()