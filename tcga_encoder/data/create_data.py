
import tcga_encoder as p
print p
#from tcga_encoder import *
#from p import utils, data, definitions
from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.locations import *

def process_dna( spec, broad_location ):
  filter_nbr = spec["filter_nbr"]
  filter_column     = spec["filter_column"]
  filtered_csv_file = spec["filter_file"]
  filtered_csv_file = os.path.join( os.environ.get('HOME','/'), filtered_csv_file)
  
  genes2keep = load_gene_filter( filtered_csv_file, filter_column, filter_nbr )
  
  h5 = ReadH5( os.path.join( broad_location, spec["data_store"]) )
  
  return h5, genes2keep 
  
  
if __name__ == "__main__":
  print "*****************************************"
  print "**                                     **"
  print "**    CREATE DATA: START               **"
  print "**                                     **"
  print "*****************************************"
  diseases = None # ie all of them
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_file = sys.argv[1]
  print "Running: ",yaml_file
  
  y = load_yaml( yaml_file)
  
  print y
  save_location = os.path.join( os.environ.get('HOME','/'), y["data_store"]["location"] )
  print "HOME:", os.environ.get('HOME','/')
  print "SAVE:",save_location
  dataset = MultiSourceData( save_location )
  broad_location = os.path.join( os.environ.get('HOME','/'), y["broad_location"] )

  # add CLINICAL first
  print "Loading CLINICAL"
  clinical_h5 =ReadH5( os.path.join( broad_location, y["sources"][CLINICAL]["data_store"]) )
  dataset.AddClinical( broad_location, y["sources"][CLINICAL], clinical_h5, diseases = diseases )
  
  for source_name in y["sources"]:
    source_spec = y["sources"][source_name]
    if source_name == DNA:
      dna_h5, dna_genes = process_dna( source_spec, broad_location )
      mutation_channels = source_spec["mutation_channels"]
      min_nbr_in_pan  = None
      if source_spec.has_key("min_nbr_in_pan"):
        min_nbr_in_pan = source_spec["min_nbr_in_pan"]
      dataset.AddDNA( broad_location, source_spec["data_store"], dna_h5, mutation_channels=mutation_channels, genes2keep=dna_genes, min_nbr_in_pan=min_nbr_in_pan )
    elif source_name == RNA:
      rna_h5 = ReadH5( os.path.join( broad_location, source_spec["data_store"]) )
      nbr = source_spec["nbr"]
      method = source_spec["method"]
      dataset.AddRNA( broad_location, source_spec["data_store"], rna_h5, nbr, method )
    elif source_name == miRNA:
      mirna_h5 = ReadH5( os.path.join( broad_location, source_spec["data_store"]) )
      nbr = source_spec["nbr"]
      method = source_spec["method"]
      dataset.AddmiRNA( broad_location, source_spec["data_store"], mirna_h5, nbr, method )
    elif source_name == METH:
      meth_h5 = ReadH5( os.path.join( broad_location, source_spec["data_store"]) )
      nbr = source_spec["nbr"]
      method = source_spec["method"]
      dataset.AddMeth( broad_location, source_spec["data_store"], meth_h5, nbr, method )
    elif source_name == CLINICAL:
      pass
    else:
      assert False, "Unknown source %s"%(source_name)

  print "*****************************************"
  print "**                                     **"
  print "**    CREATE DATA: END                **"
  print "**                                     **"
  print "*****************************************"

