
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
  h5_raw = ReadH5( os.path.join( broad_location, spec["data_store_raw"]) )
  
  print "process_dna: |genes2keep| = ", len(genes2keep)
  return h5, h5_raw, genes2keep 
  
  
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
  broad_analyses_location = os.path.join( os.environ.get('HOME','/'), y["broad_analyses_location"] )
  broad_processed_location= os.path.join( os.environ.get('HOME','/'), y["broad_processed_location"] )
  # add CLINICAL first
  print "Loading CLINICAL"
  clinical_h5 =ReadH5( os.path.join( broad_location, y["sources"][CLINICAL]["data_store"]) )
  dataset.AddClinical( broad_location, y["sources"][CLINICAL], clinical_h5, diseases = diseases )
  
  for source_name in y["sources"]:
    source_spec = y["sources"][source_name]
    if source_name == DNA:
      dna_h5, dna_h5_raw, dna_genes = process_dna( source_spec, broad_location )
      mutation_channels = source_spec["mutation_channels"]
      min_nbr_in_pan  = None
      if source_spec.has_key("min_nbr_in_pan"):
        min_nbr_in_pan = source_spec["min_nbr_in_pan"]
      dataset.AddDNA( broad_location, source_name, dna_h5, dna_h5_raw, mutation_channels=mutation_channels, genes2keep=dna_genes, min_nbr_in_pan=min_nbr_in_pan )
    elif source_name == RNA:
      rna_h5 = pd.HDFStore( os.path.join( broad_processed_location, source_spec["data_store"]),"r" )
      #print "loading ", source_spec["data_store_ga"]
      #rna_h5_ga = ReadH5( os.path.join( broad_location, source_spec["data_store_ga"]) )
      #print "loading ", source_spec["data_store_hi"]
      #rna_h5_hi = ReadH5( os.path.join( broad_location, source_spec["data_store_hi"]) )
      nbr = source_spec["nbr"]
      method = source_spec["method"]
      #dataset.AddRNA( broad_location, source_name, rna_h5_ga, rna_h5_hi, nbr, method )
      
      filter_column     = source_spec["filter_column"]
      filtered_csv_file = source_spec["filter_file"]
      filtered_csv_file = os.path.join( broad_analyses_location, filtered_csv_file)
      filter_nbr = spec["filter_nbr"]
      genes2keep = load_gene_filter( filtered_csv_file, filter_column, filter_nbr )
      
      #dataset.SelectiveAddRNA( broad_location, source_name, rna_h5_ga, rna_h5_hi, genes2keep )
      
      dataset.InitSource( RNA, broad_location, source_name )
      dataset.store[ RNA + "/" + "RSEM" + "/" ] = rna_h5[RNA + "/" + "RSEM" + "/"][ genes2keep ]
      self.store[ RNA + "/" + "FAIR" + "/" ] = rna_h5[RNA + "/" + "FAIR" + "/"][ genes2keep ]
      
      
    elif source_name == miRNA:
      #mirna_h5_ga = ReadH5( os.path.join( broad_location, source_spec["data_store_ga"]) )
      #mirna_h5_hi = ReadH5( os.path.join( broad_location, source_spec["data_store_hi"]) )
      mirna_h5 = pd.HDFStore( os.path.join( broad_processed_location, source_spec["data_store"]), "r" )
      #mirna_h5_hi = ReadH5( os.path.join( broad_location, source_spec["data_store_hi"]) )
      nbr = source_spec["nbr"]
      method = source_spec["method"]
      
      filter_column     = source_spec["filter_column"]
      filtered_csv_file = source_spec["filter_file"]
      filtered_csv_file = os.path.join( broad_analyses_location, filtered_csv_file )
      filter_nbr = spec["filter_nbr"]
      genes2keep = load_gene_filter( filtered_csv_file, filter_column, filter_nbr )
      
      #dataset.AddmiRNA( broad_location, source_name, mirna_h5_ga, mirna_h5_hi, nbr, method )
      #dataset.SelectiveAddmiRNA( broad_location, source_name, mirna_h5_ga, mirna_h5_hi, genes2keep )
      
      dataset.InitSource( miRNA, broad_location, source_name )
      dataset.store[ miRNA + "/" + "READS" + "/" ] = mirna_h5[miRNA + "/" + "READS" + "/"][ genes2keep ]
      self.store[ miRNA + "/" + "FAIR" + "/" ] = mirna_h5[miRNA + "/" + "FAIR" + "/"][ genes2keep ]
      
    elif source_name == METH:
      meth_h5 = pd.HDFStore( os.path.join( broad_processed_location, source_spec["data_store"]), "r" )
      nbr = source_spec["nbr"]
      method = source_spec["method"]
      
      filter_column     = source_spec["filter_column"]
      filtered_csv_file = source_spec["filter_file"]
      filtered_csv_file = os.path.join( broad_analyses_location, filtered_csv_file)
      filter_nbr = spec["filter_nbr"]
      genes2keep = load_gene_filter( filtered_csv_file, filter_column, filter_nbr )
      
      dataset.InitSource( METH, broad_location, source_name )
      dataset.store[ METH + "/" + "METH" + "/" ] = meth_h5[METH + "/" + "METH" + "/"][ genes2keep ]
      self.store[ METH + "/" + "FAIR" + "/" ] = meth_h5[METH + "/" + "FAIR" + "/"][ genes2keep ]

      #dataset.AddMeth( broad_location, source_name, meth_h5, nbr, method )
    elif source_name == CLINICAL:
      pass
    else:
      assert False, "Unknown source %s"%(source_name)

  print "*****************************************"
  print "**                                     **"
  print "**    CREATE DATA: END                **"
  print "**                                     **"
  print "*****************************************"

