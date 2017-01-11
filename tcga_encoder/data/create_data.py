
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
  
  #os.path.join(broad_location,RNA)
  h5 = ReadH5( os.path.join( broad_location, spec["data_store"]) )
  
  return h5, genes2keep 
  
def process_rna( spec, broad_location ):
  return None, None 
  
def process_meth( spec, broad_location ):
  return None, None 
  
def process_clinical( spec, broad_location ):
  h5 = ReadH5( os.path.join( broad_location, spec["data_store"]) )
  return h5
  
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
  clinical_h5 = process_clinical( y["sources"][CLINICAL], broad_location )
  dataset.AddClinical( broad_location, y["sources"][CLINICAL], clinical_h5, diseases = diseases )
  
  for source_name in y["sources"]:
    source_spec = y["sources"][source_name]
    if source_name == DNA:
      dna_h5, dna_genes = process_dna( source_spec, broad_location )
      mutation_channels = source_spec["mutation_channels"]
      dataset.AddDNA( broad_location, source_spec["data_store"], dna_h5, mutation_channels=mutation_channels, genes2keep=dna_genes )
    elif source_name == RNA:
      rna_h5, rna_genes = process_rna( source_spec, broad_location )
      #dataset.AddRNA( rna_h5, rna_genes )
    elif source_name == METH:
      meth_h5, meth_genes = process_meth( source_spec, broad_location )
      #dataset.AddMeth( meth_h5, meth_genes )
    elif source_name == CLINICAL:
      pass
      
    else:
      assert False, "Unknown source %s"%(source_name)

  print dna_genes
  print dna_h5
  # b_save = True
  # filter_observed_only = True
  # diseases = None #["acc","brca","blca","cesc","chol"]
  #
  # n_filtered        = 1000
  # filter_column     = "pan-min_r"
  # filtered_csv_file = 'data/broad_firehose/analyses__2016_01_28_processed_new/20160128/MUT_SIG/PAN30_sig_genes.csv'
  # filtered_csv_file = os.path.join( os.environ.get('HOME','/'),filtered_csv_file)
  #
  # genes2keep = None
  # if filtered_csv_file is not None:
  #   genes2keep = load_gene_filter( filtered_csv_file, filter_column, n_filtered )
  #
  # #genes2keep = genes2keep[:10]
  #
  # if diseases is not None:
  #   save_location = os.path.join( os.environ.get('HOME','/'), 'data/broad_processed_new/20160128/pan_%d_tissue_rna_dna_meth_nfiltered_%d'%(len(diseases),n_filtered))
  # else:
  #   #save_location = os.path.join( os.environ.get('HOME','/'), 'data/broad_processed_new/20160128/pan_all_tissue_rna_dna_meth_nfiltered_%d'%(n_filtered))
  #   save_location = os.path.join( os.environ.get('HOME','/'), 'data/broad_processed_new/20160128/pan_all_tissue_rna_nfiltered_%d'%(n_filtered))
  #
  # dataset = MultiSourceData( save_location )
  #
  #
  # #save_location = os.path.join( os.environ.get('HOME','/'), 'data/broad_processed/20160128/pan_big')
  # if os.path.exists(save_location) is False:
  #   os.makedirs(save_location)
  #
  # broad_location = os.path.join( os.environ.get('HOME','/'), 'data/broad_firehose/stddata__2016_01_28_processed_new/20160128/')
  #
  # #dna_name = "gdac.broadinstitute.org_PAN25.Mutation_Packager_Calls.Level_3.2016012800.0.0.h5"
  # #rna_name = "gdac.broadinstitute.org_PAN26.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0.h5"
  # #meth_name = "gdac.broadinstitute.org_PAN26.Methylation_Preprocess.Level_3.2016012800.0.0.h5"
  # #clinical_name = "gdac.broadinstitute.org_PAN27.Merge_Clinical.Level_1.2016012800.0.0.h5"
  #
  # dna_name = "gdac.broadinstitute.org_PAN32.Mutation_Packager_Calls.Level_3.2016012800.0.0.h5"
  # rna_name = "gdac.broadinstitute.org_PAN33.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0.h5"
  # meth_name = "gdac.broadinstitute.org_PAN33.Methylation_Preprocess.Level_3.2016012800.0.0.h5"
  # clinical_name = "gdac.broadinstitute.org_PAN33.Merge_Clinical.Level_1.2016012800.0.0.h5"
  #
  #
  # variant_filters = {"mutation_channels":[ \
  #                                     ["Missense_Mutation"],\
  #                                     ["Nonsense_Mutation","Nonstop_Mutation"],\
  #                                     ["Frame_Shift_Del","Frame_Shift_Ins","In_Frame_Del","In_Frame_Ins"],\
  #                                     ["Silent"]\
  #                                 ]}
  #
  #
  # dataset.AddClinical( os.path.join(broad_location,CLINICAL), clinical_name, diseases = diseases )
  # #dataset.AddDNA( os.path.join(broad_location,DNA), dna_name, variant_filters["mutation_channels"], genes2keep, diseases = diseases )
  # dataset.AddRNA( os.path.join(broad_location,RNA), rna_name, genes2keep, diseases = diseases )
  # #dataset.AddMeth( os.path.join(broad_location,METH), meth_name, genes2keep, diseases = diseases )
  
  print "*****************************************"
  print "**                                     **"
  print "**    CREATE DATA: END                **"
  print "**                                     **"
  print "*****************************************"

