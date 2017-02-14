import sys, os
#print sys.path 

sys.path.insert(0, os.getcwd())
#print sys.path 
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
  assembly = int(sys.argv[2])
  if len(sys.argv)>3:
    tissue = sys.argv[3]
    if tissue == "none":
      tissue = None
    #print "*** Filtering tissue %s"%(tissue)
  else:
    tissue = None  
  
  if len(sys.argv)>4:
    target = sys.argv[4]
    #print "*** Filtering tissue %s"%(tissue)
  else:
    target = "ADAM6"  
  
  
  data_location = "data/broad_firehose/stddata__2016_01_28_processed_new/20160128/DNA_by_gene_small"
  #data_location = "data/broad_firehose/stddata__2016_01_28_processed_new/20160128/DNA_by_gene"
  save_location = os.path.join( HOME_DIR,  "results/tcga_position_mutations"  )
  check_and_mkdir(save_location)
  
  groups = [['Silent'],['Missense_Mutation'],['Nonsense_Mutation','Nonstop_Mutation'],['In_Frame_Del','In_Frame_Ins'],['Frame_Shift_Del','Frame_Shift_Ins'],['Splice_Site','RNA']]

  #rna_data_location = os.path.join( HOME_DIR,  "data/broad_processed_post_recomb/20160128/pan_tiny_multi_set" )
  rna_data_location = os.path.join( HOME_DIR,  "data/broad_processed_post_recomb/20160128/pan_small_multi_set" )
  
  rna = pd.HDFStore( rna_data_location + "/data.h5" )
  

    
  
  a,b,d,s,ms,f,seq,exons,x_ticks, fig = position_view( gene, assembly, tissue, save_location = save_location, data_location = data_location, groups=groups )
  

  disease_barcodes = []
  #d = d.loc[b]
  d_bc = d[ ["admin.disease_code","patient.bcr_patient_barcode"] ].values
  for x in d_bc:
    disease_barcodes.append( "%s_%s"%(x[0],x[1]))
  RSEM = rna["/RNA/RSEM"].loc[ disease_barcodes ].fillna(0)
  RSEM_T = RSEM[target].fillna(0)
  R = RSEM_T.values
  
  all_locs = []
  all_r = []
  for i in range( len(ms) ):
    
    rna_val = RSEM_T.loc[ "%s_%s"%(a[i],b[i])]
    
    if rna_val.__class__ == pd.core.series.Series:
      rna_val = rna_val.values[0]
    print a[i],b[i],rna_val
      #rna_val = R[i]
    
    locs = pp.find(ms[i].sum(0))
  
    for loc in locs:
      all_locs.append( loc )
      all_r.append( rna_val )

  
  all_locs = np.array(all_locs)
  all_r = np.array(all_r)
    
  fig2 = pp.figure( figsize=(14,5))
  ax2 = fig2.add_subplot(211)
  ax2.semilogy( all_locs, all_r, "bo", mec='k',mew=1, ms=5, alpha=0.5)
  ax2.set_xticks(x_ticks, minor=False)
  ax2.set_yticks(ax2.get_yticks(), minor=False)
  ax2.set_xticklabels( x_ticks, fontsize=8, rotation='vertical' )
  pp.xlim(0,len(s[0]))
  pp.title( "Mutations = %s  Target = %s"%(gene,target))
  ax2 = fig2.add_subplot(212)
  ax2.plot( all_locs, all_r, "bo", mec='k',mew=1, ms=5, alpha=0.5)
  ax2.set_xticks(x_ticks, minor=False)
  ax2.set_yticks(ax2.get_yticks(), minor=False)
  ax2.set_xticklabels( x_ticks, fontsize=8, rotation='vertical' )
  pp.xlim(0,len(s[0]))
  #pp.title( "Mutations = %s  Target = %s"%(gene,target))  
  if save_location is not None: 
    if tissue is None:
      pp.savefig( save_location + "/%s_mutations_target_%s.png"%(gene,target), fmt="png" )
    else:
      pp.savefig( save_location + "/%s_%s_mutations_target_%s.png"%(gene,tissue,target), fmt="png" )

  pp.show()
  
   
  
  #a,b,d,s,ms,f,seq,exons,x_ticks, fig = position_view( gene, assembly, tissue, save_location = save_location, data_location = data_location, groups=groups, R = RSEM_T )
  
  #fig = plot_positions( f, ms, x_ticks, gene, seq, s, plot_all = plot_all, colors=colors, tissue = tissue, groups = groups, save_location = save_location, R = RSEM_T ) 