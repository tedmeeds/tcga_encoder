import sys, os, pdb


from gene_fasta import *

def get_exon_ticks( exons ):
  if len(exons) == 0:
    return []
    
  u_exons = np.unique(exons)
  #ax = pp.axes()
  x_ticks = []
  for exon_id in u_exons:
    I = pp.find( exons==exon_id )
    x_ticks.append(I[0])
    
    #pp.fill_between_x( I[0],I[1])
    
  #ax.set_yticks(x_ticks, minor=False)
  #ax.yaxis.grid(True, which='major')
  x_ticks.append(I[-1])
  return x_ticks
  
def plot_stem( x, linefmt = "b-", markerfmt='bo' ):
  
  I = pp.find(x>0)
  if len(I) > 0:
    s = pp.stem( I, x[I], linefmt=linefmt, markerfmt=markerfmt )
  else:
    s = pp.vlines( 0, 0, len(x), color='k')
  return s
  
def process_mutations( gene, d, assembly2fasta, filter_tissue = None, R = None ):
  if d is None:
    print "** nothing to process"
    return
    
  if R is not None:
    assert len(R)  == len(d)
    
  n,n_columns = d.shape
  
  cols = d.columns
  v = d.values
  
  r_values = []
  if R is not None:
    r_values = np.log(1+R.values)
    
  mut_seqs = []
  sequences = []
  tissues = []
  barcodes = []
  rs=[]
  for i in range(n):
    vi = v[i]
    
    assembly = vi[-1]
    
    if assembly == '36':
      continue
    f = assembly2fasta[assembly]
    
    #start_idx = vi[1]
    #end_idx = vi[2]
    #pdb.set_trace()
    sequence = None
    tissue = vi[0]
    barcode = vi[1]
    r_value = 0
    if R is not None:
      r_value = r_values[i]
    #print tissue, barcode
    # if filter_tissue.upper() == tissue.upper():
    #   print vi
    #   pdb.set_trace()
    if filter_tissue is not None:
      #pdb.set_trace()
      if filter_tissue.upper() == tissue.upper():
        sequence, mut_sq = f.ExtractSequence( gene, vi )
    else:
      sequence, mut_sq = f.ExtractSequence( gene, vi )
      
    if sequence is not None:
      mut_seqs.append(mut_sq)
      sequences.append(sequence)
      tissues.append(tissue)
      barcodes.append(barcode)
      rs.append(r_value)
  #pdb.set_trace()
  d2 = d.loc[ barcodes ]  
  #d2["R"] = np.array(rs)
  return d2, tissues, barcodes,sequences, np.array( mut_seqs ), np.array(rs)

def load_assemblies( gene ):
  
  assembly2fasta = OrderedDict()
  try:
    fasta_dir = os.path.join( os.environ["HOME"], "data/human_genome/assembly_%d_fasta_process"%36 )
    qtf_dir   = os.path.join( os.environ["HOME"], "data/human_genome/assembly_%d_gtf_process"%36 )
    fasta_file = fasta_dir + "/%s/sequences.fa"%(gene)
    qtf_file = qtf_dir + "/%s/annotation.gtf"%(gene)
    assembly2fasta['36'] = GeneSequences( fasta_file, qtf_file, 36 )
  except:
    print "Could not load 36"

  fasta_dir = os.path.join( os.environ["HOME"], "data/human_genome/assembly_%d_fasta_process"%37 )
  qtf_dir   = os.path.join( os.environ["HOME"], "data/human_genome/assembly_%d_gtf_process"%37 )
  fasta_file = fasta_dir + "/%s/sequences.fa"%(gene)
  qtf_file = qtf_dir + "/%s/annotation.gtf"%(gene)

  assembly2fasta['37'] = GeneSequences( fasta_file, qtf_file, 37 )

  fasta_dir = os.path.join( os.environ["HOME"], "data/human_genome/assembly_%d_fasta_process"%38 )
  qtf_dir   = os.path.join( os.environ["HOME"], "data/human_genome/assembly_%d_gtf_process"%38 )
  fasta_file = fasta_dir + "/%s/sequences.fa"%(gene)
  qtf_file = qtf_dir + "/%s/annotation.gtf"%(gene)

  assembly2fasta['38'] = GeneSequences( fasta_file, qtf_file, 38 )
  assembly2fasta['hg19'] = assembly2fasta['37']
  return assembly2fasta

def load_mutation_data( gene, assembly2fasta, data_location, tissue = None, R = None ):
  mut_file = os.path.join( os.environ["HOME"], "%s/%s/dna.h5"%(data_location,gene) )
  #
  #try:
  d = pd.read_hdf( mut_file )
  
  d=d.set_index(d["patient.bcr_patient_barcode"].values )
  
  #except:
  #  d = None
  #  print "Could not load: %s"%(mut_file)
  
  tissues = None
  barcodes = None
  
  if d is not None:  
    d2,tissues,barcodes,s,ms,rs = process_mutations( gene, d, assembly2fasta, tissue )
  else:
    s = None; ms = None; rs=None
  return d2,tissues,barcodes,d, s, ms,rs
  
  
def load_genes(gene_list, data_location ):
  gene2mutations = OrderedDict()
  
  
  
  for gene in gene_list:
    assembly2fasta = load_assemblies( gene )
    tissues,barcodes,mut_data, seq, muts = load_mutation_data( gene, assembly2fasta, data_location )
    
    if barcodes is None:
      continue
      
    used_tissues  = []
    used_barcodes = []
    used_muts     = []
    barcode = barcodes[0]
    tissue  = tissues[0]
    m = 0*muts[0]
    for idx, bc in zip( xrange(len(tissues)), barcodes ):
      if bc == barcode:
        m += muts[idx]
        
      else:
        used_tissues.append( tissue )
        used_barcodes.append( barcode )
        used_muts.append(m)
        
        barcode = bc
        tissue = tissues[idx]
        m = muts[idx]
    used_tissues.append( tissue )
    used_barcodes.append( barcode )
    used_muts.append(m)
        
    
    gene2mutations[gene] = {"tissues":np.array(used_tissues,dtype=str), "barcodes":np.array(used_barcodes,dtype=str), "mutations":np.array(used_muts,dtype=int)}
  return  gene2mutations

def plot_positions_at_group( ax, group, seq, ms, x_ticks, s, colors ):
  #ax=pp.subplot(n_groups+1,1,group_idx+2)
  variant_dx = 0
  legs=[]
  for variant in group:
    if ms[:,seq.variant2idx[variant],:].sum() > 0:
      plot_stem( ms[:,seq.variant2idx[variant],:].sum(0), linefmt=colors[variant_dx]+'-', markerfmt=colors[variant_dx]+'o' )
      legs.append(group[variant_dx])
    variant_dx+=1
  ax.set_xticks(x_ticks, minor=False)
  ax.set_yticks(ax.get_yticks(), minor=False)
  ax.set_xticklabels( x_ticks, fontsize=8, rotation='vertical' )
  ax.xaxis.grid(True, which='major')
  pp.legend(legs)
  pp.xlim(0,len(s[0]))

def plot_positions( f, ms, x_ticks, gene, seq, s, plot_all = False, colors=None, tissue = None, groups = None, figsize=(14,10), save_location = None, R = None ):
  if groups is None:
    groups = [['Silent'],['Missense_Mutation'],['Nonsense_Mutation','Nonstop_Mutation'],['In_Frame_Del','In_Frame_Ins'],['Frame_Shift_Del','Frame_Shift_Ins'],['Splice_Site','RNA']]
  
  if colors is None:
    colors = ["b","r","g","k"]
  
  fig = pp.figure( figsize=figsize )
  n_groups = len(groups)
  
  
  if plot_all:
    ax0=fig.add_subplot(n_groups+1,1,1)
    #plot_exons( exons )
    plot_stem( ms.sum(1).sum(0), linefmt='b-', markerfmt='bo' )
    ax0.set_xticks(x_ticks, minor=False)
    ax0.set_yticks(ax0.get_yticks(), minor=False)
    ax0.set_xticklabels( x_ticks, fontsize=8, rotation='vertical' )
    ax0.xaxis.grid(True, which='major')
    ax0.xlim(0,len(s[0]))
    ax0.legend(['ALL'])
  
  for group_idx in range(n_groups):
    if plot_all:
      ax_g=fig.add_subplot(n_groups+1,1,group_idx+2)
    else:
      ax_g=fig.add_subplot(n_groups,1,group_idx+1)
    plot_positions_at_group( ax_g, groups[group_idx], seq, ms, x_ticks, s, colors )

  if tissue is not None:
    pp.suptitle( "%s filtered by %s, n=%d"%(gene, tissue,ms.sum()))
  else:
    pp.suptitle("%s, n=%d"%(gene,ms.sum()))
   
  if save_location is not None: 
    if tissue is None:
      pp.savefig( save_location + "/%s_mutations.png"%gene, fmt="png" )
    else:
      pp.savefig( save_location + "/%s_%s_mutations.png"%(gene,tissue), fmt="png" )
  pp.show()
  return fig
        
def main( gene, assembly = 37, \
                tissue = None, \
                save_location = None, \
                data_location = "data/broad_firehose/stddata__2016_01_28_processed_new/20160128/DNA_by_gene_small",\
                colors = None, \
                groups = None, \
                plot_all=False, \
                R = None) : 
  fasta_dir = os.path.join( os.environ["HOME"], "data/human_genome/assembly_%d_fasta_process"%assembly )
  qtf_dir   = os.path.join( os.environ["HOME"], "data/human_genome/assembly_%d_gtf_process"%assembly )
  
  fasta_file = fasta_dir + "/%s/sequences.fa"%(gene)
  qtf_file = qtf_dir + "/%s/annotation.gtf"%(gene)
  
  f = GeneSequences( fasta_file, qtf_file, assembly )
  
  assembly2fasta = load_assemblies(gene)

  #data_location = "data/broad_firehose/stddata__2016_01_28_processed_new/20160128/DNA_by_gene_small"
  d2,a,b,d,s,ms,rs = load_mutation_data( gene, assembly2fasta, data_location, tissue, R=R )

  if d is not None:
    #d2,a,b,s,ms = process_mutations( gene, d, assembly2fasta, tissue, R=R )
    # try:
    f = assembly2fasta['37']
    seq = f.hugo_transcript2fasta[gene+"-001"]
    exons = f.hugo_transcript2fasta[gene+"-001"].genome_exon_idx
    x_ticks = get_exon_ticks( exons )
    
    #pdb.set_trace()
    if len(x_ticks) == 0:
      f = GeneSequences( fasta_file, qtf_file, 36 )
  
      assembly2fasta = load_assemblies(gene)
      
      #data_location = "data/broad_firehose/stddata__2016_01_28_processed_new/20160128/DNA_by_gene_small"
      d2,a,b,d,s,ms = load_mutation_data( gene, assembly2fasta, data_location, tissue )
      if d is not None: 
        d2,a,b,s,ms = process_mutations( gene, d, assembly2fasta, tissue )
        f = assembly2fasta['36']
        seq = f.hugo_transcript2fasta[gene+"-001"]
        exons = f.hugo_transcript2fasta[gene+"-001"].genome_exon_idx
        x_ticks = get_exon_ticks( exons )

      
  #plot_positions()
  if R is None:
    fig = plot_positions( f, ms, x_ticks, gene, seq, s, plot_all = plot_all, colors=colors, tissue = tissue, groups = groups, save_location = save_location ) 
  else:
    fig = plot_positions( f, ms, x_ticks, gene, seq, s, plot_all = plot_all, colors=colors, tissue = tissue, groups = groups, save_location = save_location, rs=rs ) 
    
    
  return a,b,d2,s,ms,f,seq,exons,x_ticks, fig
    
  
if __name__ == "__main__":
  gene = sys.argv[1]
  assembly = int(sys.argv[2])
  if len(sys.argv)>3:
    tissue = sys.argv[3]
    print "*** Filtering tissue %s"%(tissue)
  else:
    tissue = None  
  print "*** Analyzing GENE = %s using assembly %d"%(gene,assembly)
  
  assert False, "do not call this way, call main() directly (needs to be completed)"
  

  fasta_dir = os.path.join( os.environ["HOME"], "data/human_genome/assembly_%d_fasta_process"%assembly )
  qtf_dir   = os.path.join( os.environ["HOME"], "data/human_genome/assembly_%d_gtf_process"%assembly )
  
  fasta_file = fasta_dir + "/%s/sequences.fa"%(gene)
  qtf_file = qtf_dir + "/%s/annotation.gtf"%(gene)
  
  f = GeneSequences( fasta_file, qtf_file, assembly )
  
  assembly2fasta = load_assemblies(gene)

  data_location = "data/broad_firehose/stddata__2016_01_28_processed_new/20160128/DNA_by_gene_small"
  a,b,d,s,ms = load_mutation_data( gene, assembly2fasta, data_location, tissue )
  # 
  #
  # mut_file = os.path.join( os.environ["HOME"], "data/broad_firehose/stddata__2016_01_28_processed_new/20160128/DNA_by_gene_small/%s/dna.h5"%gene )
  # #
  # try:
  #   d = pd.read_hdf( mut_file )
  # except:
  #   d = None
  #   print "Could not load: %s"%(mut_file)


  groups = [['Silent'],['Missense_Mutation'],['Nonsense_Mutation','Nonstop_Mutation'],['In_Frame_Del','In_Frame_Ins'],['Frame_Shift_Del','Frame_Shift_Ins'],['Splice_Site','RNA']]
  colors = ["b","r","g","k"]
  if d is not None:
    a,b,s,ms = process_mutations( gene, d, assembly2fasta, tissue )
    try:
      f = assembly2fasta['37']
      seq = f.hugo_transcript2fasta[gene+"-001"]
      exons = f.hugo_transcript2fasta[gene+"-001"].genome_exon_idx
      x_ticks = get_exon_ticks( exons )
    except:
      f = assembly2fasta['36']
      seq = f.hugo_transcript2fasta[gene+"-001"]
      exons = f.hugo_transcript2fasta[gene+"-001"].genome_exon_idx
      x_ticks = get_exon_ticks( exons )
    pp.figure( figsize=(14,10))
    n_groups = len(groups)
    
    
    ax=pp.subplot(n_groups+1,1,1)
    #plot_exons( exons )
    plot_stem( ms.sum(1).sum(0), linefmt='b-', markerfmt='bo' )
    ax.set_xticks(x_ticks, minor=False)
    ax.set_yticks(ax.get_yticks(), minor=False)
    ax.set_xticklabels( x_ticks, fontsize=8, rotation='vertical' )
    ax.xaxis.grid(True, which='major')
    pp.xlim(0,len(s[0]))
    pp.legend(['ALL'])
    
    for group_idx in range(n_groups):
      ax=pp.subplot(n_groups+1,1,group_idx+2)
      variant_dx = 0
      legs=[]
      for variant in groups[group_idx]:
        if ms[:,seq.variant2idx[variant],:].sum() > 0:
          plot_stem( ms[:,seq.variant2idx[variant],:].sum(0), linefmt=colors[variant_dx]+'-', markerfmt=colors[variant_dx]+'o' )
          legs.append(groups[group_idx][variant_dx])
        variant_dx+=1
      ax.set_xticks(x_ticks, minor=False)
      ax.set_yticks(ax.get_yticks(), minor=False)
      ax.set_xticklabels( x_ticks, fontsize=8, rotation='vertical' )
      ax.xaxis.grid(True, which='major')
      pp.legend(legs)
      pp.xlim(0,len(s[0]))
      

    if tissue is not None:
      pp.suptitle( "%s filtered by %s, n=%d"%(gene, tissue,ms.sum()))
    else:
      pp.suptitle("%s, n=%d"%(gene,ms.sum()))
      
    pp.savefig( "%s_mutations.png"%gene, fmt="png" )
    pp.show()
  
  