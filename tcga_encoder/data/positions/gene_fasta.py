from tcga_encoder.utils.helpers import *
#from tcga_encoder.definitions import *
import pandas as pd
from collections import OrderedDict


class FastaSequence(object):
  def __init__(self, hugo_gene, ensembl_gene, hugo_transcript, ensembl_transcript, length, sequence ):
    self.hugo_gene          = hugo_gene
    self.ensembl_gene       = ensembl_gene
    self.hugo_transcript    = hugo_transcript
    self.ensembl_transcript = ensembl_transcript
    self.length             = length
    self.sequence           = sequence
    self.exons_ok           = False
    self.exons              = OrderedDict()
    self.mapped             = False
    self.use_splice_location = True
    self.length_with_splice = self.length + 2*len(self.exons)
    
    assert length == len(sequence), "should be same length"
  
  def MapIndices( self ):
    if self.mapped is True:
      return
    
    self.length_with_splice = self.length + 2*len(self.exons)
      
    
    print "** mapping indices"
    if self.length_with_splice:
      self.genome_indices    = np.zeros( self.length_with_splice )
      self.genome_exon_idx   = np.zeros( self.length_with_splice )
      self.splice_site_idx   = np.zeros( self.length_with_splice )
    else:
      self.genome_indices    = np.zeros( self.length )
      self.genome_exon_idx   = np.zeros( self.length )
    
    relative_idx = 0
    exon_idx = 0
    for exon, exon_info in self.exons.iteritems():
       start_idx   = exon_info[1]
       end_idx     = exon_info[2]
       if self.length_with_splice:
         genome_idx  = start_idx - 1
         self.genome_indices[ relative_idx ]  = genome_idx
         self.genome_exon_idx[ relative_idx ] = exon_idx
         self.splice_site_idx[relative_idx]   = 1
         relative_idx+=1
         genome_idx+=1
       else:
         genome_idx  = start_idx
         
       while genome_idx <= end_idx:
         self.genome_indices[ relative_idx ] = genome_idx
         self.genome_exon_idx[ relative_idx ] = exon_idx
         genome_idx += 1
         relative_idx += 1
         
       if self.length_with_splice:
         self.genome_indices[ relative_idx ] = genome_idx
         self.genome_exon_idx[ relative_idx ] = exon_idx
         self.splice_site_idx[relative_idx]=1
         genome_idx += 1
         relative_idx += 1
       exon_idx+=1
   
    self.genome2relative_indices = OrderedDict()
    for i in xrange(self.length):
      self.genome2relative_indices[ self.genome_indices[i] ] = i
    self.mapped = True
    
  
  
  def MapMutationsToPositions(self, mutation_line, verbose = False ):
    if self.mapped is False:
      self.MapIndices()
    
    #print "** mapping mutations"
    sequence = bytearray( self.sequence )
    mut_sequence = np.zeros( self.length, dtype = int )
    
    # n_channels = 4
    # snp_idx             = 0
    # dnp_idx             = 1
    # frame_shift_ins_idx = 2
    # frame_shift_del_idx = 3
    
    self.variant2idx = OrderedDict()
    idx=0
    self.variant2idx[ "Silent" ] = idx; idx+=1
    self.variant2idx[ "Missense_Mutation" ] = idx; idx+=1
    self.variant2idx[ "Nonsense_Mutation" ] = idx; idx+=1
    self.variant2idx[ "Nonstop_Mutation" ] = idx; idx+=1
    self.variant2idx[ "In_Frame_Del" ] = idx; idx+=1
    self.variant2idx[ "In_Frame_Ins" ] = idx; idx+=1
    self.variant2idx[ "Frame_Shift_Del" ] = idx; idx+=1
    self.variant2idx[ "Frame_Shift_Ins" ] = idx; idx+=1
    self.variant2idx[ "Splice_Site" ] = idx; idx+=1
    self.variant2idx[ "RNA" ] = idx; idx+=1
    self.n_channels = len(self.variant2idx)
    
    mut_by_channels = np.zeros( (self.n_channels, self.length), dtype = int  )
    # array(['ucec', 'tcga-ax-a2hf', '112154723', '112154723', 'SNP', 'C', 'C',
    #   'T', 'Nonsense_Mutation', '37'], dtype=object)
    
    start_idx = int(mutation_line[2])
    end_idx   = int(mutation_line[3])
    allele2onehot = {'A':0,'T':1,'C':2,'G':3}
    try:
      relative_start_idx = self.genome2relative_indices[ start_idx ]
      relative_end_idx = self.genome2relative_indices[ end_idx ]
      
      
    except:
      
      
      #print "trying to find closest"
      k =  np.array(self.genome2relative_indices.keys())
      closest_start = k[np.argmin( np.abs( start_idx - k ) )]
      closest_end = k[np.argmin( np.abs( end_idx - k ) )]
      #print "moving %d to %d"%(start_idx,closest_start)
      dif = np.abs( start_idx - closest_start )
      start_idx = closest_start
      end_idx = closest_end
      relative_start_idx = self.genome2relative_indices[ start_idx ]
      relative_end_idx = self.genome2relative_indices[ end_idx ]
      #pdb.set_trace()
      
      if dif > 10000:
        print "!! mutation mapping problem !!"
        print "distance is too far..", dif
        print self.genome_indices
      
        print mutation_line
        return None, None
      
    tissue = mutation_line[0]
    barcode = mutation_line[1]  
    mut_type  = mutation_line[4]
    normal_allele    = mutation_line[5]
    variant_allele = mutation_line[6]
    variant = mutation_line[8]
    
    #pdb.set_trace()
    if mut_type == "SNP":
      
      if self.sequence[ relative_start_idx ] != normal_allele:
        if verbose:
          print " !! normal allele (%s) does not match sequence (%s) at position %d"%(normal_allele,self.sequence[ relative_start_idx ], start_idx)
          print mutation_line
        
      sequence[ relative_start_idx ] = variant_allele
      mut_sequence[relative_start_idx] += 1
      
      mut_by_channels[self.variant2idx[variant], relative_start_idx] += 1
    elif mut_type == "DNP":
      fasta_seq = self.sequence[ relative_start_idx:relative_start_idx+2 ]
      if fasta_seq != normal_allele:
        if verbose:
          print " !! normal allele (%s) does not match sequence (%s) at position %d"%(normal_allele,fasta_seq, start_idx)
          print mutation_line
        
      sequence[ relative_start_idx ] = variant_allele[0]
      sequence[ relative_start_idx+1 ] = variant_allele[1]
      mut_sequence[relative_start_idx] += 1
      mut_sequence[relative_start_idx+1] += 1
      
      mut_by_channels[self.variant2idx[variant], relative_start_idx] += 1
      #mut_by_channels[dnp_idx, relative_start_idx] += 1
      # else:
      #   print " !! normal allele (%s) does not match sequence (%s) at position %d"%(normal_allele,self.sequence[ relative_start_idx ], start_idx)
      #   print mutation_line
    elif mut_type == "DEL":
      #mut_by_channels[frame_shift_ins_idx, relative_start_idx] += 1
      mut_by_channels[self.variant2idx[variant], relative_start_idx] += 1
    elif mut_type == "INS":
      #mut_by_channels[frame_shift_del_idx, relative_start_idx] += 1
      mut_by_channels[self.variant2idx[variant], relative_start_idx] += 1
    elif mut_type == "Splice_Site":
      mut_by_channels[self.variant2idx[variant], relative_start_idx] += 1
    else:
      print "not processing this type", mutation_line
    
    return sequence, mut_by_channels
  
  def AddExon( self, exon_name, exon_nbr, start_idx, end_idx ):
    if exon_name is None:
      assert exon_nbr is None, "shouldnt know exon nbr"
      exon_nbr = len(self.exons)+1
      exon_name = self.hugo_transcript + "-%d"%exon_nbr
    self.exons[ exon_name ] = np.array( [exon_nbr, start_idx, end_idx, end_idx-start_idx+1] )
  
  def CheckExons(self, verbose = False):
    exon_length = 0
    for ex in self.exons.itervalues():
      exon_length += ex[-1]
    
    if exon_length == self.length:
      if verbose:
        print "%s !! Exons add up to length"%self.hugo_transcript
      self.exons_ok = True
    else:
      print "%s?? Exons DO NOT add up to length"%self.hugo_transcript, exon_length, self.length
  
  def ReportExons(self):
    print self.hugo_transcript, self.ensembl_transcript, " (%d) "%self.length
    print "-------------"
    for exon, exon_info in self.exons.iteritems():
      print exon, exon_info
    print ""
      
      


class GeneSequences( object ):
  def __init__( self, fasta_filename, qtf_filename, assembly, verbose = False ):
    self.fasta_filename  = fasta_filename
    self.qtf_filename    = qtf_filename
    self.assembly        = assembly
    
    self.LoadFasta( self.fasta_filename, self.assembly )
    self.LoadAnnotations( self.qtf_filename, self.assembly )
    self.CheckExons()
    if verbose:
      self.ReportExons()
  
  def ExtractSequence( self, gene, mutation_line ):
    # assume hugo_transcript
    hugo_transcript = gene + "-001"
    
    if self.hugo_transcript2fasta.has_key(hugo_transcript ):
      transcript = self.hugo_transcript2fasta[  hugo_transcript ]
    else:
      hugo_transcript = gene + "-201"
      transcript = self.hugo_transcript2fasta[  hugo_transcript ]
    transcript.MapIndices()
    
    return transcript.MapMutationsToPositions( mutation_line )

    
  
  def CheckExons(self):
    for transcript, sequence in self.hugo_transcript2fasta.iteritems():
      sequence.CheckExons()
  
  def ReportExons(self):
    for transcript, sequence in self.hugo_transcript2fasta.iteritems():
      sequence.ReportExons()
  
  def LoadFasta(self, filename, assembly ):
    print "** LOADING FASTA ** "
    self.hugo_transcript2fasta = OrderedDict()
    
    f = open( filename, "r")
    lines = f.readlines()
    f.close()
    
    idx = 0
    while idx < len(lines):
      s = lines[idx]
      
      assert s[0] == ">", "should be a new line"
      
      ss =s.split("|")
      ensembl_transcript = ss[0][1:]
      ensembl_gene       = ss[1]
      hugo_gene          = ss[5]
      hugo_transcript    = ss[4]
      length             = int(ss[6])
      
      #print hugo_transcript, length
      
      sequence = lines[idx+1].rstrip("\n")
      
      idx+=2
      self.hugo_transcript2fasta[ hugo_transcript ] = FastaSequence( hugo_gene, \
                                                                     ensembl_gene, \
                                                                     hugo_transcript, \
                                                                     ensembl_transcript, \
                                                                     length, \
                                                                     sequence)
  def LoadAnnotations(self, filename, assembly ):
    print "** LOADING QTF ** "
    
    # chr5	HAVANA	gene	112043195	112181936	.	+	.	gene_id "ENSG00000134982.12"; transcript_id "ENSG00000134982.12"; gene_type "protein_coding"; gene_status "KNOWN"; gene_name "APC"; transcript_type "protein_coding"; transcript_status "KNOWN"; transcript_name "APC"; level 2; tag "ncRNA_host"; havana_gene "OTTHUMG00000128806.6";
    
    #self.hugo_transcript2qtf = OrderedDict()
    
    f = open( filename, "r")
    lines = f.readlines()
    f.close()
    
    
    idx = 0
    while idx < len(lines):
      s = lines[idx]
      
      hugo_transcript, \
      exon_gene_etc, \
      exon_id, \
      exon_number, \
      start_idx, \
      end_idx = self.ProcessAnnotationLine( s, assembly )
      
      
      idx+=1
      
      if exon_gene_etc == "exon":
        if self.hugo_transcript2fasta.has_key(hugo_transcript):
          self.hugo_transcript2fasta[hugo_transcript].AddExon( exon_id, exon_number, start_idx, end_idx )
  
  def ProcessAnnotationLine( self, s, assembly ):
    if assembly == 36:
      return self.ProcessAssembly36( s )
    if assembly == 37:
      return self.ProcessAssembly37( s )
    if assembly == 38:
      return self.ProcessAssembly38( s )
  
  def ProcessAssembly36( self, s ):
    # chr5	ENSEMBL	exon	112071117	112071478	.	+	.	gene_id "ENSG00000134982"; transcript_id "ENST00000457016"; gene_type "protein_coding"; gene_status "KNOWN"; gene_name "APC"; transcript_type "protein_coding"; transcript_status "KNOWN"; transcript_name "APC-202"; level 3; tag "CCDS"; ccdsid "CCDS4107"; havana_gene "OTTHUMG00000128806";
    
    ss = s.split("\t")
    chrom = ss[0]  # eg chr5
    method = ss[1] # eg HAVANA
    exon_gene_etc = ss[2]
    
    if exon_gene_etc == "gene" or exon_gene_etc == "transcript":
      return None, exon_gene_etc, None, None, None, None
      
    start_idx = int(ss[3])
    end_idx = int(ss[4])
    strand = ss[6]
    
    length = end_idx-start_idx+1
    tt=ss[8].split(";")
    
    ensembl_gene       = tt[0].split(" ")[-1].rstrip('"').lstrip('"')
    ensembl_transcript = tt[1].split(" ")[-1].rstrip('"').lstrip('"')
    gene_type          = tt[2].split(" ")[-1].rstrip('"').lstrip('"')
    gene_status        = tt[3].split(" ")[-1].rstrip('"').lstrip('"')
    hugo_gene          = tt[4].split(" ")[-1].rstrip('"').lstrip('"')
    transcript_type    = tt[5].split(" ")[-1].rstrip('"').lstrip('"')
    transcript_status  = tt[6].split(" ")[-1].rstrip('"').lstrip('"')
    hugo_transcript    = tt[7].split(" ")[-1].rstrip('"').lstrip('"')
    
    exon_id = None
    exon_number = None
    # if exon_gene_etc== "exon":
    #   pdb.set_trace()
    return hugo_transcript, exon_gene_etc, exon_id, exon_number, start_idx, end_idx
  
  def ProcessAssembly37( self, s ):
      #chr12	HAVANA	exon	53709511	53709566	.	-	.	gene_id "ENSG00000094914.8"; transcript_id "ENST00000209873.4"; gene_type "protein_coding"; gene_status "KNOWN"; gene_name "AAAS"; transcript_type "protein_coding"; transcript_status "KNOWN"; transcript_name "AAAS-001"; exon_number 3; exon_id "ENSE00003656148.1"; level 2; protein_id "ENSP00000209873.4"; tag "basic"; tag "appris_principal"; tag "CCDS"; ccdsid "CCDS8856.1"; havana_gene "OTTHUMG00000169729.3"; havana_transcript "OTTHUMT00000405632.1";
    
    ss = s.split("\t")
    chrom = ss[0]  # eg chr5
    method = ss[1] # eg HAVANA
    exon_gene_etc = ss[2]
    if exon_gene_etc == "gene" or exon_gene_etc == "transcript":
      return None, exon_gene_etc, None, None, None, None
      
    start_idx = int(ss[3])
    end_idx = int(ss[4])
    strand = ss[6]
    
    length = end_idx-start_idx+1
    tt=ss[8].split(";")
    
    ensembl_gene       = tt[0].split(" ")[-1].rstrip('"').lstrip('"')
    ensembl_transcript = tt[1].split(" ")[-1].rstrip('"').lstrip('"')
    gene_type          = tt[2].split(" ")[-1].rstrip('"').lstrip('"')
    
    if gene_type == "snRNA":
      return None, exon_gene_etc, None, None, None, None
      
    gene_status        = tt[3].split(" ")[-1].rstrip('"').lstrip('"')
    hugo_gene          = tt[4].split(" ")[-1].rstrip('"').lstrip('"')
    transcript_type    = tt[5].split(" ")[-1].rstrip('"').lstrip('"')
    transcript_status  = tt[6].split(" ")[-1].rstrip('"').lstrip('"')
    hugo_transcript    = tt[7].split(" ")[-1].rstrip('"').lstrip('"')
    exon_or_level              = tt[8].rstrip(' ').lstrip(' ').split(" ")[0]
    exon_number              = int( tt[8].split(" ")[-1] )
    exon_id                = tt[9].split(" ")[-1].rstrip('"').lstrip('"')
    havana_gene        = tt[10].split(" ")[-1].rstrip('"').lstrip('"')
    
    return hugo_transcript, exon_gene_etc, exon_id, exon_number, start_idx, end_idx
  
  def ProcessAssembly38( self, s ):
      #chr17	HAVANA	exon	33529787	33530198	.	-	.	gene_id "ENSG00000265544.1"; transcript_id "ENST00000579745.1"; gene_type "sense_intronic"; gene_status "KNOWN"; gene_name "AA06"; transcript_type "sense_intronic"; transcript_status "KNOWN"; transcript_name "AA06-001"; exon_number 3; exon_id "ENSE00002726282.1"; level 2; transcript_support_level "1"; tag "basic"; havana_gene "OTTHUMG00000179663.1"; havana_transcript "OTTHUMT00000447557.1";

    
    ss = s.split("\t")
    chrom = ss[0]  # eg chr5
    method = ss[1] # eg HAVANA
    exon_gene_etc = ss[2]
    if exon_gene_etc == "gene" or exon_gene_etc == "transcript":
      return None, exon_gene_etc, None, None, None, None
    start_idx = int(ss[3])
    end_idx = int(ss[4])
    strand = ss[6]
    
    length = end_idx-start_idx+1
    tt=ss[8].split(";")
    
    ensembl_gene       = tt[0].split(" ")[-1].rstrip('"').lstrip('"')
    ensembl_transcript = tt[1].split(" ")[-1].rstrip('"').lstrip('"')
    gene_type          = tt[2].split(" ")[-1].rstrip('"').lstrip('"')
    gene_status        = tt[3].split(" ")[-1].rstrip('"').lstrip('"')
    hugo_gene          = tt[4].split(" ")[-1].rstrip('"').lstrip('"')
    transcript_type    = tt[5].split(" ")[-1].rstrip('"').lstrip('"')
    transcript_status  = tt[6].split(" ")[-1].rstrip('"').lstrip('"')
    hugo_transcript    = tt[7].split(" ")[-1].rstrip('"').lstrip('"')
    exon_or_level              = tt[8].rstrip(' ').lstrip(' ').split(" ")[0]
    exon_number              = int( tt[8].split(" ")[-1] )
    exon_id                = tt[9].split(" ")[-1].rstrip('"').lstrip('"')
    havana_gene        = tt[10].split(" ")[-1].rstrip('"').lstrip('"')
    
    return hugo_transcript, exon_gene_etc, exon_id, exon_number, start_idx, end_idx