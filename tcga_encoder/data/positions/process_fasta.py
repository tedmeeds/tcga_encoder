

from utils.utils import *
from utils.definitions import *
import pandas as pd

version = 36

if version == 36:
  fasta_file = "data/human_genome/assembly_36/gencode.v3c.pc_transcripts.fa"
  savedir = os.path.join( os.environ["HOME"], "data/human_genome/assembly_36_fasta_process" )
  
if version == 37:
  fasta_file = "data/human_genome/assembly_37/gencode.v19.pc_transcripts.fa"
  savedir = os.path.join( os.environ["HOME"], "data/human_genome/assembly_37_fasta_process" )
  
if version == 38:
  fasta_file = "data/human_genome/assembly_38/gencode.v25.transcripts.fa"
  savedir = os.path.join( os.environ["HOME"], "data/human_genome/assembly_38_fasta_process" )
  
print "opening fasta"
f = open( os.environ["HOME"] + "/" + fasta_file, 'r')

hugo2fptr = {}

#nbr = 95310

#print "processing nbr %d"%nbr
s = f.readline()
i=0
while len(s)>0:
  #s = f.readline()
  if np.mod(i,1000)==0:
    #print "S:"
    print s
  # 
  
  if s[0] == ">":
    ss =s.split("|")
    ensembl_transcript = ss[0][1:]
    ensembl_gene = ss[1]
    hugo_gene = ss[5]
    hugo_transcript = ss[4]
    length = int(ss[6])
  else:
    continue
    
  seq = f.readline()
  assert seq[0] != ">", "something wrong"
  
  if hugo2fptr.has_key( hugo_gene ) is False:
    dirname = os.path.join( savedir, hugo_gene )
    check_and_mkdir( dirname )
      
  hugo2fptr[ hugo_gene ] = open( dirname + "/sequences.fa", "a" )
  hugo2fptr[ hugo_gene ].write( s )
  hugo2fptr[ hugo_gene ].write( seq )
  hugo2fptr[ hugo_gene ].close()
  
  s = f.readline()
  i+=1

print "closing file pointers"
for hugo_gene, fptr in hugo2fptr.iteritems():
  fptr.close()
  
print "closing fasta"
f.close()
print "done"