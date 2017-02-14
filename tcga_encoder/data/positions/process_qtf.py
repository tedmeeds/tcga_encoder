

from utils.utils import *
from utils.definitions import *
import pandas as pd

version = 38

if version == 36:
  gtf_file = "data/human_genome/assembly_36/gencode.v3c.annotation.NCBI36.gtf"
  savedir = os.path.join( os.environ["HOME"], "data/human_genome/assembly_36_gtf_process" )
  
if version == 37:
  gtf_file = "data/human_genome/assembly_37/gencode.v19.annotation.gtf"
  savedir = os.path.join( os.environ["HOME"], "data/human_genome/assembly_37_gtf_process" )
  
if version == 38:
  gtf_file = "data/human_genome/assembly_38/gencode.v25.annotation.gtf"
  savedir = os.path.join( os.environ["HOME"], "data/human_genome/assembly_38_gtf_process" )
  
#gtf_file = "data/human_genome/gencode.v19.annotation.gtf"

#savedir = os.path.join( os.environ["HOME"], "data/human_genome/gtf_process" )

print "opening fasta"
f = open( os.environ["HOME"] + "/" + gtf_file, 'r')

hugo2fptr = {}

nbr = 95310

print "processing nbr %d"%nbr
s = f.readline()
i=0
#for i in range(100):
while len(s)>0:
  #s = f.readline()
  #print s
  try:
    hugo_gene = s.split("\t")[8].split(";")[4].split(" ")[-1].lstrip('"').rstrip('"')
    #d.lstrip('"').rstrip('"')
    if hugo2fptr.has_key( hugo_gene ) is False:
      dirname = os.path.join( savedir, hugo_gene )
      check_and_mkdir( dirname )

    hugo2fptr[ hugo_gene ] = open( dirname + "/annotation.gtf", "a" )
    hugo2fptr[ hugo_gene ].write( s )
    hugo2fptr[ hugo_gene ].close()
    
    if np.mod(i,1000)==0:
      print hugo_gene
    #print hugo_gene
  except:
    print "FAILED %s"%s

    
  # 
  
#   if s[0] == ">":
#     ss =s.split("|")
#     ensembl_transcript = ss[0][1:]
#     ensembl_gene = ss[1]
#     hugo_gene = ss[5]
#     hugo_transcript = ss[4]
#     length = int(ss[6])
#   else:
#     continue
#
#   seq = f.readline()
#   assert seq[0] != ">", "something wrong"
#

#
  s = f.readline()
  i+=1
#
# print "closing file pointers"
# for hugo_gene, fptr in hugo2fptr.iteritems():
#   fptr.close()
#
# print "closing fasta"
f.close()
print "done"