from tcga_encoder.definitions.locations import *
import numpy as np
from collections import OrderedDict, Counter
import pandas as pd

class Pathways( object ):
  def __init__( self, kegg_location = "data/kegg/hsa_KEGG", hgnc_location = "data/hgnc_processed"):
    self.kegg_dir = os.path.join( HOME_DIR, kegg_location )
    self.hgnc_dir = os.path.join( HOME_DIR, hgnc_location )
    
    print "Loading Hugo"
    self.LoadGenes()
    print "Loading Kegg"
    self.LoadKegg()
    print "Joining"
    self.JoinHugoWithKegg()
    
  def LoadGenes(self):
    self.entrez = np.loadtxt( self.hgnc_dir + "/entrez.txt", dtype="str" )
    self.hugo   = np.loadtxt( self.hgnc_dir + "/hugo.txt", dtype="str" )
    
    self.entrez2hugo = OrderedDict()
    self.hugo2entrez = OrderedDict()
    
    for entrez,hugo in zip( self.entrez, self.hugo ):
      self.entrez2hugo[ entrez ] = hugo 
      self.hugo2entrez[ hugo ]   = entrez 
      
  def LoadKegg(self):
    self.sets_kgmls = pd.read_csv( self.kegg_dir + "/sets_kgmls.txt", sep="\t", index_col="ID" )
    
    pathway_lines = open( self.kegg_dir + "/gene_pathway.txt", "r" ).readlines()
    
    self.pathway2hsa = OrderedDict()
    self.hsa2pathway = OrderedDict()
    for pathway_line in pathway_lines:
      symbols = pathway_line.split("\t")
      
      hsa_gene = symbols[0].split(":")[1]
      pathways = symbols[1:]; pathways[-1] = pathways[-1].rstrip("\n")
      
      self.hsa2pathway[hsa_gene] = pathways
      
      for pathway in pathways:
        if self.pathway2hsa.has_key(pathway):
          self.pathway2hsa[pathway].append( hsa_gene )
        else:
          self.pathway2hsa[pathway] = [hsa_gene]
          
  def JoinHugoWithKegg(self):
    self.pathway2hugo = OrderedDict()
    self.hugo2pathway = OrderedDict()
    
    for entrez,hugo in zip( self.entrez, self.hugo ):
      if self.hsa2pathway.has_key( entrez ):
        self.hugo2pathway[ hugo ] = self.hsa2pathway[ entrez ]
    
        for pathway in self.hugo2pathway[ hugo ]:
          if self.pathway2hugo.has_key(pathway):
            self.pathway2hugo[pathway].append( hugo )
          else:
            self.pathway2hugo[pathway] = [hugo]
            
  def Enrichment( self, hugo_list, weights = None ):
    c = Counter()
    if weights is None:
      weights = np.ones( len(hugo_list) )
      
    for hugo,w in zip( hugo_list, weights ):
      
      if self.hugo2pathway.has_key( hugo ):
        pathways = self.hugo2pathway[ hugo ]
        path_weights = w*np.ones(len(pathways))/len(pathways)
        
        c.update( dict( zip(pathways, path_weights ) ) )
    
    most_common = pd.Series( c.values(), index = c.keys(), name="kegg")
    most_common_readable = pd.Series( c.values(), index = self.sets_kgmls.loc[c.keys()]["Symbol"], name="readable" )
    #for mc in most_common:
    #  most_common_readable.append( (self.sets_kgmls.loc[mc[0]]["Symbol"], mc[1]) )    
    return most_common.sort_values(ascending=False), most_common_readable.sort_values(ascending=False)
      
    
    

if __name__ == "__main__":
  p = Pathways()
    
      