import tensorflow as tf
import pdb
from tcga_encoder.models.layers import *
from tcga_encoder.models.regularizers import *
from tcga_encoder.algorithms import *

from tcga_encoder.models.survival import *
from tcga_encoder.models.analyses import *
from tcga_encoder.utils.helpers import *

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("talk")

import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#import itertools

def compute_mean( a, b ):
  return a/(a+b)
  
def compute_variance( a, b ):
  return (a*b)/( pow(a+b,2)*(a+b+1))
  
def compute_alpha( m, v ):
  bad = v==0
  A = -m*(v+m*m-m) / v
  A[bad] = 1.0
  
  #if bad.sum() > 0:
  #  pdb.set_trace()
  return A
  
def compute_beta( m, v ):
  bad = v==0
  B = (v+m*m-m)*(m-1.0)/v
  B[bad] = 1
  #if bad.sum() > 0:
  #  pdb.set_trace()
  return B

class TCGABatcherABC( object ):
  def __init__(self, network_name, network, data_dict, algo_dict, arch_dict, logging_dict, default_store_mode="w" ):
    self.network_name   = network_name
    self.network        = network
    self.data_dict      = data_dict
    self.algo_dict      = algo_dict
    self.arch_dict      = arch_dict
    self.logging_dict   = logging_dict
    self.var_dict       = self.arch_dict[VARIABLES]
    self.default_store_mode   = default_store_mode
    self.validation_tissues = data_dict["validation_tissues"]
    self.batcher_rates = [0.25,0.90,0.1] # A    
    self.Initialize()
    
    # these are tissues that have 0 or only tiny fully observed
    
    
  def CloseAll(self):
    self.data_store.close()
    self.survival_store.close()
    self.fill_store.close()
    self.latent_store.close()
    self.model_store.close()
    self.epoch_store.close()  
    
  def Initialize(self):
    self.beta = self.algo_dict["beta_init"]
    self.free_bits = self.algo_dict["free_bits_init"]
    self.r1 = self.algo_dict["r1"]
    self.r2 = self.algo_dict["r2"]
    self.savedir    = self.logging_dict[SAVEDIR]
    
    self.keep_rates = OrderedDict()
    if self.arch_dict.has_key( KEEP_RATES ):
      for kr in self.arch_dict[KEEP_RATES]:
        self.keep_rates[ kr[LAYER] ] = kr[KEEP_RATE]
    
    self.n_dna_channels = self.data_dict["n_dna_channels"]    
    self.dna_dim    = self.data_dict[DATASET].GetDimension(DNA)
    self.meth_dim   = self.data_dict[DATASET].GetDimension(METH)
    self.rna_dim    = self.data_dict[DATASET].GetDimension(RNA)
    self.mirna_dim    = self.data_dict[DATASET].GetDimension(miRNA)
    self.tissue_dim = self.data_dict[DATASET].GetDimension(TISSUE)
    
    self.dims_dict = {miRNA:self.mirna_dim,RNA:self.rna_dim, DNA:self.dna_dim, TISSUE:self.tissue_dim, METH:self.meth_dim, METH+"_b":self.meth_dim,miRNA+"_b":self.mirna_dim,RNA+"_b":self.rna_dim}
    print "DIMENSIONS: "
    print self.dims_dict
    self.data_store     = self.data_dict[DATASET].store
    self.batch_size     = self.algo_dict[BATCH_SIZE]
    
    np.random.seed( self.algo_dict['split_seed'] )
    
    self.batch_imputation_dict = {}
    self.batch_feed_dict       = {}
    
    # store keys
    self.OBSERVED_key = CLINICAL+"/"+OBSERVED
    self.TISSUE_key   = CLINICAL+"/"+TISSUE
    self.RNA_key      = RNA+"/"+FAIR
    self.miRNA_key      = miRNA+"/"+FAIR
    self.METH_key     = METH+"/"+FAIR
    self.DNA_keys     = [DNA+"/"+CHANNEL+"/%d"%i for i in range(self.n_dna_channels)]
    
    self.n_z            = self.var_dict[N_Z]
    self.z_columns = ["z%d"%z for z in range(self.n_z)]
    
    self.rna_genes = self.data_store[self.RNA_key].columns
    self.mirna_hsas = self.data_store[self.miRNA_key].columns
    self.dna_genes = self.data_store[self.DNA_keys[0]].columns
    self.meth_genes = self.data_store[self.METH_key].columns
    
    self.tissue_names = self.data_store[self.TISSUE_key].columns

      
    self.StoreNames()
    
    
    self.observed_order = self.data_store[self.OBSERVED_key].columns
    self.observed_source2idx = OrderedDict()
    for source, idx in zip(self.observed_order,range(len(self.observed_order)) ):
      self.observed_source2idx[source] = idx
    
    self.input_sources = self.arch_dict[INPUT_SOURCES]
    self.target_sources = self.arch_dict[TARGET_SOURCES]
    
    self.Colors()
          

        
    self.observation_source_indices = []
    for source in self.arch_dict["sources"]:
      self.observation_source_indices.append( self.observed_source2idx[source] )

    self.observation_source_indices_input = []
    for source in self.arch_dict[INPUT_SOURCES]:
      self.observation_source_indices_input.append( self.observed_source2idx[source] )

    self.observation_source_indices_target = []
    for source in self.arch_dict[TARGET_SOURCES]:
      try:
        self.observation_source_indices_target.append( self.observed_source2idx[source] )   
      except:
        print "skipping ", source             

    self.observed_batch_order = OrderedDict()
    self.observed_product_sources = []
    source_idx = 0
    for source in self.arch_dict["product_sources"]:
      self.observed_product_sources.append( self.observed_source2idx[source] )     
      self.observed_batch_order[ source ] = source_idx
      source_idx+=1
       
    print "** getting validations"
    self.at_least_one_query = self.data_store[self.OBSERVED_key].values[:,self.observed_product_sources].sum(1)>0
    
    self.validation_tissue2barcodes = OrderedDict()
    
    self.obs_store_bc_2idx = OrderedDict()
    for idx,bc in zip( range(len(self.data_store[self.OBSERVED_key].index)),self.data_store[self.OBSERVED_key].index):
      self.obs_store_bc_2idx[bc] = idx
      
    self.observed_tissue_and_bcs = self.data_store[self.OBSERVED_key].index
    self.observation_tissues = np.array([s.split("_")[0] for s in self.observed_tissue_and_bcs])
    self.validation_obs_query = np.zeros( len(self.data_store[self.OBSERVED_key]), dtype=bool)
    
    #coad_bc = "coad_tcga-t9-a92h"
    for tissue in self.validation_tissues:
      i=self.data_store["/CLINICAL/TISSUE"][tissue]==1
      self.validation_tissue2barcodes[ tissue ] = self.data_store["/CLINICAL/TISSUE"][i].index
      
      for bc in self.validation_tissue2barcodes[ tissue ]:
        found = False
        for tissue in self.validation_tissues:
          if bc.split("_")[0] == tissue:
            found = True
            break
        assert found is True, "could not find " + bc
      
      ids = self.observation_tissues==tissue
      for bc in self.validation_tissue2barcodes[ tissue ]:
        self.validation_obs_query[ self.obs_store_bc_2idx[bc] ] = True
      print tissue, self.validation_obs_query.sum()
      
    self.validation_obs_query *= self.at_least_one_query 
    self.not_validation_query = (1-self.validation_obs_query).astype(bool)
    self.validation_barcodes = self.data_store[self.OBSERVED_key].loc[self.validation_obs_query].index
    
    for bc in self.validation_barcodes:
      found = False
      for tissue in self.validation_tissues:
        if bc.split("_")[0] == tissue:
          found = True
          break
      assert found is True, "could not find " + bc
    # must have more than just tissue observed (ie one of dna, rna, meth, etc)
    
    self.usable_observed_query = self.at_least_one_query*self.not_validation_query
    self.usable_observed = self.data_store[self.OBSERVED_key][self.usable_observed_query]
    self.usable_barcodes = self.data_store[self.OBSERVED_key][self.usable_observed_query].index
    assert len(np.intersect1d( self.usable_barcodes, self.validation_barcodes)) == 0, "train and test are not mutually exclusive!!"
  
    self.n_usable = len(self.usable_barcodes)
  
    # find cases where RNA and DNA are observed (some have no METH -- COAD, READ, LAML more)
    # make fully observed having tissue, dna, rna, meth just so all the experiments get the same patients for test
    self.fully_observed_query = self.not_validation_query*self.usable_observed_query*(self.data_store[self.OBSERVED_key].values.sum(1)==len(self.observed_order))
  
    self.fully_observed = self.data_store[self.OBSERVED_key][self.fully_observed_query]
    self.fully_barcodes = self.fully_observed.index
    
    assert len(np.intersect1d( self.fully_barcodes, self.validation_barcodes)) == 0, "train and test are not mutually exclusive!!"
    
    
    self.full_observed_ids = pp.find( self.fully_observed_query )
    self.n_fully_observed = self.fully_observed_query.sum()
    self.n_full_train = self.n_fully_observed
    self.train_full_id_query = np.zeros( len(self.data_store[self.OBSERVED_key]), dtype=bool)
  
    
    self.train_full_ids = np.random.permutation( self.n_fully_observed )[:self.n_full_train]
    self.train_full_id_query[ self.full_observed_ids[self.train_full_ids] ] = 1
    
    self.test_full_id_query = (self.not_validation_query*self.fully_observed_query*(1-self.train_full_id_query).astype(np.bool) ).astype(np.bool)
    
    # all the usable, but not fully observed data
    self.non_full_observed_query = self.not_validation_query*self.at_least_one_query*(1-self.fully_observed_query).astype(np.bool)
    
    self.non_full_observed_ids = pp.find( self.non_full_observed_query )
    self.n_non_fully_observed = self.non_full_observed_query.sum()
    
    #if self.n_non_fully_observed <= self.n_non_full_train:
    #  print "==> setting test set from %d to %d"%(self.n_non_full_train,self.n_non_fully_observed)
    self.n_non_full_train = self.n_non_fully_observed
    
    self.train_non_full_id_query = np.zeros( len(self.data_store[self.OBSERVED_key]), dtype=bool)
    self.train_non_full_ids = np.random.permutation( self.n_non_fully_observed )[:self.n_non_full_train]
    self.train_non_full_id_query[ self.non_full_observed_ids[self.train_non_full_ids] ] = 1
    
    self.test_non_full_id_query = (self.not_validation_query*self.non_full_observed_query*(1-self.train_non_full_id_query).astype(np.bool) ).astype(np.bool)

    self.train_full_barcodes     = self.data_store[self.OBSERVED_key][self.train_full_id_query].index
    assert len(np.intersect1d( self.train_full_barcodes, self.validation_barcodes)) == 0, "train and test are not mutually exclusive!!"

    self.train_non_full_barcodes = self.data_store[self.OBSERVED_key][self.train_non_full_id_query].index
    assert len(np.intersect1d( self.train_non_full_barcodes, self.validation_barcodes)) == 0, "train and test are not mutually exclusive!!"

    self.test_full_barcodes      = self.data_store[self.OBSERVED_key][self.test_full_id_query].index
    assert len(np.intersect1d( self.test_full_barcodes, self.validation_barcodes)) == 0, "train and test are not mutually exclusive!!"

    self.test_non_full_barcodes  = self.data_store[self.OBSERVED_key][self.test_non_full_id_query].index
    assert len(np.intersect1d( self.test_non_full_barcodes, self.validation_barcodes)) == 0, "train and test are not mutually exclusive!!"
    
    print "** n_train_full     ", len(self.train_full_barcodes)
    print "** n_train_non_full ", len(self.train_non_full_barcodes)
    print "** n_test_full      ", len(self.test_full_barcodes)
    print "** n_test_non_full  ", len(self.test_non_full_barcodes)
    
    assert len( np.intersect1d( self.train_full_barcodes,self.train_non_full_barcodes ) ) == 0, "problem"
    assert len( np.intersect1d( self.train_full_barcodes,self.test_full_barcodes)) == 0, "problem"
    assert len( np.intersect1d( self.train_full_barcodes,self.test_non_full_barcodes)) == 0, "problem"
    assert len( np.intersect1d( self.train_non_full_barcodes,self.test_full_barcodes)) == 0, "problem"
    assert len( np.intersect1d( self.train_non_full_barcodes,self.test_non_full_barcodes)) == 0, "problem"
    assert len( np.intersect1d( self.test_full_barcodes,self.test_non_full_barcodes)) == 0, "problem"
    
    self.test_barcodes = np.union1d( self.test_full_barcodes, self.test_non_full_barcodes )
    self.train_barcodes = np.union1d( self.train_full_barcodes, self.train_non_full_barcodes )

    self.MoveValidation2Train( 0.5 )
    self.RemoveUnwantedTrain()
  
    assert len(np.intersect1d( self.test_barcodes, self.train_barcodes)) == 0, "train and test are not mutually exclusive!!"
    assert len(np.intersect1d( self.test_barcodes, self.validation_barcodes)) == 0, "test and validation are not mutually exclusive!!"
    assert len(np.intersect1d( self.train_barcodes, self.validation_barcodes)) == 0, "train and validation are not mutually exclusive!!"
    
    self.test_tissue  = self.data_store[self.TISSUE_key].loc[ self.test_barcodes ]
    self.train_tissue = self.data_store[self.TISSUE_key].loc[ self.train_barcodes ]
    self.val_tissue   = self.data_store[self.TISSUE_key].loc[ self.validation_barcodes ]
  
    self.n_train = len(self.train_barcodes)
    self.n_test  = len(self.test_barcodes)
    self.n_val  = len(self.validation_barcodes)
  
    self.data_dict[N_TRAIN] = self.n_train
    self.data_dict[N_TEST]  = self.n_test
    

    self.test_tissue  = self.data_store[self.TISSUE_key].loc[ self.test_barcodes ]
    self.train_tissue = self.data_store[self.TISSUE_key].loc[ self.train_barcodes ]
    self.val_tissue   = self.data_store[self.TISSUE_key].loc[ self.validation_barcodes ]

  
    self.data_dict[N_TRAIN] = self.n_train
    self.data_dict[N_TEST]  = self.n_test
    
    print "** n_train = ", self.n_train
    print "** n_test  = ", self.n_test
    print "** n_val  = ", self.n_val
    
    print "TEST: " 
    print self.test_tissue.sum()
    print "TRAIN: " 
    print self.train_tissue.sum()
    print "VAL: " 
    print self.val_tissue.sum()
    #pdb.set_trace()
    
    self.fill_store.open()
    z_columns = ["z%d"%zidx for zidx in range(self.n_z)]
    self.fill_store["Z/TRAIN/Z/mu"]  = pd.DataFrame( np.zeros( (len(self.train_barcodes),self.n_z) ), index = self.train_barcodes, columns = z_columns )
    self.fill_store["Z/TRAIN/Z/var"] = pd.DataFrame( np.zeros( (len(self.train_barcodes),self.n_z) ), index = self.train_barcodes, columns = z_columns )
    self.fill_store.close()
    
    self.n_train = len(self.train_barcodes)
    self.n_test  = len(self.test_barcodes)
    self.n_val   = len(self.validation_barcodes)
    
    self.MakeVizFilenames()
    self.batch_size     = min( self.batch_size, self.n_train )
    
    self.SummarizeData()
    #assert False, "todo"
    # make classfier to tissue
    # used filled tissue for validation without tissue in train
    # check weights from tissue to hidden make sure there arent too many (figures)
    # compare training with just one tissue
    

  def InitializeAnythingYouWant( self, sess, network ):
    pass
    
  def SummarizeData(self):
    print "Running : SummarizeData(self)"
    #pass
    # self.OBSERVED_key = CLINICAL+"/"+OBSERVED
    # self.TISSUE_key   = CLINICAL+"/"+TISSUE
    # self.RNA_key      = RNA+"/"+FAIR
    # self.miRNA_key      = miRNA+"/"+FAIR
    # self.METH_key     = METH+"/"+FAIR
    # self.DNA_keys     = [DNA+"/"+CHANNEL+"/%d"%i for i in range(self.n_dna_channels)]
    self.rna_mean = self.data_store[self.RNA_key].mean(0)
    self.rna_std = self.data_store[self.RNA_key].std(0)
    self.mirna_mean = self.data_store[self.miRNA_key].mean(0)
    self.mirna_std = self.data_store[self.miRNA_key].std(0)
    self.meth_mean = self.data_store[self.METH_key].mean(0)
    self.meth_std = self.data_store[self.METH_key].std(0)
    
    self.rna_order = np.argsort( self.rna_mean.values )
    self.mirna_order = np.argsort( self.mirna_mean.values )
    self.meth_order = np.argsort( self.meth_mean.values )
    
    self.tissue_statistics = {}
    
    tissue_names = self.train_tissue.columns
    stats = np.zeros( (5,len(tissue_names)))
    for t_idx, tissue in zip( range(len(tissue_names)),tissue_names ):
      bcs = self.train_tissue.loc[self.train_tissue[tissue]==1].index.values
      
      rna=self.data_store[self.RNA_key].loc[ bcs ]
      mirna=self.data_store[self.miRNA_key].loc[ bcs ]
      meth=self.data_store[self.METH_key].loc[ bcs ]
      
      self.tissue_statistics[ tissue ] = {}
      self.tissue_statistics[ tissue ][ RNA ] = {}
      self.tissue_statistics[ tissue ][ miRNA ] = {}
      self.tissue_statistics[ tissue ][ METH ] = {}
      self.tissue_statistics[ tissue ][ RNA ][ "mean"]   = rna.mean(0).fillna(0)
      self.tissue_statistics[ tissue ][ miRNA ][ "mean"] = mirna.mean(0).fillna(0)
      self.tissue_statistics[ tissue ][ METH ][ "mean"]  = meth.mean(0).fillna(0)
      self.tissue_statistics[ tissue ][ RNA ][ "var"]   = rna.var(0).fillna(0)
      self.tissue_statistics[ tissue ][ miRNA ][ "var"] = mirna.var(0).fillna(0)
      self.tissue_statistics[ tissue ][ METH ][ "var"]  = meth.var(0).fillna(0)
      

      self.tissue_statistics[ tissue ][ RNA ][ "alpha"]   = compute_alpha( self.tissue_statistics[ tissue ][ RNA ][ "mean"], self.tissue_statistics[ tissue ][ RNA ][ "var"])
      self.tissue_statistics[ tissue ][ miRNA ][ "alpha"] = compute_alpha( self.tissue_statistics[ tissue ][ miRNA ][ "mean"], self.tissue_statistics[ tissue ][ miRNA ][ "var"])
      self.tissue_statistics[ tissue ][ METH ][ "alpha"]  = compute_alpha( self.tissue_statistics[ tissue ][ METH ][ "mean"], self.tissue_statistics[ tissue ][ METH ][ "var"])

      self.tissue_statistics[ tissue ][ RNA ][ "beta"]   = compute_beta( self.tissue_statistics[ tissue ][ RNA ][ "mean"], self.tissue_statistics[ tissue ][ RNA ][ "var"])
      self.tissue_statistics[ tissue ][ miRNA ][ "beta"] = compute_beta( self.tissue_statistics[ tissue ][ miRNA ][ "mean"], self.tissue_statistics[ tissue ][ miRNA ][ "var"])
      self.tissue_statistics[ tissue ][ METH ][ "beta"]  = compute_beta( self.tissue_statistics[ tissue ][ METH ][ "mean"], self.tissue_statistics[ tissue ][ METH ][ "var"])
      
      #if tissue == "laml":
      #  pdb.set_trace()
      
      #pdb.set_trace()
    
    #self.tissue_statistics = pd.DataFrame()
    
    
    
  def MoveValidation2Train( self, percent2move = 0.5  ):
    I = np.random.permutation( len( self.validation_barcodes ) )
    n = int(percent2move*len(I))
    
    self.train_barcodes = np.hstack((self.train_barcodes,self.validation_barcodes[I[:n]]))
    self.validation_barcodes = self.validation_barcodes[I[n:]]
    
    print self.train_barcodes
    print self.validation_barcodes
  
  def RemoveUnwantedTrain(self):
    if self.data_dict.has_key("train_tissues") is False:
      return
    
    train_tissues = np.array( [bc.split("_")[0] for bc in self.train_barcodes], dtype=str )
    train_bcs = []
    for tissue in self.data_dict["train_tissues"]:
      I = pp.find( train_tissues==tissue )
      train_bcs.extend(self.train_barcodes[I])
    
    self.train_barcodes = np.array( train_bcs, dtype=str)
    assert len(self.train_barcodes)>0, "NO TRAINING DATA"
      
  def MakeVizFilenames(self):
    self.viz_filename_survival      =  os.path.join( self.savedir, "survival" )
    self.viz_filename_survival_lda  =  os.path.join( self.savedir, "survival__lda" )
    self.viz_filename_z_to_dna      =  os.path.join( self.savedir, "lda_dna" )
    self.viz_filename_z_rec_scatter          =  os.path.join( self.savedir, "z_rec_scatter.png" )
    self.viz_filename_z_rec_on_z_gen         =  os.path.join( self.savedir, "z_rec_on_z_gen.png" )
    self.viz_filename_rna_prediction_scatter =  os.path.join( self.savedir, "rna_prediction_scatter.png" )
    self.viz_filename_dna_batch_target       =  os.path.join( self.savedir, "dna_batch_target" )
    self.viz_filename_dna_batch_predict      =  os.path.join( self.savedir, "dna_batch_predict" )
    self.viz_filename_dna_aucs               =  os.path.join( self.savedir, "dna_aucs.png" )
    self.viz_filename_weights        =  os.path.join( self.savedir, "weights_" )
    self.viz_filename_lower_bound            =  os.path.join( self.savedir, "lower_bound.png" )
    self.viz_filename_log_pdf_sources        = os.path.join( self.savedir, "log_pdf_sources_z.png" )
    self.viz_filename_log_pdf_sources_per_gene = os.path.join( self.savedir, "log_pdf_batch.png" )
    self.viz_filename_log_pdf_sources_per_gene_fill = os.path.join( self.savedir, "log_pdf_fill.png" )
    self.viz_filename_error_sources_per_gene_fill = os.path.join( self.savedir, "errors_fill.png" )
    self.viz_filename_log_pdf_sources_per_gene_fill_all = os.path.join( self.savedir, "log_pdf_sources_z_per_gene_fill_all.png" )
    self.viz_filename_error_sources_per_gene_fill_all = os.path.join( self.savedir, "errors_sources_z_per_gene_fill_all.png" )
  
  def Colors(self):
    self.tissue2color = OrderedDict()
    self.tissue2shape = OrderedDict()
    cmap = pp.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(self.tissue_names)))
    for idx,t in zip( range(len(colors)),self.tissue_names):
      self.tissue2color[t] = colors[idx]
      self.tissue2shape[t] = 'o'
          
    self.source2darkcolor = {RNA:"darkblue", DNA:"darkgreen", METH:"darkred", miRNA:"darkorange", RNA+"_b":"lightblue", DNA+"_b":"palegreen", METH+"_b":"lightsalmon", miRNA+"_b":"moccasin"}
    self.source2lightcolor = {RNA:"lightblue", DNA:"palegreen", METH:"lightsalmon", miRNA:"moccasin",RNA+"_b":"lightblue", DNA+"_b":"palegreen", METH+"_b":"lightsalmon", miRNA+"_b":"moccasin"}
    self.source2mediumcolor = {RNA:"dodgerblue", DNA:"darksage", METH:"red", miRNA:"orange", RNA+"_b":"darkblue", DNA+"_b":"darkgreen", METH+"_b":"darkred", miRNA+"_b":"darkorange"}
    
    self.source2mediumcolor[RNA+"+"+DNA]="turquoise"
    self.source2mediumcolor[RNA+"+"+METH] = "fuchsia"
    self.source2mediumcolor[DNA+"+"+METH] = "sandybrown"
    self.source2mediumcolor[RNA+"+"+DNA+"+"+METH]="grey"
    
    self.input_combo2fillcolor  = {}
    self.input_combo2shape      = {}
    self.input_combo2markersize = {}
    
    for nbr in np.arange( 1, len(self.input_sources)+1 ):
      for combo in itertools.combinations( self.input_sources, nbr ):
        inputs2use = np.array(combo)
        inputs = inputs2use[0]
        for ss in inputs2use[1:]:
          inputs += "+%s"%(ss)
          
        if nbr == 1:
          self.input_combo2fillcolor[inputs] = self.source2lightcolor[inputs]
          self.input_combo2shape[inputs] = 'D'
          self.input_combo2markersize[inputs] = 5
     
  def StoreNames(self):
    self.latent_store_name = self.network_name + "_" + LATENT
    self.latent_store = OpenHdfStore(self.savedir, self.latent_store_name, mode=self.default_store_mode )
    self.model_store_name = self.network_name + "_" + MODEL
    self.survival_store_name = self.network_name + "_" + SURVIVAL
    
    # open in "a" mode
    self.model_store = OpenHdfStore(self.savedir, self.model_store_name, mode="a" )
    
    self.epoch_store_name = self.network_name + "_" + EPOCH
    self.epoch_store = OpenHdfStore(self.savedir, self.epoch_store_name, mode=self.default_store_mode )
    
    self.fill_store_name = self.network_name + "_" + FILL
    self.fill_store = OpenHdfStore(self.savedir, self.fill_store_name, mode="a")
    
    self.survival_store = OpenHdfStore(self.savedir, self.survival_store_name, mode=self.default_store_mode )
    
    self.survival_store.close()
    self.fill_store.close()
    self.latent_store.close()
    self.model_store.close()
    self.epoch_store.close()
     
  def CallBack( self, function_name, sess, cb_info ):
    if function_name == BATCH_EPOCH:
      self.BatchEpoch( sess, cb_info )
      #self.BatchFillZ( sess, cb_info )
  
    elif function_name == TEST_EPOCH:
      self.TestEpoch( sess, cb_info )
      self.ValEpoch( sess, cb_info )
      
    elif function_name == EPOCH_VIZ:
      self.VizEpochs( sess, cb_info )
  
    elif function_name == SAVE_MODEL:
      self.SaveModel( sess, cb_info )
 
    elif function_name == MODEL_VIZ:
      self.VizModel( sess, cb_info )
   
    elif function_name == SAVE_LATENT:
      self.SaveTestLatent( sess, cb_info )
  
    elif function_name == LATENT_VIZ:
      self.VizLatent( sess, cb_info )
      
    elif function_name == "survival":
      self.RunSurvival( sess, cb_info )
  
    elif function_name == TEST_FILL:
      self.TestFill2( sess, cb_info )
      self.TestFillZ( sess, cb_info )
      self.TrainFillZ( sess, cb_info )
      
    elif function_name == "beta":
      if self.algo_dict["beta_growth"] < 0:
        self.beta = max( self.algo_dict["beta_min"], -self.beta*self.algo_dict["beta_growth"] )
      else:
        self.beta = min( self.algo_dict["beta_max"], self.beta*self.algo_dict["beta_growth"] )
      print "BETA ", self.beta
      
    elif function_name == "free_bits":
      if self.algo_dict["free_bits_growth"] < 0:
        self.free_bits = max( self.algo_dict["free_bits_min"], -self.free_bits*self.algo_dict["free_bits_growth"] )
      else:
        self.free_bits = min( self.algo_dict["free_bits_max"], self.free_bits*self.algo_dict["free_bits_growth"] )
      print "FREE_BITS ", self.free_bits

  def SaveSurvival( self, disease_list, predict_survival_train, g1, g2 ):
    if disease_list.__class__ == list:
      disease = disease_list[0]
      disease_query_train    = predict_survival_train["disease"].values == disease_list[0]
      #disease_query_test    = predict_survival_test["disease"].values == disease_list[0]
      
      for disease in disease_list[1:]:
        disease += "_%s"%(disease)
        disease_query_train    += predict_survival_train["disease"].values == disease
        #disease_query_test     += predict_survival_test["disease"].values == disease
    else:
      disease = disease_list
      disease_query_train    = predict_survival_train["disease"].values == disease_list
      #disease_query_test    = predict_survival_test["disease"].values == disease_list

    #disease_query_train    = predict_survival_train["disease"].values == disease
    #disease_survival_train = predict_survival_train[ disease_query_train ]
    
    
    #disease_query_train    = predict_survival_train["disease"].values == disease
    disease_survival_train = predict_survival_train[ disease_query_train ]
    #T_train = disease_survival_train["T"].values
    #E_train = disease_survival_train["E"].values
    #Z_train = disease_survival_train[z_columns].values
    barcodes = disease_survival_train.index
    diseases = disease_survival_train["disease"].values
    disease_barcodes = [ "%s_%s"%(dis,barcode) for dis,barcode in zip(diseases,barcodes)]
    
    try:
      dna = self.data_store[self.DNA_keys[0]].loc[ disease_barcodes ]
    except:
      print "No DNA for %s"%disease
      print "Not Saving!"
      return 
    
    index1 = pd.MultiIndex.from_tuples( zip( barcodes, g1 ), names=['barcode', 'group'])
    index2 = pd.MultiIndex.from_tuples( zip( barcodes, g2 ), names=['barcode', 'group'])
    self.survival_store.open()
    self.survival_store["/%s/split1"%(disease)] = pd.DataFrame( dna.fillna(0).values.astype(int), columns = dna.columns, index = index1 )
    self.survival_store["/%s/split2"%(disease)] = pd.DataFrame( dna.fillna(0).values.astype(int), columns = dna.columns, index = index2 )
    
    #pdb.set_trace()
    self.survival_store.close()
    
    
    
  def RunSurvival( self, sess, cb_info ):
     #kmeans_then_survival( self, sess, cb_info )
     #lda_then_survival( self, sess, cb_info )
     #lda_on_mutations( self, sess, cb_info )
     
     #pdb.set_trace()
     #for disease in self.validation_tissues:
     lda_then_survival_on_disease( self, sess, cb_info, self.validation_tissues )
  
  def TestFill2( self, sess, info_dict ):
    epoch       = info_dict[EPOCH]
    # feed_dict   = info_dict[TEST_FEED_DICT]
    # impute_dict = info_dict[TEST_FEED_IMPUTATION]
    #
    # self.RunFill2( epoch, sess, feed_dict, impute_dict, mode="TEST" )
    
    feed_dict   = info_dict[VAL_FEED_DICT]
    impute_dict = info_dict[VAL_FEED_IMPUTATION]
    
    self.RunFill2( epoch, sess, feed_dict, impute_dict, mode="VAL" )

  def TestFillZ( self, sess, info_dict ):
    epoch       = info_dict[EPOCH]
    # feed_dict   = info_dict[TEST_FEED_DICT]
    # impute_dict = info_dict[TEST_FEED_IMPUTATION]
    #
    # self.RunFillZ( epoch, sess, feed_dict, impute_dict, mode="TEST" )
    
    feed_dict   = info_dict[VAL_FEED_DICT]
    impute_dict = info_dict[VAL_FEED_IMPUTATION]
    
    self.RunFillZ( epoch, sess, feed_dict, impute_dict, mode="VAL" )
    
    feed_dict   = info_dict[BATCH_FEED_DICT]
    impute_dict = info_dict[BATCH_FEED_IMPUTATION]
    self.batch_ids = info_dict["batch_ids"]
    self.RunFillZ( epoch, sess, feed_dict, impute_dict, mode="BATCH" )

  def TrainFillZ( self, sess, info_dict ):
    epoch       = info_dict[EPOCH]
    
    for batch_ids in chunks( np.arange(len(self.train_barcodes)), 500 ):
      barcodes = self.train_barcodes[batch_ids]
      impute_dict = self.NextBatch(batch_ids)
      impute_dict[BARCODES] = barcodes
      self.batch_ids = batch_ids
      #pdb.set_trace()
      #batch = self.FillBatch( impute_dict[BARCODES], mode )
      self.RunFillZ( epoch, sess, {}, impute_dict, mode="TRAIN" )
  
    
  def BatchFillZ( self, sess, info_dict ):
    epoch       = info_dict[EPOCH]
    feed_dict   = info_dict[BATCH_FEED_DICT]
    impute_dict = info_dict[BATCH_FEED_IMPUTATION]
    self.batch_ids = info_dict["batch_ids"]
    self.RunFillZ( epoch, sess, feed_dict, impute_dict, mode="BATCH" )


  def RunFillZ( self, epoch, sess, feed_dict, impute_dict, mode ):
    #print "FILL Z"
          
          
    barcodes = impute_dict[BARCODES]
    #batch = self.FillBatch( impute_dict[BARCODES], mode )
        
    rec_z_space_tensors       = self.network.GetTensor( "rec_z_space" )
    rna_rec_z_space_tensors   = self.network.GetTensor( "rec_z_space_rna" )
    mirna_rec_z_space_tensors   = self.network.GetTensor( "rec_z_space_mirna" )
    #dna_rec_z_space_tensors   = self.network.GetTensor( "rec_z_space_dna" )
    meth_rec_z_space_tensors  = self.network.GetTensor( "rec_z_space_meth" )
  
    tensors = []
    tensors.extend(rec_z_space_tensors)
    tensors.extend(rna_rec_z_space_tensors)
    #tensors.extend(dna_rec_z_space_tensors)
    tensors.extend(meth_rec_z_space_tensors)
    tensors.extend(mirna_rec_z_space_tensors)

    self.network.FillFeedDict( feed_dict, impute_dict )
    
    z_eval = sess.run( tensors, feed_dict = feed_dict )

          
    self.WriteRunFillZ( epoch, "Z", barcodes, self.z_columns, z_eval[0],z_eval[1], mode )      
    self.WriteRunFillZ( epoch, RNA, barcodes, self.z_columns, z_eval[2],z_eval[3], mode )
    #self.WriteRunFillZ( epoch, DNA, barcodes, self.z_columns, z_eval[4],z_eval[5], mode )
    self.WriteRunFillZ( epoch, METH, barcodes, self.z_columns, z_eval[4],z_eval[5], mode )
    self.WriteRunFillZ( epoch, miRNA, barcodes, self.z_columns, z_eval[6],z_eval[7], mode )

  def WriteRunFillZ( self, epoch, target, barcodes, columns, z_mu, z_var, mode ):
    #inputs = inputs2use[0]
    #for s in inputs2use[1:]:
    #  inputs += "+%s"%(s)

    self.fill_store.open()
    
    if mode == "TRAIN" and target == "Z":
      #pdb.set_trace()
      X_mu = self.fill_store["/Z/TRAIN/Z/mu"].values
      X_mu[self.batch_ids,:] = z_mu
      X_var = self.fill_store["/Z/TRAIN/Z/var"].values
      X_var[self.batch_ids,:] = z_var
      self.fill_store["Z/TRAIN/Z/mu"]  = pd.DataFrame( X_mu, index = self.train_barcodes, columns = columns )
      self.fill_store["Z/TRAIN/Z/var"] = pd.DataFrame( X_var, index = self.train_barcodes, columns = columns )
    else:
      self.fill_store["Z/%s/%s/mu"%(mode,target)]  = pd.DataFrame( z_mu, index = barcodes, columns = columns )
      self.fill_store["Z/%s/%s/var"%(mode,target)] = pd.DataFrame( z_var, index = barcodes, columns = columns )
      
      #
      #
      # for bc,z_mu_val in zip( barcodes, z_mu ):
      #   for z_mu_val_i, z_column in zip( z_mu_val, columns ):
      #     self.fill_store["/Z/TRAIN/Z/mu"].loc[bc,z_column] = z_mu_val_i
      #pdb.set_trace()

    self.fill_store.close()
        
  def RunFill2( self, epoch, sess, feed_dict, impute_dict, mode ):
    print "COMPUTE Z-SPACE"
    use_dna = False
    use_rna = True
    use_meth = True
    use_mirna = True
          
    barcodes = impute_dict[BARCODES]
    batch = self.FillBatch( impute_dict[BARCODES], mode )
    #not_observed = np.setdiff1d( self.input_sources, inputs2use )
        
    rec_z_space_tensors       = self.network.GetTensor( "rec_z_space" )
    rna_rec_z_space_tensors   = self.network.GetTensor( "rec_z_space_rna" )
    mirna_rec_z_space_tensors   = self.network.GetTensor( "rec_z_space_mirna" )
    #dna_rec_z_space_tensors   = self.network.GetTensor( "rec_z_space_dna" )
    meth_rec_z_space_tensors  = self.network.GetTensor( "rec_z_space_meth" )
  
    rna_expectation_tensor = self.network.GetLayer( "gen_rna_space" ).expectation
    mirna_expectation_tensor = self.network.GetLayer( "gen_mirna_space" ).expectation
    meth_expectation_tensor = self.network.GetLayer( "gen_meth_space" ).expectation
    
    if use_dna:
      dna_expectation_tensor = self.network.GetLayer( "gen_dna_space" ).expectation
      dna_data = np.zeros( (len(barcodes),self.dna_dim) )
      for idx,DNA_key in zip(range(len(self.DNA_keys)),self.DNA_keys):
        batch_data = self.data_store[DNA_key].loc[ barcodes ].fillna( 0 ).values
        dna_data += batch_data
      
      dna_data = np.minimum(1.0,dna_data)
      
    loglikes_data_as_matrix = self.network.loglikes_data_as_matrix
  
    tensors = []
    tensors.extend(rec_z_space_tensors)
    tensors.extend(rna_rec_z_space_tensors)
    tensors.extend(mirna_rec_z_space_tensors)
    #tensors.extend(dna_rec_z_space_tensors)
    tensors.extend(meth_rec_z_space_tensors)
    tensors.extend([rna_expectation_tensor,mirna_expectation_tensor,meth_expectation_tensor])
  
    tensor_names = ["z_mu","z_var",\
                    "z_mu_rna","z_var_rna",\
                    "z_mu_mirna","z_var_mirna",\
                    "z_mu_meth","z_var_meth",\
                    "rna_expecation","mirna_expectation","meth_expectation"]
  
    assert len(tensor_names)==len(tensors), "should be same number"
    self.network.FillFeedDict( feed_dict, impute_dict )

    #pdb.set_trace()
    rna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[RNA]] == 1
    #dna_observed_query = batch[ DNA_OBSERVATIONS ] == 1
    meth_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[METH]] == 1
    mirna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[miRNA]] == 1
        
    rna_expectation = np.zeros( (len(barcodes), self.dims_dict[RNA] ), dtype=float )
    rna_loglikelihood  = np.zeros( (np.sum(rna_observed_query), self.dims_dict[RNA] ), dtype=float )
    meth_expectation = np.zeros( (len(barcodes), self.dims_dict[METH] ), dtype=float )
    meth_loglikelihood  = np.zeros( (np.sum(meth_observed_query), self.dims_dict[METH] ), dtype=float )
    mirna_expectation = np.zeros( (len(barcodes), self.dims_dict[miRNA] ), dtype=float )
    mirna_loglikelihood  = np.zeros( (np.sum(mirna_observed_query), self.dims_dict[miRNA] ), dtype=float )
        
      #drop_likelihoods = np.zeros( rna_dim )
    # dna_dim = self.dims_dict[DNA] #/self.n_dna_channels
    # dna_expectation = np.zeros( (len(barcodes),dna_dim), dtype=float )
    # dna_loglikelihood = np.zeros( (np.sum(dna_observed_query),dna_dim), dtype=float )
    
    nbr_splits = 50
    tensor2fill = []
    drop_factor = float(nbr_splits)/float(nbr_splits-1)
    for drop_idx in range(nbr_splits):
      
      
      
      
      
      id_start = 0
      # ------
      # RNA
      # -----
      if use_rna:
        drop_rna_ids = np.arange(drop_idx,self.dims_dict[RNA],nbr_splits, dtype=int)
        batch_data = self.data_store[self.RNA_key].loc[ barcodes ]
        nans = np.isnan( batch_data.values )
        batch[ RNA_INPUT ] = drop_factor*self.NormalizeRnaInput( batch_data.fillna( 0 ).values )
        batch[ RNA_INPUT ][nans] = 0
        batch[ RNA_INPUT][:,drop_rna_ids] = 0
        tensor2fill.extend( [rna_expectation_tensor, loglikes_data_as_matrix["gen_rna_space"] ] )
        rna_ids = [id_start,id_start+1]
        id_start+=2

      # ------
      # miRNA
      # -----
      if use_mirna:
        drop_mirna_ids = np.arange(drop_idx,self.dims_dict[miRNA],nbr_splits, dtype=int)
        batch_data = self.data_store[self.miRNA_key].loc[ barcodes ]
        nans = np.isnan( batch_data.values )
        batch[ miRNA_INPUT ] = drop_factor*self.NormalizemiRnaInput( batch_data.fillna( 0 ).values )
        batch[ miRNA_INPUT ][nans] = 0
        batch[ miRNA_INPUT][:,drop_mirna_ids] = 0
        tensor2fill.extend( [mirna_expectation_tensor, loglikes_data_as_matrix["gen_mirna_space"] ] )
        mirna_ids = [id_start,id_start+1]
        id_start+=2
             
      
      # ------
      # DNA
      # -----
      if use_dna:
        drop_dna_ids = np.arange(drop_idx,dna_dim,nbr_splits, dtype=int)
        dna_data_inputs = np.minimum(1.0,dna_data)
        dna_data_inputs[:,drop_dna_ids] = 0
        batch[ DNA_INPUT ] = drop_factor*dna_data_inputs
        tensor2fill.extend( [dna_expectation_tensor, loglikes_data_as_matrix["gen_dna_space"] ] )
        dna_ids = [id_start,id_start+1]
        id_start+=2
        
      
      # ------
      # METH
      # -----
      if use_meth:
        drop_meth_ids = np.arange(drop_idx,self.dims_dict[METH],nbr_splits, dtype=int)
        batch_data = self.data_store[self.METH_key].loc[ barcodes ]
        batch[ METH_INPUT ] = drop_factor*batch_data.fillna( 0 ).values
        batch[ METH_INPUT][:,drop_meth_ids] = 0
        tensor2fill.extend( [meth_expectation_tensor, loglikes_data_as_matrix["gen_meth_space"] ] )
        meth_ids = [id_start,id_start+1]
        id_start+=2
      # columns = self.meth_genes
      # observations = self.data_store[self.METH_key].loc[ barcodes ].values
        
      
      # ---------
      # RUN SESS
      # ---------
      self.network.FillFeedDict( feed_dict, batch )
      tensor2fill_eval = sess.run( tensor2fill, feed_dict = feed_dict )

      # ------
      # FILL EVALUATION 
      # -----      
      if use_rna:
        rna_expectation[:,drop_rna_ids]   = tensor2fill_eval[rna_ids[0]][:,drop_rna_ids]
        rna_loglikelihood[:,drop_rna_ids] = tensor2fill_eval[rna_ids[1]][:,drop_rna_ids]

      if use_mirna:
        mirna_expectation[:,drop_mirna_ids]   = tensor2fill_eval[mirna_ids[0]][:,drop_mirna_ids]
        mirna_loglikelihood[:,drop_mirna_ids] = tensor2fill_eval[mirna_ids[1]][:,drop_mirna_ids]
      
      if use_dna:
        for idx,DNA_key in zip(range(len(self.DNA_keys)-1),self.DNA_keys[:-1]):
          dna_expectation[:,drop_dna_ids] = tensor2fill_eval[dna_ids[0]][:,drop_dna_ids]
          dna_loglikelihood[:,drop_dna_ids] = tensor2fill_eval[dna_ids[1]][:,drop_dna_ids]
        
      if use_meth:
        meth_expectation[:,drop_meth_ids]   = tensor2fill_eval[meth_ids[0]][:,drop_meth_ids]
        meth_loglikelihood[:,drop_meth_ids] = tensor2fill_eval[meth_ids[1]][:,drop_meth_ids]
    
    #pdb.set_trace()   
    if use_rna:   
      self.WriteRunFillExpectation( epoch, RNA, barcodes, self.rna_genes, rna_observed_query, rna_expectation, self.data_store[self.RNA_key].loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, RNA, barcodes[rna_observed_query], self.rna_genes, rna_loglikelihood, mode )

    if use_meth:
      self.WriteRunFillExpectation( epoch, METH, barcodes, self.meth_genes, meth_observed_query, meth_expectation, self.data_store[self.METH_key].loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, METH, barcodes[meth_observed_query], self.meth_genes, meth_loglikelihood, mode )

    if use_mirna:
      self.WriteRunFillLoglikelihood( epoch, miRNA, barcodes[mirna_observed_query], self.mirna_hsas, mirna_loglikelihood, mode )
      self.WriteRunFillExpectation( epoch, miRNA, barcodes, self.mirna_hsas, mirna_observed_query, mirna_expectation, self.data_store[self.miRNA_key].loc[ barcodes ].values, mode )
    
    if use_dna:
      self.WriteRunFillExpectation( epoch, DNA, barcodes, self.dna_genes, dna_observed_query, dna_expectation, dna_data, mode )
      self.WriteRunFillLoglikelihood( epoch, DNA, barcodes[dna_observed_query], self.dna_genes, dna_loglikelihood, mode )

    #self.WriteRunFillLoglikelihood( epoch, target, inputs2use, barcodes, columns, target_loglikelihood, is_test )
    #pdb.set_trace()

  def WriteRunFillExpectation( self, epoch, target, barcodes, columns, obs_query, X, Y, mode ):
    #inputs = inputs2use[0]
    #for s in inputs2use[1:]:
    #  inputs += "+%s"%(s)

    self.fill_store.open()
    if target == DNA:
      #for channel in range(self.n_dna_channels):
      s = "/Fill/%s/%s/"%(mode,target )
      self.fill_store[ s ] = pd.DataFrame( X, index = barcodes, columns = columns )
      x = X[obs_query,:].flatten()
      y = Y[obs_query,:].flatten()
      if y.sum()>0:
        auc = roc_auc_score(y,x)
      else:
        auc = 1.0
      errors = 1.0-auc
      #aa,cc,dd = roc_curve(flattened_dna_target,flattened_dna_predict)

    else:
      s = "/Fill/%s/%s/"%(mode,target )
      self.fill_store[ s ] = pd.DataFrame( X, index = barcodes, columns = columns )
      errors = np.mean( np.square( X[obs_query,:]-Y[obs_query,:] ) )
    
    self.fill_store.close()
    
    values = [epoch, target, errors]
    columns = ["Epoch","Target","Error"]
    #pdb.set_trace()
    if mode == "TEST":
      self.AddSeries(  self.epoch_store, TEST_FILL_ERROR, values = values, columns = columns )  
    elif mode == "VAL":
      self.AddSeries(  self.epoch_store, VAL_FILL_ERROR, values = values, columns = columns )  
    

  def WriteRunFillLoglikelihood( self, epoch, target, barcodes, columns, X, mode ):
    
    self.fill_store.open()
    if target == DNA:
      
      #for channel in range(self.n_dna_channels):
      s = "/Loglik/%s/%s/"%(mode,target )
      self.fill_store[ s] = pd.DataFrame( X, index = barcodes, columns = columns )
        
    else:
      s = "/Loglik/%s/%s/"%(mode,target )
      self.fill_store[ s ] = pd.DataFrame( X, index = barcodes, columns = columns )
    
    loglik = X.flatten().mean()
    
    self.fill_store.close()
    
    values = [epoch, target, loglik]
    columns = ["Epoch","Target","LogLik"]
    
    if mode == "TEST":
      self.AddSeries(  self.epoch_store, TEST_FILL_LOGLIK, values = values, columns = columns )  
    elif mode == "VAL":
      #pdb.set_trace()
      self.AddSeries(  self.epoch_store, VAL_FILL_LOGLIK, values = values, columns = columns )  
    
    
        
  # ---------------------------------- #
  # ------- EPOCH CALLBACKS
  # ---------------------------------- #
    
  def BatchEpoch( self, sess, info_dict ):
    epoch       = info_dict[EPOCH]
    feed_dict   = info_dict[BATCH_FEED_DICT]
    impute_dict = info_dict[BATCH_FEED_IMPUTATION]
    
    self.Epoch( "Batch", sess, info_dict, epoch, feed_dict, impute_dict, mode = "BATCH" )

  def TestEpoch( self, sess, info_dict ):
    epoch       = info_dict[EPOCH]
    feed_dict   = info_dict[TEST_FEED_DICT]
    impute_dict = info_dict[TEST_FEED_IMPUTATION]
    
    self.Epoch( "Test", sess, info_dict, epoch, feed_dict, impute_dict, mode = "TEST" )

  def ValEpoch( self, sess, info_dict ):    
    epoch       = info_dict[EPOCH]
    feed_dict   = info_dict[VAL_FEED_DICT]
    impute_dict = info_dict[VAL_FEED_IMPUTATION]
    
    self.Epoch( "Val", sess, info_dict, epoch, feed_dict, impute_dict, mode = "VAL" )
            
  def VizEpochs(self, sess, info_dict ):
    print "** VIZ Epochs"
    self.epoch_store.open()
  
    f = pp.figure()
  
    pp.plot( self.epoch_store["Batch"]["Epoch"].values, self.epoch_store["Batch"]["Lower Bound"], 'bo-', lw=2 , label="Batch")
    if self.n_test > 0:
      pp.plot( self.epoch_store["Test"]["Epoch"].values, self.epoch_store["Test"]["Lower Bound"], 'ro-', lw=2, label="Test" )
    if self.n_val > 0:
      pp.plot( self.epoch_store["Val"]["Epoch"].values, self.epoch_store["Val"]["Lower Bound"], 'ro-', lw=2, label="Val" )
    pp.legend( loc="lower right")
    pp.xlabel("Epoch")
    pp.ylabel("Lower Bound")
    pp.grid('on')
  
    pp.savefig( self.viz_filename_lower_bound, dpi = 300, fmt="png", bbox_inches = "tight")

  
    pp.close(f)
    f = pp.figure()
    legends  = []
    colours = "bgr"
    fill_colours = ["lightblue","lightgreen","lightred"]
    n_sources = len(self.arch_dict[TARGET_SOURCES])
    for idx,target_source in zip( range(n_sources),self.arch_dict[TARGET_SOURCES]):
      s = f.add_subplot(1,n_sources,idx+1)
      pp.plot( self.epoch_store[BATCH_SOURCE_LOGPDF]["Epoch"].values, 
               self.epoch_store[BATCH_SOURCE_LOGPDF][target_source]/self.dims_dict[target_source], 's--', \
               color=self.source2mediumcolor[target_source], \
               mec=self.source2darkcolor[target_source], mew=1, \
               mfc=self.source2lightcolor[target_source], lw=1, \
               ms = 5, \
               alpha=0.75, \
               label="Batch (%0.4f)"%(self.epoch_store[BATCH_SOURCE_LOGPDF][target_source].values[-1]/self.dims_dict[target_source]) )
      if self.n_test > 0:
        pp.plot( self.epoch_store[TEST_SOURCE_LOGPDF]["Epoch"].values, \
               self.epoch_store[TEST_SOURCE_LOGPDF][target_source]/self.dims_dict[target_source], 'o-', \
               color=self.source2mediumcolor[target_source],\
               mec=self.source2darkcolor[target_source], mew=2, \
               mfc=self.source2lightcolor[target_source], lw=3, \
               ms = 8, \
               label="Test  (%0.4f)"%(self.epoch_store[TEST_SOURCE_LOGPDF][target_source].values[-1]/self.dims_dict[target_source])  )

      if self.n_val > 0:
        pp.plot( self.epoch_store[VAL_SOURCE_LOGPDF]["Epoch"].values, \
               self.epoch_store[VAL_SOURCE_LOGPDF][target_source]/self.dims_dict[target_source], 'v-', \
               color=self.source2darkcolor[target_source],\
               mec=self.source2darkcolor[target_source], mew=2, \
               mfc=self.source2lightcolor[target_source], lw=3, \
               ms = 8, \
               label="Val  (%0.4f)"%(self.epoch_store[VAL_SOURCE_LOGPDF][target_source].values[-1]/self.dims_dict[target_source])  )

      
      if idx==0:
        pp.ylabel("log p(x|z)") #%(target_source))
      pp.legend(loc="lower right")
      pp.title( "%s"%(target_source))
      pp.xlabel("Epoch")
    
    pp.grid('on')
    #pdb.set_trace()
  
    pp.savefig( self.viz_filename_log_pdf_sources_per_gene, dpi = 300, fmt="png", bbox_inches = "tight")

    f = pp.figure()
    legends  = []
    colours = "bgr"
    fill_colours = ["lightblue","lightgreen","lightred"]
    n_sources = len(self.arch_dict[TARGET_SOURCES])
    for idx,target_source in zip( range(n_sources),self.arch_dict[TARGET_SOURCES]):
      s = f.add_subplot(1,n_sources,idx+1)
      pp.plot( self.epoch_store[BATCH_SOURCE_LOGPDF]["Epoch"].values, 
               self.epoch_store[BATCH_SOURCE_LOGPDF][target_source]/self.dims_dict[target_source], 's--', \
               color=self.source2mediumcolor[target_source], \
               mec=self.source2darkcolor[target_source], mew=1, \
               mfc=self.source2lightcolor[target_source], lw=1, \
               ms = 5, \
               alpha=0.75, \
               label="Batch (%0.4f)"%(self.epoch_store[BATCH_SOURCE_LOGPDF][target_source].values[-1]/self.dims_dict[target_source]) )
      if self.n_test > 0:
        query1 = self.epoch_store[TEST_FILL_LOGLIK]["Target"] == target_source
        query = query1#&query2
        loglik_df = self.epoch_store[TEST_FILL_LOGLIK][query]
        epochs = loglik_df["Epoch"].values
        loglik = loglik_df["LogLik"].values
        if len(loglik) == 0:
          continue
        pp.plot( epochs, loglik, 'o-', \
               color=self.source2darkcolor[target_source],\
               mec=self.source2darkcolor[target_source], mew=1, \
               mfc=self.source2lightcolor[target_source], lw=2, \
               ms = 8, \
               label="Test (%0.4f)"%(loglik[-1]) )

      if self.n_val > 0:
        query1 = self.epoch_store[VAL_FILL_LOGLIK]["Target"] == target_source
        query = query1#&query2
        loglik_df = self.epoch_store[VAL_FILL_LOGLIK][query]
        epochs = loglik_df["Epoch"].values
        loglik = loglik_df["LogLik"].values
        if len(loglik) == 0:
          continue
        
        pp.plot( epochs, loglik, 'v-', \
               color=self.source2mediumcolor[target_source],\
               mec=self.source2lightcolor[target_source], mew=1, \
               mfc=self.source2darkcolor[target_source], lw=2, \
               ms = 8, \
               label="Val (%0.4f)"%(loglik[-1]) )
                     
      if idx==0:
        pp.ylabel("log p(x|z)") #%(target_source))
      pp.legend(loc="lower right")
      pp.title( "%s"%(target_source))
      pp.xlabel("Epoch")
    
    pp.grid('on')
    #pdb.set_trace()
  
    pp.savefig( self.viz_filename_log_pdf_sources_per_gene_fill, dpi = 300, fmt="png", bbox_inches = "tight")

    f = pp.figure(figsize=(12,10))
    legends  = []
    #colours = "bgr"
    #fill_colours = ["lightblue","lightgreen","lightred"]
    n_sources = len(self.arch_dict[TARGET_SOURCES])
    for idx,target_source in zip( range(n_sources),self.arch_dict[TARGET_SOURCES]):
      s = f.add_subplot(1,n_sources,idx+1)
      inputs = "RNA+DNA+METH"
      if self.n_test > 0:
        query1 = self.epoch_store[TEST_FILL_ERROR]["Target"] == target_source
        #query2 = self.epoch_store[TEST_FILL_ERROR]["Inputs"] == inputs
        query = query1#&query2
        df = self.epoch_store[TEST_FILL_ERROR][query]
        epochs = df["Epoch"].values
        loglik = df["Error"].values
        if len(loglik) == 0:
          continue
        pp.plot( epochs, loglik, 'o-', \
                 color=self.source2darkcolor[target_source],\
                 mec=self.source2darkcolor[target_source], mew=1, \
                 mfc=self.source2lightcolor[target_source], lw=2, \
                 ms = 8, \
                 label="Test (%0.6f)"%(loglik[-1]) )
      if self.n_val > 0:
        query1 = self.epoch_store[VAL_FILL_ERROR]["Target"] == target_source
        #query2 = self.epoch_store[TEST_FILL_ERROR]["Inputs"] == inputs
        query = query1#&query2
        df = self.epoch_store[VAL_FILL_ERROR][query]
        epochs = df["Epoch"].values
        loglik = df["Error"].values
        if len(loglik) == 0:
          continue
        pp.plot( epochs, loglik, 'v-', \
                 color=self.source2mediumcolor[target_source],\
                 mec=self.source2darkcolor[target_source], mew=1, \
                 mfc=self.source2lightcolor[target_source], lw=2, \
                 ms = 8, \
                 label="Val (%0.6f)"%(loglik[-1]) )
      
      if idx==0:
        pp.ylabel("Error") #%(target_source))
      pp.legend(loc="upper right")
      pp.title( "%s"%(target_source))
      pp.xlabel("Epoch")
    
    pp.grid('on')
    #pdb.set_trace()
  
    pp.savefig( self.viz_filename_error_sources_per_gene_fill, dpi = 300, fmt="png", bbox_inches = "tight")

    pp.close('all')
    
  # ---------------------------------- #
  # ------- MODEL CALLBACKS
  # ---------------------------------- #  
  def SaveModel( self, sess, info_dict ):
    print "** SAVE Model"
    self.model_store.open()
    self.network.SaveModel( self.model_store )
    self.model_store.close()
    
  def VizModel( self, sess, info_dict ): 
    print "** VIZ Model"
    self.model_store.open()
    #pdb.set_trace()
    
    keys = self.model_store.keys()
    
    #dum,layer_name, W_or_b, W_or_b_id = k.split("/")
    old_layer = ""
    needs_closing=False
    for k in keys:
      dum,layer_name, W_or_b, W_or_b_id = k.split("/")
      if W_or_b == "b":
        continue
      #print "processing %s"%(k)
      if old_layer != layer_name:
        if needs_closing is True:
          #print "  closing figure, ",old_layer
          pp.legend()
          pp.suptitle(old_layer)
          pp.savefig( self.viz_filename_weights + "%s.png"%old_layer, fmt="png", bbox_inches = "tight")
          pp.close(fig_)
          needs_closing = False
          
        if W_or_b == "W":
          #print "  new figure"
          fig_ = pp.figure()
          ax1_ = fig_.add_subplot(121)
          ax2_ = fig_.add_subplot(122)
          needs_closing = True

      if W_or_b == "W":
        #print "  adding weights, ",layer_name
        W = np.squeeze( self.model_store[k].values ).flatten()
        ax1_.hist( W, 20, normed=True, alpha=0.5, label = "%s/%s"%(layer_name,W_or_b_id) )
        pp.grid('on')
        ax2_.plot( np.sort(W), lw=2, alpha=0.85, label = "%s/%s"%(layer_name,W_or_b_id) )
        pp.grid('on')
        needs_closing = True
        #pdb.set_trace()
        
      old_layer = layer_name
    if needs_closing:
      #print "  closing figure, ",old_layer
      pp.legend()
      pp.suptitle(old_layer)
      pp.savefig( self.viz_filename_weights + "%s.png"%old_layer, fmt="png", bbox_inches = "tight")
      pp.close(fig_)
      needs_closing = False
    # try:
    #   rec_rna_weights = self.model_store[ "/rec_hidden1/W/0" ].values.flatten()
    #   f = pp.figure()
    #   pp.hist(  rec_rna_weights, 50, normed=True, alpha=0.5 )
    #   pp.grid('on')
    #   pp.savefig( self.viz_filename_weights_rec_rna, dpi = 300, fmt="png", bbox_inches = "tight")
    #   pp.close(f)
    # except:
    #   print "** could not viz any model"
    self.model_store.close()
    pp.close('all')
      
  def SaveTestLatent( self, sess, info_dict ):
    print "** SAVE Latent"
    self.latent_store.open() # = OpenHdfStore(self.savedir, self.store_name, mode="a" )
    
    feed_dict          = info_dict[ VAL_FEED_DICT ]
    feed_imputation    = info_dict[ VAL_FEED_IMPUTATION ]
    barcodes           = feed_imputation[BARCODES]
    
    z_rec_space_tensor = self.network.GetLayer( REC_Z_SPACE).tensor
    #z_dna_rec_space_tensor = self.network.GetLayer( REC_Z_SPACE+"_dna" ).tensor
    z_rna_rec_space_tensor = self.network.GetLayer( REC_Z_SPACE+"_rna" ).tensor
    z_meth_rec_space_tensor = self.network.GetLayer( REC_Z_SPACE+"_meth" ).tensor
    z_mirna_rec_space_tensor = self.network.GetLayer( REC_Z_SPACE+"_mirna" ).tensor
    
    z_gen_space_tensor = self.network.GetLayer( GEN_Z_SPACE ).tensor


    #z_rec_space_dna        = sess.run( z_dna_rec_space_tensor, feed_dict = feed_dict )
    z_rec_space_rna        = sess.run( z_rna_rec_space_tensor, feed_dict = feed_dict )
    z_rec_space_meth        = sess.run( z_meth_rec_space_tensor, feed_dict = feed_dict )
    z_rec_space_mirna        = sess.run( z_mirna_rec_space_tensor, feed_dict = feed_dict )
    
    z_rec_space        = sess.run( z_rec_space_tensor, feed_dict = feed_dict )
    z_gen_space        = sess.run( z_gen_space_tensor, feed_dict = feed_dict )
    
    
    for idx, z_s in zip( range(len(z_rec_space)),z_rec_space ):
      #self.latent_store[ REC_Z_SPACE +"_dna" + "/s%d/"%(idx)] = pd.DataFrame( z_rec_space_dna[idx], index = barcodes, columns=self.z_columns)
      self.latent_store[ REC_Z_SPACE +"_rna" + "/s%d/"%(idx)] = pd.DataFrame( z_rec_space_rna[idx], index = barcodes, columns=self.z_columns)
      self.latent_store[ REC_Z_SPACE +"_meth" + "/s%d/"%(idx)] = pd.DataFrame( z_rec_space_meth[idx], index = barcodes, columns=self.z_columns)
      self.latent_store[ REC_Z_SPACE +"_mirna" + "/s%d/"%(idx)] = pd.DataFrame( z_rec_space_mirna[idx], index = barcodes, columns=self.z_columns)
      self.latent_store[ REC_Z_SPACE + "/s%d/"%(idx)] = pd.DataFrame( z_s, index = barcodes, columns=self.z_columns)

    for idx, z_s in zip( range(len(z_gen_space)),z_gen_space ):
      self.latent_store[ GEN_Z_SPACE + "/s%d/"%(idx)] = pd.DataFrame( z_s, index = barcodes, columns=self.z_columns)

    self.latent_store.close()
    
  def VizLatent( self, sess, info_dict ): 
    print "** VIZ Latent"
    self.latent_store.open()
  
    #pdb.set_trace()
    #rec_z_dna = self.latent_store[ REC_Z_SPACE + "_dna" + "/s%d/"%(0)]
    rec_z_rna = self.latent_store[ REC_Z_SPACE + "_rna" + "/s%d/"%(0)]
    rec_z_meth = self.latent_store[ REC_Z_SPACE + "_meth" + "/s%d/"%(0)]
    #pdb.set_trace()
    rec_z_mirna = self.latent_store[ REC_Z_SPACE + "_mirna" + "/s%d/"%(0)]
    rec_z = self.latent_store[ REC_Z_SPACE + "/s%d/"%(0)]
    mean_gen_z = self.latent_store[ GEN_Z_SPACE + "/s%d/"%(0)]

    obs = info_dict['val_feed_imputation']['observed_sources']
    #dna_obs = info_dict['test_feed_imputation'][DNA_OBSERVATIONS]
    
    dna_obs = obs[:,self.observed_source2idx[DNA]].astype(bool)
    rna_obs = obs[:,self.observed_source2idx[RNA]].astype(bool)
    meth_obs = obs[:,self.observed_source2idx[METH]].astype(bool)
    mirna_obs = obs[:,self.observed_source2idx[miRNA]].astype(bool)
    
    mean_rec_z      = rec_z.mean().values
    std_rec_z      = rec_z.std().values
    #mean_rec_z_dna  = rec_z_dna.mean().values
    mean_rec_z_rna  = rec_z_rna.mean().values
    mean_rec_z_meth = rec_z_meth.mean().values
    mean_rec_z_mirna = rec_z_mirna.mean().values
    #std_rec_z_dna  = rec_z_dna.std().values
    std_rec_z_rna  = rec_z_rna.std().values
    std_rec_z_meth = rec_z_meth.std().values
    std_rec_z_mirna = rec_z_mirna.std().values
    
    mean_gen_z_mean = mean_gen_z.mean().values
    mean_gen_z_std  = mean_gen_z.std().values
    
    #pdb.set_trace()
    f = pp.figure()
    pp.plot( mean_gen_z_mean, "ko", lw=1 )
    pp.plot( mean_gen_z_mean + 2*mean_gen_z_std, "k-", lw=0.5 )
    pp.plot( mean_gen_z_mean - 2*mean_gen_z_std, "k-", lw=0.5 )
    pp.fill_between( np.arange(len(mean_gen_z_mean)), mean_gen_z_mean - 2*mean_gen_z_std, mean_gen_z_mean + 2*mean_gen_z_std, color="k", alpha=0.5 )
    sns.violinplot( x=None, y=None, data=mean_rec_z)
    pp.grid('on')
    pp.savefig( self.viz_filename_z_rec_on_z_gen, dpi = 300, fmt="png", bbox_inches = "tight")
    pp.close(f)
    
    if self.n_z == 2:
      sp_a = 1
      sp_b = 2
    elif self.n_z < 10:
      sp_a = 3
      sp_b = 3
    elif self.n_z == 10:
      sp_a = 2
      sp_b = 5
    elif self.n_z <=16:
      sp_a = 4
      sp_b = 4
    elif self.n_z <= 25:
      sp_a = 5
      sp_b = 5
    elif self.n_z <= 36:
      sp_a = 6
      sp_b = 6
    elif self.n_z <= 80:
      sp_a = 8
      sp_b = 10
    max_figs = sp_a*sp_b
    
    f = pp.figure(figsize=(14,12))
    for z_idx in range( min(max_figs,self.n_z) ):
      #z_idx = 0
      I = np.argsort( rec_z.values[:,z_idx] )
      x = np.arange(len(I))
      pp.subplot(sp_a,sp_b,z_idx+1)
      pp.plot( x[mirna_obs[I]], rec_z_mirna.values[I,z_idx][mirna_obs[I]], 'o', \
               color=self.source2mediumcolor[miRNA],\
               mec=self.source2darkcolor[miRNA], mew=0.5, \
               mfc=self.source2lightcolor[miRNA], lw=2, \
               ms = 4, \
               alpha=0.5,\
               label="z-miRNA" )
      pp.plot( x[rna_obs[I]], rec_z_rna.values[I,z_idx][rna_obs[I]], 'o', \
               color=self.source2mediumcolor[RNA],\
               mec=self.source2darkcolor[RNA], mew=0.5, \
               mfc=self.source2lightcolor[RNA], lw=2, \
               ms = 4, \
               alpha=0.5,\
               label="z-RNA" )
      pp.plot( x[meth_obs[I]], rec_z_meth.values[I,z_idx][meth_obs[I]], 'o', \
               color=self.source2mediumcolor[METH],\
               mec=self.source2darkcolor[METH], mew=0.5, \
               mfc=self.source2lightcolor[METH], lw=2, \
               ms = 4, \
               alpha=0.5,\
               label="z-METH" )
      # pp.plot( x[dna_obs[I]], rec_z_dna.values[I,z_idx][dna_obs[I]], 'o', \
      #          color=self.source2mediumcolor[DNA],\
      #          mec=self.source2darkcolor[DNA], mew=0.5, \
      #          mfc=self.source2lightcolor[DNA], lw=2, \
      #          ms = 4, \
      #          alpha=0.5,\
      #          label="z-DNA" )
      pp.plot( rec_z.values[I,z_idx], '.', \
               color='k',\
               mec='k', mew=0.5, \
               mfc='w', lw=2, \
               ms = 4, \
               alpha=0.5,\
               label="z" )
    
    pp.savefig( self.viz_filename_z_rec_scatter, dpi = 300, fmt="png", bbox_inches = "tight")
    pp.close()
    
    pp.close('all')

      
    self.latent_store.close()
    
  # ---------------------------------- #
  # ------- other functions
  # ---------------------------------- #
  def AddSeries( self, store, store_key, values=[], columns=[] ):
    store.open()

    try:
      store[store_key]
    except:
      print "AddSeries: Cannot access store with key %s"%(store_key)
      #pdb.set_trace()
      store[store_key] = pd.DataFrame( [], columns = columns )      
    
    s = pd.Series( values, index = store[ store_key ].columns)
  
    store[ store_key ] = store[ store_key ].append( s, ignore_index = True )
    
    store.close()
    
  def PrintRow( self, store, key, row_index = -1 ):
    store.open()
    n = len(store[key])
    s = store[key][n-1:n]
    print key
    print s
    store.close()
  
  def CountSourcesInDict( self, d ):
    counts = {}
    if d.has_key(RNA_TARGET_MASK):
      n_rna = d[RNA_TARGET_MASK].sum()
    else:
      n_rna = len(d[BARCODES])

    if d.has_key(DNA_TARGET_MASK):
      n_dna = d[DNA_TARGET_MASK].sum()
    else:
      n_dna = len(d[BARCODES])

    if d.has_key(METH_TARGET_MASK):
      n_meth = d[METH_TARGET_MASK].sum()
    else:
      n_meth = len(d[BARCODES])
      
    if d.has_key(miRNA_TARGET_MASK):
      n_mirna = d[miRNA_TARGET_MASK].sum()
    else:
      n_mirna = len(d[BARCODES])
   
    # HACK to get the sizes correct
    n_batch_size = len(d[BARCODES])
    counts[RNA] = n_rna
    counts[DNA] = n_dna
    counts[METH] = n_meth
    counts[miRNA] = n_mirna
    counts[RNA+"_b"] = n_rna
    counts[DNA+"_b"] = n_dna
    counts[METH+"_b"] = n_meth
    counts[miRNA+"_b"] = n_mirna
    return counts
    
  def Epoch( self, epoch_key, sess, info_dict, epoch, feed_dict, impute_dict, mode ):  
    barcodes = impute_dict[BARCODES]
    batch_tensor_evals = sess.run( self.network.batch_log_tensors, feed_dict = feed_dict )
    
    batch_counts = self.CountSourcesInDict( impute_dict )
    
    n_batch = []             
    for source in self.arch_dict[TARGET_SOURCES]:
      n_batch.append( batch_counts[source] )
    n_batch = np.array(n_batch).astype(float)
    
    n_batch_size = len(impute_dict[BARCODES])   
    
    log_p_z         = batch_tensor_evals[2]/float(n_batch_size)
    log_q_z         = batch_tensor_evals[3]/float(n_batch_size)
    
    # normalize by nbr observed for each source
    log_p_source_z_values = batch_tensor_evals[4:]/n_batch
    
    #print np.sort(info_dict[BATCH_IDS])
    new_log_p_x_given_z = log_p_source_z_values.sum()
    lower_bound = log_p_z-log_q_z + new_log_p_x_given_z
    
    new_values = [epoch, lower_bound, new_log_p_x_given_z, log_p_z, log_q_z]
    new_values.extend( log_p_source_z_values )

    self.AddSeries(  self.epoch_store, epoch_key, values = new_values, columns = self.network.batch_log_columns )
    
    epoch_log_p_source_z_values = [epoch]
    epoch_log_p_source_z_values.extend( log_p_source_z_values )
    epoch_source_columns = ['Epoch']
    epoch_source_columns.extend(self.arch_dict[TARGET_SOURCES])
    
    if mode == "BATCH":
      self.AddSeries(  self.epoch_store, BATCH_SOURCE_LOGPDF, values = epoch_log_p_source_z_values, columns = epoch_source_columns )
    elif mode == "TEST" and self.n_test>0:
      self.AddSeries(  self.epoch_store, TEST_SOURCE_LOGPDF, values = epoch_log_p_source_z_values, columns = epoch_source_columns )
      self.PrintRow( self.epoch_store, epoch_key )
    elif mode == "VAL" and self.n_val>0:
      self.AddSeries(  self.epoch_store, VAL_SOURCE_LOGPDF, values = epoch_log_p_source_z_values, columns = epoch_source_columns )
      self.PrintRow( self.epoch_store, epoch_key )
    
    if mode == "TEST" or mode == "VAL":
      #print "!!!!!!!!"
      #print "testing product model"
      input_observations = impute_dict[INPUT_OBSERVATIONS]

      tensors = []
      tensors.extend( self.network.GetTensor("rec_z_space_rna") )
      #tensors.extend( self.network.GetTensor("rec_z_space_dna") )
      tensors.extend( self.network.GetTensor("rec_z_space_meth") )
      tensors.extend( self.network.GetTensor("rec_z_space") )
      
      if self.network is tcga_encoder.models.networks.ConditionalVariationalAutoEncoder:
        tensors.extend( self.network.GetTensor("gen_z_space") )

      tensor_evals = sess.run( tensors, feed_dict = feed_dict )

      rna_mean  = tensor_evals[0]
      rna_var   = tensor_evals[1]
      #dna_mean  = tensor_evals[2]
      #dna_var   = tensor_evals[3]
      meth_mean = tensor_evals[2]
      meth_var  = tensor_evals[3]
      z_mean    = tensor_evals[4]
      z_var     = tensor_evals[5]
      if self.network is tcga_encoder.models.networks.ConditionalVariationalAutoEncoder:
        z_mean_g  = tensor_evals[6]
        z_var_g   = tensor_evals[7]
      
      self.fill_store.open()
      self.fill_store[ "%s/Z/%s/mu"%(mode,RNA)]   = pd.DataFrame( rna_mean, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/%s/var"%(mode,RNA)]  = pd.DataFrame( rna_var, index = barcodes, columns = self.z_columns )
      #self.fill_store[ "%s/Z/%s/mu"%(mode,DNA)]   = pd.DataFrame( dna_mean, index = barcodes, columns = self.z_columns )
      #self.fill_store[ "%s/Z/%s/var"%(mode,DNA)]  = pd.DataFrame( dna_var, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/%s/mu"%(mode,METH)]  = pd.DataFrame( meth_mean, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/%s/var"%(mode,METH)] = pd.DataFrame( meth_var, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/rec/mu"%mode]        = pd.DataFrame( z_mean, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/rec/var"%mode]       = pd.DataFrame( z_var, index = barcodes, columns = self.z_columns )
      if self.network is tcga_encoder.models.networks.ConditionalVariationalAutoEncoder:
        self.fill_store[ "%s/Z/gen/mu"%mode]        = pd.DataFrame( z_mean_g, index = barcodes, columns = self.z_columns )
        self.fill_store[ "%s/Z/gen/var"%mode]       = pd.DataFrame( z_var_g, index = barcodes, columns = self.z_columns )
      self.fill_store.close()
     
    
  def NextBatch( self, batch_ids, mode = "BATCH" ):
    return self.FillBatch( self.train_barcodes[batch_ids], mode )
    
  def TestBatch( self ):
    return self.FillBatch( self.test_barcodes, mode = "TEST" )
    
  def ValBatch( self ):
    return self.FillBatch( self.validation_barcodes, mode = "VAL" )

  def FillBatch( self, batch_barcodes, mode = "BATCH" ):
    batch = OrderedDict()
    
    n = len(batch_barcodes)
    #observed_columns = self.store[self.OBSERVED_key].columns[1:]
    #obs2idx = OrderedDict()
    #for s,idx in zip( self.observed_order, range(len(self.observed_order)) ):
    #  obs2idx[ s ] = idx
      
    batch_observed = self.data_store[self.OBSERVED_key].loc[ batch_barcodes ].values
    batch[ "observed_sources" ] = batch_observed
    
    batch[ "barcodes" ]         = batch_barcodes
    
    batch[DNA_OBSERVATIONS] = batch_observed[ :, self.observed_source2idx[DNA] ]
    
    for layer_name, layer in self.network.layers.iteritems():
      #print layer_name, layer  
      if layer_name == RNA_INPUT:
        batch_data = self.data_store[self.RNA_key].loc[ batch_barcodes ]
        nans = np.isnan( batch_data.values )
        batch_data_values = batch_data.values
        if mode == "BATCH":
         batch_data_values = self.AddRnaNoise( batch_data.values, rate = 0.1 )
          
        batch[ layer_name ] = self.NormalizeRnaInput( batch_data_values )
        batch[ layer_name ][nans] = 0
        #pdb.set_trace()
        
      elif layer_name == RNA_TARGET:
        batch_data = self.data_store[self.RNA_key].loc[ batch_barcodes ]
        nans = np.isnan( batch_data.values )
        
        batch_data_values = batch_data.values
        # if mode == "BATCH":
        #   batch_data_values = self.AddRnaNoise( batch_data.values, rate = 0.1 )
          
        batch[ layer_name ] = self.NormalizeRnaTarget( batch_data.fillna( 0 ).values )
        batch[ layer_name ][nans] = 0
        
      elif layer_name == RNA_TARGET_MASK:
        #batch_data = self.store[self.RNA_key].loc[ batch_barcodes ]
        batch[ layer_name ] = batch_observed[:,self.observed_source2idx[ RNA ]].astype(bool)
        
        nbr_observed = batch_observed[:,self.observed_source2idx[ RNA ]].astype(bool).sum()
      
      elif layer_name == miRNA_INPUT:
        batch_data = self.data_store[self.miRNA_key].loc[ batch_barcodes ]
        nans = np.isnan( batch_data.values )
        batch_data_values = batch_data.values
        if mode == "BATCH":
         batch_data_values = self.AddmiRnaNoise( batch_data.values, rate = 0.1 )
          
        batch[ layer_name ] = self.NormalizemiRnaInput( batch_data_values )
        batch[ layer_name ][nans] = 0
        #pdb.set_trace()
        
      elif layer_name == miRNA_TARGET:
        batch_data = self.data_store[self.miRNA_key].loc[ batch_barcodes ]
        nans = np.isnan( batch_data.values )
        
        batch_data_values = batch_data.values
        # if mode == "BATCH":
        #   batch_data_values = self.AddRnaNoise( batch_data.values, rate = 0.1 )
          
        batch[ layer_name ] = self.NormalizemiRnaTarget( batch_data.fillna( 0 ).values )
        batch[ layer_name ][nans] = 0
        
      elif layer_name == miRNA_TARGET_MASK:
        batch[ layer_name ] = batch_observed[:,self.observed_source2idx[ miRNA ]].astype(bool)
        nbr_observed = batch_observed[:,self.observed_source2idx[ miRNA ]].astype(bool).sum()
        
      elif layer_name == DNA_TARGET_MASK:
        #batch_data = self.store[self.RNA_key].loc[ batch_barcodes ]
        batch[ layer_name ] = batch_observed[:,self.observed_source2idx[ DNA ]].astype(bool)
        
        nbr_observed = batch_observed[:,self.observed_source2idx[ DNA ]].astype(bool).sum()
        
        #print "-- DNA observed = %d"%(nbr_observed)
        
      elif layer_name == DNA_INPUT or layer_name == DNA_TARGET:
        dna_data = np.zeros( (len(batch_barcodes), self.dna_dim) )
        #for idx,DNA_key in zip(range(len(self.DNA_keys)-1),self.DNA_keys[:-1]):
        for idx,DNA_key in zip(range(len(self.DNA_keys)),self.DNA_keys):
          batch_data = self.data_store[DNA_key].loc[ batch_barcodes ].fillna( 0 ).values
          if mode == "TEST" or mode == "VAL" or mode == "TRAIN":
            dna_data += batch_data
          else:
            if layer_name == DNA_TARGET or layer_name == DNA_INPUT:
            #if layer_name == DNA_TARGET:
              dna_data = self.AddDnaNoise( batch_data, rate = 0.1 )
          #
          #dna_data.append(batch_data.fillna( 0 ).values)
        
        batch[ layer_name ] = np.minimum(1.0,dna_data)# np.array( dna_data )
        #pdb.set_trace()
        #print "DNA batch count: %d"%(batch[ layer_name ].sum())
        
      elif layer_name == METH_INPUT :
        batch_data = self.data_store[self.METH_key].loc[ batch_barcodes ]
        nans = np.isnan( batch_data.values )

        batch_data_values = batch_data.values
        if mode == "BATCH":
         batch_data_values = self.AddMethNoise( batch_data.values, rate = 0.1 )
          
        batch[ layer_name ] = self.NormalizeMethInput( batch_data_values )
        batch[ layer_name ][nans] = 0
        #pdb.set_trace()
        
      elif layer_name == METH_TARGET:
        batch_data = self.data_store[self.METH_key].loc[ batch_barcodes ]
        nans = np.isnan( batch_data.values )
        batch_data_values = batch_data.values
        # if mode == "BATCH":
        #   batch_data_values = self.AddMethNoise( batch_data.values, rate = 0.1 )
          
        batch[ layer_name ] = self.NormalizeMethTarget( batch_data_values )
        batch[ layer_name ][nans] = 0
        
      elif layer_name == METH_TARGET_MASK:
        #batch_data = self.store[self.RNA_key].loc[ batch_barcodes ]
        batch[ layer_name ] = batch_observed[:,self.observed_source2idx[ METH ]].astype(bool)
        
        nbr_observed = batch_observed[:,self.observed_source2idx[ METH ]].astype(bool).sum()
                  
      elif layer_name == TISSUE_INPUT or layer_name == TISSUE_TARGET:
        batch_data = self.data_store[self.TISSUE_key].loc[ batch_barcodes ]
        batch[ layer_name ] = batch_data.fillna( 0 ).values
       
      elif layer_name == U_Z:
        batch_data = np.random.randn( n, layer.tensor.get_shape()[1].value ).astype(np.float32)
        batch[ layer_name ] = batch_data
      
      elif layer_name == INPUT_WEIGHTED_OBSERVATIONS:
        batch_data = batch[ "observed_sources" ][:,self.observation_source_indices]
        #print "-- observed : |R|= %d  |D| = %d  |R*D| = %d |R*1-D| = %d  |1-R*D| = %d  neither=%d"%(sm[0],sm[1], both, only_first, only_second, neither)
        
        if mode == "BATCH":
          # find all data where batch observations has more than one source available
          I = pp.find( batch_data.sum(1) == 3 )
          for i_idx in I:
            u=np.random.rand()
            # for half of these, select one observation at random to turn off
            if u < self.batcher_rates[0]:
              J = pp.find( batch_data[i_idx,:] )
              j = np.random.randint( len(J) )
              #pdb.set_trace()
              batch_data[i_idx,J[j]] = 0
            
            # randonly set 2 to 0
            elif u>self.batcher_rates[1]:
              J = pp.find( batch_data[i_idx,:] )
              jj = np.random.permutation( len(J) )
              #pdb.set_trace()
              batch_data[i_idx,J[jj[0]]] = 0
              batch_data[i_idx,J[jj[1]]] = 0
              
          II = pp.find( batch_data.sum(1) == 2 )
          for i_idx in II:
            u=np.random.rand()
            # for half of these, select one observation at random to turn off
            if u < self.batcher_rates[2]:
              J = pp.find( batch_data[i_idx,:] )
              j = np.random.randint( len(J) )
              #pdb.set_trace()
              batch_data[i_idx,J[j]] = 0
              
        w = float(len(self.observation_source_indices))/batch_data.sum(1)
        #w = batch[ "observed_sources" ][:,self.observation_source_indices].sum(1)
        
        
        sm = batch_data.sum(0)
        both = np.sum( batch_data.sum(1)==2 )
        only_first  = np.sum( batch_data[:,0]*(1-batch_data[:,1]))
        only_second = np.sum( batch_data[:,1]*(1-batch_data[:,0]))
        neither = np.sum( (1-batch_data[:,1])*(1-batch_data[:,0]))
              
        batch[ layer_name ] = w[:,np.newaxis]*batch_data
        
        #batch[ layer_name ] = batch_data
        
        
      elif layer_name == INPUT_OBSERVATIONS:
        batch_data = batch[ "observed_sources" ][:,self.observed_product_sources]
        #print "-- observed : |R|= %d  |D| = %d  |R*D| = %d |R*1-D| = %d  |1-R*D| = %d  neither=%d"%(sm[0],sm[1], both, only_first, only_second, neither)
      
        #pdb.set_trace()
        if mode == "BATCH":
          # find all data where batch observations has more than one source available
          I = pp.find( batch_data.sum(1) == 3 )
          for i_idx in I:
            u=np.random.rand()
            # for half of these, select one observation at random to turn off
            if u < self.batcher_rates[0]:
              J = pp.find( batch_data[i_idx,:] )
              j = np.random.randint( len(J) )
              #pdb.set_trace()
              batch_data[i_idx,J[j]] = 0
            
            # randonly set 2 to 0
            elif u>self.batcher_rates[1]:
              J = pp.find( batch_data[i_idx,:] )
              jj = np.random.permutation( len(J) )
              #pdb.set_trace()
              batch_data[i_idx,J[jj[0]]] = 0
              batch_data[i_idx,J[jj[1]]] = 0
              
          II = pp.find( batch_data.sum(1) == 2 )
          for i_idx in II:
            u=np.random.rand()
            # for half of these, select one observation at random to turn off
            if u < self.batcher_rates[2]:
              J = pp.find( batch_data[i_idx,:] )
              j = np.random.randint( len(J) )
              #pdb.set_trace()
              batch_data[i_idx,J[j]] = 0
            
        w = float(len(self.observation_source_indices))/batch_data.sum(1)
        #w = batch[ "observed_sources" ][:,self.observation_source_indices].sum(1)
      
      
        sm = batch_data.sum(0)
        both = np.sum( batch_data.sum(1)==2 )
        only_first  = np.sum( batch_data[:,0]*(1-batch_data[:,1]))
        only_second = np.sum( batch_data[:,1]*(1-batch_data[:,0]))
        neither = np.sum( (1-batch_data[:,1])*(1-batch_data[:,0]))
            
            
      
      
        #batch[ layer_name ] = w[:,np.newaxis]*batch_data
        batch[ layer_name ] = batch_data
    
      elif layer_name == INPUT_MISSING:
        batch_data = batch[ "observed_sources" ][:,self.observed_product_sources]
        #print "-- observed : |R|= %d  |D| = %d  |R*D| = %d |R*1-D| = %d  |1-R*D| = %d  neither=%d"%(sm[0],sm[1], both, only_first, only_second, neither)
        
        if mode == "BATCH":
          # find all data where batch observations has more than one source available
          I = pp.find( batch_data.sum(1) > 1 )
          for i_idx in I:
            # for half of these, select one observation at random to turn off
            if np.random.rand() < 0.5:
              J = pp.find( batch_data[i_idx,:] )
              j = np.random.randint( len(J) )
              #pdb.set_trace()
              batch_data[i_idx,J[j]] = 0
              
        batch[ layer_name ] = 1-batch_data
        #pdb.set_trace()
      else:
        pass #print "cannot batch this source " + layer_name
        
      # if mode == "BATCH":
      #   print layer_name
      #   try:
      #     print batch[ layer_name ]
      #   except:
      #     pass
      #   pdb.set_trace()
    
    if mode != "BATCH":
      batch[ "beta" ] = 1.0
      batch["free_bits"] = 0.0
    else:
      batch[ "beta" ] = self.beta
      batch["free_bits"] = self.free_bits
      
    for layer_name, layer in self.network.dropouts.iteritems():
      #print "** Found dropout layer"
      if mode == "BATCH":
        #print "** setting dropout keep rate"
        batch[ layer_name ] = self.keep_rates[ layer_name ]
        
        
    return batch            
    
  def NormalizeMethInput( self, X ):
    return X
    return 2*X-1.0

  def NormalizeMethTarget( self, X ):
    return 0.0001+0.9999*X
    return X

  def NormalizeRnaInput( self, X ):
    return X
    return 2*X-1.0

  def NormalizeRnaTarget( self, X ):
    return 0.0001+0.9999*X
    return X

  def NormalizemiRnaInput( self, X ):
    return X
    return 2*X-1.0

  def NormalizemiRnaTarget( self, X ):
    return 0.0001+0.9999*X
    return X


  def AddMethNoise( self, X, rate=0.5 ):
    #return X
    
    a,b = X.shape

    x = X.flatten()

    #I = 
    #I=pp.find(x>0)
    #J=pp.find(x==0)

    r1 = np.random.rand(len(x)) < rate
    #r2 = np.random.rand(len(J)) < rate2

    x[r1]=1.0-x[r1]
    #x[J[r2]]=1

    return x.reshape((a,b))

  def AddmiRnaNoise( self, X, rate=0.5 ):
    #return X
    
    a,b = X.shape

    x = X.flatten()

    #I = 
    #I=pp.find(x>0)
    #J=pp.find(x==0)

    r1 = np.random.rand(len(x)) < rate
    #r2 = np.random.rand(len(J)) < rate2

    x[r1]=1.0-x[r1]
    #x[J[r2]]=1

    return x.reshape((a,b))

  def AddRnaNoise( self, X, rate=0.5 ):
    #return X
    
    a,b = X.shape

    x = X.flatten()

    #I = 
    #I=pp.find(x>0)
    #J=pp.find(x==0)

    r1 = np.random.rand(len(x)) < rate
    #r2 = np.random.rand(len(J)) < rate2

    x[r1]=1.0-x[r1]
    #x[J[r2]]=1

    return x.reshape((a,b))
    
  def AddDnaNoise( self, X, rate=0.5 ):
    #return X
    
    a,b = X.shape

    x = X.flatten()

    #I = 
    I=pp.find(x>0)
    #J=pp.find(x==0)

    r1 = np.random.rand(len(I)) < rate
    #r2 = np.random.rand(len(J)) < rate2

    x[I[r1]]=0
    #x[J[r2]]=1

    return x.reshape((a,b))
    
    
  def AddNoise( self, X, rate1=0.01, rate2=0.001 ):
    #return X
    
    a,b = X.shape

    x = X.flatten()

    I=pp.find(x>0)
    J=pp.find(x==0)

    r1 = np.random.rand(len(I)) < rate1
    r2 = np.random.rand(len(J)) < rate2

    x[I[r1]]=0
    x[J[r2]]=1

    return x.reshape((a,b))
    
  def AddNoiseOld( self, X, rate1, rate2 ):
    return X
    
    a,b = X.shape

    x = X.flatten()

    I=pp.find(x>0)
    J=pp.find(x==0)

    #r1 = np.random.rand(len(I)) < rate1
    #r2 = np.random.rand(len(J)) < rate2

    x[I] = rate1 + (1.0-rate1)*np.random.rand(len(I) )
    x[J] = rate2*np.random.rand(len(J))
    #x[J[r2]]=1

    return x.reshape((a,b))
     
  

    
      
      
      
            