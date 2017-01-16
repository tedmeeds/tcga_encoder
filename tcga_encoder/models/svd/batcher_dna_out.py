import tensorflow as tf
#from tensorflow import *
import pdb
from tcga_encoder.models.layers import *
from tcga_encoder.models.regularizers import *
from tcga_encoder.algorithms import *

#from models.vae.tcga_models import *
from tcga_encoder.utils.helpers import *
#from tcga_encoder.data import load_sources

#from utils.image_utils import *
#from tcga_encoder.definitions.tcga import *
#from data.data import *
#import pdb, os, sys
#from sklearn.metrics import roc_auc_score, roc_curve

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("talk")

import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
BATCH_SOURCE_LOGPDF = "Batch_SourceLogPdf"
TEST_SOURCE_LOGPDF = "Test_SourceLogPdf"
VAL_SOURCE_LOGPDF = "Val_SourceLogPdf"

TEST_FILL_LOGLIK  = "Test_LogLik"
TEST_FILL_ERROR  = "Test_Error"

VAL_FILL_LOGLIK  = "Val_LogLik"
VAL_FILL_ERROR  = "Val_Error"
  #
  # def __init__(self, network_name, network, data_dict, algo_dict, arch_dict, logging_dict, default_store_mode="w" ):
  #   self.network_name   = network_name
  #   #self.network        = network
  #   self.data_dict      = data_dict
  #   self.algo_dict      = algo_dict
  #   self.arch_dict      = arch_dict
  #   self.logging_dict   = logging_dict
  #   self.var_dict       = self.arch_dict[VARIABLES]
  #   self.default_store_mode   = default_store_mode
  #   self.validation_tissues = ["coad","laml","meso","read","ucec"]
  #   self.batcher_rates = [0.25,0.90,0.1] # A


class TCGABatcher( object ):
  def __init__(self, network_name, network, data_dict, algo_dict, arch_dict, logging_dict, default_store_mode="w" ):
    self.network_name   = network_name
    #self.network        = network
    self.data_dict      = data_dict
    self.algo_dict      = algo_dict
    self.arch_dict      = arch_dict
    self.logging_dict   = logging_dict
    self.var_dict       = self.arch_dict[VARIABLES]
    self.default_store_mode   = default_store_mode
    self.validation_tissues = ["coad","laml","meso","read","ucec"]
    self.batcher_rates = [0.25,0.90,0.1] # A
    #self.batcher_rates = [0.75,0.90,0.25] # B
    #self.batcher_rates = [1.0/8,0.95,0.05] # C
    #self.batcher_rates = [1.0/2,0.27,0.25] # D
    #self.batcher_rates = [0.25,0.90,0.5] # A
    
    
    self.Initialize()
    
    # these are tissues that have 0 or only tiny fully observed
    #dims = np.array([rna_test.shape[1],dna_0_test.shape[1],dna_1_test.shape[1],dna_2_test.shape[1],dna_3_test.shape[1],meth_test.shape[1]])
    dna_dim = self.dna_dim #/ self.n_dna_channels
    self.dims = np.array([self.rna_dim, dna_dim, self.meth_dim])
    self.cum_dims = np.cumsum(self.dims)
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
    self.tissue_dim = self.data_dict[DATASET].GetDimension(TISSUE)
    
    self.dims_dict = {RNA:self.rna_dim, DNA:self.dna_dim, TISSUE:self.tissue_dim, METH:self.meth_dim}
    self.data_store     = self.data_dict[DATASET].store
    self.batch_size     = self.algo_dict[BATCH_SIZE]
    #self.n_test         = self.algo_dict[N_TEST]
    self.n_full_train = self.algo_dict[N_TRAIN_FULL]
    self.n_non_full_train  = self.algo_dict[N_TRAIN_NON_FULL]
    self.train_mode = self.algo_dict[TRAIN_MODE]
    
    # keep BRCA in train and test sets, if False, move them all to validation
    self.include_brca = bool(self.algo_dict["include_brca"])
    self.nbr_brca_train = int(self.algo_dict["nbr_brca_train"])
    # keep non-full tissues in validation, otherwise move to train
    self.include_validation = bool(self.algo_dict["include_validation"])
    
    self.batch_imputation_dict = {}
    self.batch_feed_dict       = {}
    
    # store keys
    self.OBSERVED_key = CLINICAL+"/"+OBSERVED
    self.TISSUE_key   = CLINICAL+"/"+TISSUE
    self.RNA_key      = RNA+"/"+FAIR
    self.METH_key     = METH+"/"+FAIR
    self.DNA_keys     = [DNA+"/"+CHANNEL+"/%d"%i for i in range(self.n_dna_channels)]
    
    self.n_z            = self.var_dict[N_Z]
    self.z_columns = ["z%d"%z for z in range(self.n_z)]
    self.rna_genes = self.data_store[self.RNA_key].columns
    self.dna_genes = self.data_store[self.DNA_keys[0]].columns
    self.meth_genes = self.data_store[self.METH_key].columns
    self.tissue_names = self.data_store[self.TISSUE_key].columns
    self.tissue2color = OrderedDict()
    self.tissue2shape = OrderedDict()
    cmap = pp.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(self.tissue_names)))
    for idx,t in zip( range(len(colors)),self.tissue_names):
      self.tissue2color[t] = colors[idx]
      self.tissue2shape[t] = 'o'
      
    self.latent_store_name = self.network_name + "_" + LATENT
    self.latent_store = OpenHdfStore(self.savedir, self.latent_store_name, mode=self.default_store_mode )
    self.model_store_name = self.network_name + "_" + MODEL
    
    # open in "a" mode
    self.model_store = OpenHdfStore(self.savedir, self.model_store_name, mode="a" )
    
    self.epoch_store_name = self.network_name + "_" + EPOCH
    self.epoch_store = OpenHdfStore(self.savedir, self.epoch_store_name, mode=self.default_store_mode )
    
    self.fill_store_name = self.network_name + "_" + FILL
    self.fill_store = OpenHdfStore(self.savedir, self.fill_store_name, mode="a")
    
    self.fill_store.close()
    self.latent_store.close()
    self.model_store.close()
    self.epoch_store.close()
    
    self.source2darkcolor = {RNA:"darkblue", DNA:"darkgreen", METH:"darkred"}
    self.source2lightcolor = {RNA:"lightblue", DNA:"palegreen", METH:"lightsalmon"}
    self.source2mediumcolor = {RNA:"dodgerblue", DNA:"darksage", METH:"red"}
    
    self.source2mediumcolor[RNA+"+"+DNA]="turquoise"
    self.source2mediumcolor[RNA+"+"+METH] = "fuchsia"
    self.source2mediumcolor[DNA+"+"+METH] = "sandybrown"
    self.source2mediumcolor[RNA+"+"+DNA+"+"+METH]="grey"
    
    self.observed_order = self.data_store[self.OBSERVED_key].columns
    self.observed_source2idx = OrderedDict()
    for source, idx in zip(self.observed_order,range(len(self.observed_order)) ):
      self.observed_source2idx[source] = idx
    
    self.input_sources = self.arch_dict[INPUT_SOURCES]
    self.target_sources = self.arch_dict[TARGET_SOURCES]
    
    # pp.plot( epochs, loglik, self.input_combo2shape[inputs]+'-', \
    #          color=self.source2mediumcolor[target_source],\
    #          mec=self.source2darkcolor[target_source], mew=1, \
    #          mfc=self.input_combo2fillcolor[target_source], lw=2, \
    #          ms = self.input_combo2markersize[inputs], \
    #          label="%s  (%0.3f)"%(inputs,loglik[-1]) )
    #
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
          #self.input_combo2linecolor[inputs] = self.source2lightcolor[inputs]
          
        if nbr == 2:
          self.input_combo2fillcolor[inputs] = self.source2mediumcolor[inputs]
          self.input_combo2shape[inputs] = 's'
          self.input_combo2markersize[inputs] = 8
          
        if nbr == 3:
          self.input_combo2fillcolor[inputs] = self.source2mediumcolor[inputs]
          self.input_combo2shape[inputs] = 'o'
          self.input_combo2markersize[inputs] = 10
          

        
    self.observation_source_indices = []
    for source in self.arch_dict["sources"]:
      self.observation_source_indices.append( self.observed_source2idx[source] )

    self.observation_source_indices_input = []
    for source in self.arch_dict[INPUT_SOURCES]:
      self.observation_source_indices_input.append( self.observed_source2idx[source] )

    self.observation_source_indices_target = []
    for source in self.arch_dict[TARGET_SOURCES]:
      self.observation_source_indices_target.append( self.observed_source2idx[source] )                
    
    self.observed_product_sources = []
    for source in self.arch_dict["product_sources"]:
      self.observed_product_sources.append( self.observed_source2idx[source] )     
       
    print "** getting validations"
    #self.at_least_one_query = self.data_store[self.OBSERVED_key].values[:,self.observation_source_indices].sum(1)>0
    self.at_least_one_query = self.data_store[self.OBSERVED_key].values[:,self.observed_product_sources].sum(1)>0
    
    self.validation_tissue2barcodes = OrderedDict()
    
    self.obs_store_bc_2idx = OrderedDict()
    for idx,bc in zip( range(len(self.data_store[self.OBSERVED_key].index)),self.data_store[self.OBSERVED_key].index):
      self.obs_store_bc_2idx[bc] = idx
      
    self.observed_tissue_and_bcs = self.data_store[self.OBSERVED_key].index
    self.observation_tissues = np.array([s.split("_")[0] for s in self.observed_tissue_and_bcs])
    self.validation_obs_query = np.zeros( len(self.data_store[self.OBSERVED_key]), dtype=bool)
    
    coad_bc = "coad_tcga-t9-a92h"
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
        
      #query = self.data_store[self.OBSERVED_key].loc[self.validation_tissue2barcodes[ tissue ]].values[:,self.observation_source_indices].sum(1)>0
      
      ids = self.observation_tissues==tissue
      #self.validation_obs_query[i.values] = True
      for bc in self.validation_tissue2barcodes[ tissue ]:
        self.validation_obs_query[ self.obs_store_bc_2idx[bc] ] = True
      #self.validation_obs_query[ids] = True
      print tissue, self.validation_obs_query.sum()
      #if tissue=="coad":
      #  pdb.set_trace()
      
    coad_bc = "coad_tcga-t9-a92h"
    
    #pdb.set_trace()
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
    
    self.observed_batch_order = OrderedDict()
    self.observed_batch_order[RNA] = 0
    #self.observed_batch_order[DNA] = 1
    self.observed_batch_order[METH] = 1
    # if self.train_mode == ALL:
    #   print "************************"
    #   print "************************"
    #   print "ALL ALL ALL ALL ALL ALL "
    #   print "************************"
    #   print "************************"
    #   self.usable_observed_query = self.at_least_one_query*self.not_validation_query
    #
    #   #*(self.data_store[self.OBSERVED_key].values[:,self.observation_source_indices].sum(1)>0)
    # elif self.train_mode == FULL:
    #   print "************************"
    #   print "************************"
    #   print "FULL FULL FULL FULL FULL"
    #   print "************************"
    #   print "************************"
    #   self.usable_observed_query = self.at_least_one_query*self.not_validation_query*(self.data_store[self.OBSERVED_key].values.sum(1)==len(self.observed_order))
    # else:
    #   assert False, "must be ALL or FULL"
    
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
  
    if self.n_fully_observed <= self.n_full_train:
      print "==> setting test set from %d to %d"%(self.n_full_train,self.n_fully_observed)
      self.n_full_train = self.n_fully_observed
    
    # index into usable for test and train
    
    self.train_full_id_query = np.zeros( len(self.data_store[self.OBSERVED_key]), dtype=bool)
  
    np.random.seed( self.algo_dict['split_seed'] )
    self.train_full_ids = np.random.permutation( self.n_fully_observed )[:self.n_full_train]
    self.train_full_id_query[ self.full_observed_ids[self.train_full_ids] ] = 1
    
    self.test_full_id_query = (self.not_validation_query*self.fully_observed_query*(1-self.train_full_id_query).astype(np.bool) ).astype(np.bool)
    
    # all the usable, but not fully observed data
    self.non_full_observed_query = self.not_validation_query*self.at_least_one_query*(1-self.fully_observed_query).astype(np.bool)
    
    self.non_full_observed_ids = pp.find( self.non_full_observed_query )
    self.n_non_fully_observed = self.non_full_observed_query.sum()
    
    if self.n_non_fully_observed <= self.n_non_full_train:
      print "==> setting test set from %d to %d"%(self.n_non_full_train,self.n_non_fully_observed)
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
    
    if self.train_mode == ALL:
      print "************************"
      print "************************"
      print "ALL ALL ALL ALL ALL ALL "
      print "************************"
      print "************************"
      self.train_barcodes = np.union1d( self.train_full_barcodes, self.train_non_full_barcodes )
    else:
      print "************************"
      print "************************"
      print "FULL FULL FULL FULL FULL"
      print "************************"
      print "************************"
      self.train_barcodes = self.train_full_barcodes
    #
    # #pdb.set_trace()
    # # self.test_barcodes = self.usable_barcodes[self.test_id_query]
    #
    # # correct view of observed for train and test
    # self.test_observed  = self.data_store[self.OBSERVED_key][ self.test_id_query ]
    # self.train_observed = self.data_store[self.OBSERVED_key][ self.train_id_query ]
    #
    # # use these barcodes to locate sources
    # self.test_barcodes  = self.test_observed.index
    # self.train_barcodes = self.train_observed.index
  
    #pdb.set_trace()
    assert len(np.intersect1d( self.test_barcodes, self.train_barcodes)) == 0, "train and test are not mutually exclusive!!"
    assert len(np.intersect1d( self.test_barcodes, self.validation_barcodes)) == 0, "train and test are not mutually exclusive!!"
    assert len(np.intersect1d( self.train_barcodes, self.validation_barcodes)) == 0, "train and test are not mutually exclusive!!"
    
    print "TEST BARCODES: "
    print self.test_barcodes
    self.test_tissue  = self.data_store[self.TISSUE_key].loc[ self.test_barcodes ]
    self.train_tissue = self.data_store[self.TISSUE_key].loc[ self.train_barcodes ]
    self.val_tissue   = self.data_store[self.TISSUE_key].loc[ self.validation_barcodes ]
  

    #pdb.set_trace()
    self.n_train = len(self.train_barcodes)
    self.n_test  = len(self.test_barcodes)
  
    self.data_dict[N_TRAIN] = self.n_train
    self.data_dict[N_TEST]  = self.n_test
    
    print "** n_train = ", self.n_train
    print "** n_test  = ", self.n_test

    #pdb.set_trace()
    
    if self.include_brca is True:
      # do nothing
      if self.include_validation is True:
        # mode validation to train, validation is empty
        assert False, "this is probably only for when include brca is false"
      pass
    else:
      print "+++ removing BRCA from train and test putting into validation"
      # self.fully_barcodes
      # self.usable_barcodes
      all_brcas = []
      full_brcas = []
      for bc in self.fully_barcodes:
        if bc.split("_")[0] == "brca":
          full_brcas.append( bc )
      for bc in self.usable_barcodes:
        if bc.split("_")[0] == "brca":
          all_brcas.append( bc )
      
      brca_train_barcodes = []
      for bc in self.train_barcodes:
        if bc.split("_")[0]=="brca":
          brca_train_barcodes.append( bc )
      brca_test_barcodes = []
      for bc in self.test_barcodes:
        if bc.split("_")[0]=="brca":
          brca_test_barcodes.append( bc )
          
      np.random.seed(0)
      ids_brcas = np.random.permutation( len(full_brcas) )[:self.nbr_brca_train]
      self.min_bcra_bcs = [full_brcas[idx] for idx in ids_brcas]
      self.all_other_brcas = np.setdiff1d(all_brcas, self.min_bcra_bcs )
      #self.min_bcra_bcs = brca_train_barcodes[:self.nbr_brca_train]
      #brca_train_barcodes = brca_train_barcodes[self.nbr_brca_train:]
      self.brca_barcodes = all_brcas #np.union1d( brca_train_barcodes, brca_test_barcodes )
      
      self.train_barcodes = np.setdiff1d( self.train_barcodes, self.brca_barcodes )
      self.train_barcodes = np.union1d(self.min_bcra_bcs, self.train_barcodes )
      self.test_barcodes = np.setdiff1d( self.test_barcodes, self.brca_barcodes )
      
      if self.include_validation is True:
        # move validation to train
        self.train_barcodes = np.union1d( self.train_barcodes, self.validation_barcodes )
        self.validation_barcodes = self.all_other_brcas
      else:
        # add BRCA to validation
        #self.validation_barcodes = np.union1d( self.brca_barcodes, self.validation_barcodes )
        self.validation_barcodes = self.all_other_brcas

    self.test_tissue  = self.data_store[self.TISSUE_key].loc[ self.test_barcodes ]
    self.train_tissue = self.data_store[self.TISSUE_key].loc[ self.train_barcodes ]
    self.val_tissue   = self.data_store[self.TISSUE_key].loc[ self.validation_barcodes ]

    self.n_train = len(self.train_barcodes)
    self.n_test  = len(self.test_barcodes)
  
    self.data_dict[N_TRAIN] = self.n_train
    self.data_dict[N_TEST]  = self.n_test
    
    print "** n_train = ", self.n_train
    print "** n_test  = ", self.n_test
    
    print "TEST: " 
    print self.test_tissue.sum()
    print "TRAIN: " 
    print self.train_tissue.sum()
    print "VAL: " 
    print self.val_tissue.sum()
    #pdb.set_trace()
    
    self.viz_filename_z_rec_scatter          =  os.path.join( self.savedir, "z_rec_scatter.png" )
    self.viz_filename_z_rec_on_z_gen         =  os.path.join( self.savedir, "z_rec_on_z_gen.png" )
    self.viz_filename_rna_prediction_scatter =  os.path.join( self.savedir, "rna_prediction_scatter.png" )
    self.viz_filename_dna_batch_target       =  os.path.join( self.savedir, "dna_batch_target" )
    self.viz_filename_dna_batch_predict      =  os.path.join( self.savedir, "dna_batch_predict" )
    self.viz_filename_dna_aucs               =  os.path.join( self.savedir, "dna_aucs.png" )
    self.viz_filename_weights_rec_rna        =  os.path.join( self.savedir, "weights_rec_rna.png" )
    self.viz_filename_lower_bound            =  os.path.join( self.savedir, "lower_bound.png" )
    self.viz_filename_log_pdf_sources        = os.path.join( self.savedir, "log_pdf_sources_z.png" )
    self.viz_filename_log_pdf_sources_per_gene = os.path.join( self.savedir, "log_pdf_batch.png" )
    self.viz_filename_log_pdf_sources_per_gene_fill = os.path.join( self.savedir, "log_pdf_fill.png" )
    self.viz_filename_error_sources_per_gene_fill = os.path.join( self.savedir, "errors_fill.png" )
    self.viz_filename_log_pdf_sources_per_gene_fill_all = os.path.join( self.savedir, "log_pdf_sources_z_per_gene_fill_all.png" )
    self.viz_filename_error_sources_per_gene_fill_all = os.path.join( self.savedir, "errors_sources_z_per_gene_fill_all.png" )
  
  
  def InitializeSVD_OLD(self):
    self.savedir    = self.logging_dict[SAVEDIR]
    
    self.keep_rates = OrderedDict()
    if self.arch_dict.has_key( KEEP_RATES ):
      for kr in self.arch_dict[KEEP_RATES]:
        self.keep_rates[ kr[LAYER] ] = kr[KEEP_RATE]
    
    self.n_dna_channels = self.data_dict["n_dna_channels"]    
    self.dna_dim    = self.data_dict[DATASET].GetDimension(DNA)
    self.meth_dim   = self.data_dict[DATASET].GetDimension(METH)
    self.rna_dim    = self.data_dict[DATASET].GetDimension(RNA)
    self.tissue_dim = self.data_dict[DATASET].GetDimension(TISSUE)
    
    self.dims_dict = {RNA:self.rna_dim, DNA:self.dna_dim*self.n_dna_channels, TISSUE:self.tissue_dim, METH:self.meth_dim}
    self.data_store     = self.data_dict[DATASET].store
    self.batch_size     = self.algo_dict[BATCH_SIZE]
    #self.n_test         = self.algo_dict[N_TEST]
    self.n_full_train = self.algo_dict[N_TRAIN_FULL]
    self.n_non_full_train  = self.algo_dict[N_TRAIN_NON_FULL]
    self.train_mode = self.algo_dict[TRAIN_MODE]
    
    # keep BRCA in train and test sets, if False, move them all to validation
    self.include_brca = bool(self.algo_dict["include_brca"])
    self.nbr_brca_train = int(self.algo_dict["nbr_brca_train"])
    # keep non-full tissues in validation, otherwise move to train
    self.include_validation = bool(self.algo_dict["include_validation"])
    
    self.batch_imputation_dict = {}
    self.batch_feed_dict       = {}
    
    # store keys
    self.OBSERVED_key = CLINICAL+"/"+OBSERVED
    self.TISSUE_key   = CLINICAL+"/"+TISSUE
    self.RNA_key      = RNA+"/"+FAIR
    self.METH_key     = METH+"/"+FAIR
    self.DNA_keys     = [DNA+"/"+CHANNEL+"/%d"%i for i in range(self.n_dna_channels)]
    
    self.n_z            = self.var_dict[N_Z]
    self.z_columns = ["z%d"%z for z in range(self.n_z)]
    self.rna_genes = self.data_store[self.RNA_key].columns
    self.dna_genes = self.data_store[self.DNA_keys[0]].columns
    self.meth_genes = self.data_store[self.METH_key].columns
    self.tissue_names = self.data_store[self.TISSUE_key].columns
    self.tissue2color = OrderedDict()
    self.tissue2shape = OrderedDict()
    cmap = pp.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(self.tissue_names)))
    for idx,t in zip( range(len(colors)),self.tissue_names):
      self.tissue2color[t] = colors[idx]
      self.tissue2shape[t] = 'o'
      
    self.latent_store_name = self.network_name + "_" + LATENT
    self.latent_store = OpenHdfStore(self.savedir, self.latent_store_name, mode=self.default_store_mode )
    self.model_store_name = self.network_name + "_" + MODEL
    
    # open in "a" mode
    self.model_store = OpenHdfStore(self.savedir, self.model_store_name, mode="a" )
    
    self.epoch_store_name = self.network_name + "_" + EPOCH
    self.epoch_store = OpenHdfStore(self.savedir, self.epoch_store_name, mode=self.default_store_mode )
    
    self.fill_store_name = self.network_name + "_" + FILL
    self.fill_store = OpenHdfStore(self.savedir, self.fill_store_name, mode="a")
    
    self.fill_store.close()
    self.latent_store.close()
    self.model_store.close()
    self.epoch_store.close()
    
    self.source2darkcolor = {RNA:"darkblue", DNA:"darkgreen", METH:"darkred"}
    self.source2lightcolor = {RNA:"lightblue", DNA:"palegreen", METH:"lightsalmon"}
    self.source2mediumcolor = {RNA:"dodgerblue", DNA:"darksage", METH:"red"}
    
    self.source2mediumcolor[RNA+"+"+DNA]="turquoise"
    self.source2mediumcolor[RNA+"+"+METH] = "fuchsia"
    self.source2mediumcolor[DNA+"+"+METH] = "sandybrown"
    self.source2mediumcolor[RNA+"+"+DNA+"+"+METH]="grey"
    
    self.observed_order = self.data_store[self.OBSERVED_key].columns
    self.observed_source2idx = OrderedDict()
    for source, idx in zip(self.observed_order,range(len(self.observed_order)) ):
      self.observed_source2idx[source] = idx
    
    self.input_sources = self.arch_dict[INPUT_SOURCES]
    self.target_sources = self.arch_dict[TARGET_SOURCES]
    
    # pp.plot( epochs, loglik, self.input_combo2shape[inputs]+'-', \
    #          color=self.source2mediumcolor[target_source],\
    #          mec=self.source2darkcolor[target_source], mew=1, \
    #          mfc=self.input_combo2fillcolor[target_source], lw=2, \
    #          ms = self.input_combo2markersize[inputs], \
    #          label="%s  (%0.3f)"%(inputs,loglik[-1]) )
    #
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
          #self.input_combo2linecolor[inputs] = self.source2lightcolor[inputs]
          
        if nbr == 2:
          self.input_combo2fillcolor[inputs] = self.source2mediumcolor[inputs]
          self.input_combo2shape[inputs] = 's'
          self.input_combo2markersize[inputs] = 8
          
        if nbr == 3:
          self.input_combo2fillcolor[inputs] = self.source2mediumcolor[inputs]
          self.input_combo2shape[inputs] = 'o'
          self.input_combo2markersize[inputs] = 10
          

        
    self.observation_source_indices = []
    for source in self.arch_dict["sources"]:
      self.observation_source_indices.append( self.observed_source2idx[source] )

    self.observation_source_indices_input = []
    for source in self.arch_dict[INPUT_SOURCES]:
      self.observation_source_indices_input.append( self.observed_source2idx[source] )

    self.observation_source_indices_target = []
    for source in self.arch_dict[TARGET_SOURCES]:
      self.observation_source_indices_target.append( self.observed_source2idx[source] )                
    
    print "** getting validations"
    self.at_least_one_query = self.data_store[self.OBSERVED_key].values[:,self.observation_source_indices].sum(1)>0
    
    self.validation_tissue2barcodes = OrderedDict()
    
    self.obs_store_bc_2idx = OrderedDict()
    for idx,bc in zip( range(len(self.data_store[self.OBSERVED_key].index)),self.data_store[self.OBSERVED_key].index):
      self.obs_store_bc_2idx[bc] = idx
      
    self.observed_tissue_and_bcs = self.data_store[self.OBSERVED_key].index
    self.observation_tissues = np.array([s.split("_")[0] for s in self.observed_tissue_and_bcs])
    self.validation_obs_query = np.zeros( len(self.data_store[self.OBSERVED_key]), dtype=bool)
    
    coad_bc = "coad_tcga-t9-a92h"
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
        
      #query = self.data_store[self.OBSERVED_key].loc[self.validation_tissue2barcodes[ tissue ]].values[:,self.observation_source_indices].sum(1)>0
      
      ids = self.observation_tissues==tissue
      #self.validation_obs_query[i.values] = True
      for bc in self.validation_tissue2barcodes[ tissue ]:
        self.validation_obs_query[ self.obs_store_bc_2idx[bc] ] = True
      #self.validation_obs_query[ids] = True
      print tissue, self.validation_obs_query.sum()
      #if tissue=="coad":
      #  pdb.set_trace()
      
    coad_bc = "coad_tcga-t9-a92h"
    
    #pdb.set_trace()
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
    
    self.observed_batch_order = OrderedDict()
    self.observed_batch_order[RNA] = 0
    self.observed_batch_order[DNA] = 1
    self.observed_batch_order[METH] = 2
    # if self.train_mode == ALL:
    #   print "************************"
    #   print "************************"
    #   print "ALL ALL ALL ALL ALL ALL "
    #   print "************************"
    #   print "************************"
    #   self.usable_observed_query = self.at_least_one_query*self.not_validation_query
    #
    #   #*(self.data_store[self.OBSERVED_key].values[:,self.observation_source_indices].sum(1)>0)
    # elif self.train_mode == FULL:
    #   print "************************"
    #   print "************************"
    #   print "FULL FULL FULL FULL FULL"
    #   print "************************"
    #   print "************************"
    #   self.usable_observed_query = self.at_least_one_query*self.not_validation_query*(self.data_store[self.OBSERVED_key].values.sum(1)==len(self.observed_order))
    # else:
    #   assert False, "must be ALL or FULL"
    
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
  
    if self.n_fully_observed <= self.n_full_train:
      print "==> setting test set from %d to %d"%(self.n_full_train,self.n_fully_observed)
      self.n_full_train = self.n_fully_observed
    
    # index into usable for test and train
    
    self.train_full_id_query = np.zeros( len(self.data_store[self.OBSERVED_key]), dtype=bool)
  
    np.random.seed( self.algo_dict['split_seed'] )
    self.train_full_ids = np.random.permutation( self.n_fully_observed )[:self.n_full_train]
    self.train_full_id_query[ self.full_observed_ids[self.train_full_ids] ] = 1
    
    self.test_full_id_query = (self.not_validation_query*self.fully_observed_query*(1-self.train_full_id_query).astype(np.bool) ).astype(np.bool)
    
    # all the usable, but not fully observed data
    self.non_full_observed_query = self.not_validation_query*self.at_least_one_query*(1-self.fully_observed_query).astype(np.bool)
    
    self.non_full_observed_ids = pp.find( self.non_full_observed_query )
    self.n_non_fully_observed = self.non_full_observed_query.sum()
    
    if self.n_non_fully_observed <= self.n_non_full_train:
      print "==> setting test set from %d to %d"%(self.n_non_full_train,self.n_non_fully_observed)
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
    
    if self.train_mode == ALL:
      print "************************"
      print "************************"
      print "ALL ALL ALL ALL ALL ALL "
      print "************************"
      print "************************"
      self.train_barcodes = np.union1d( self.train_full_barcodes, self.train_non_full_barcodes )
    else:
      print "************************"
      print "************************"
      print "FULL FULL FULL FULL FULL"
      print "************************"
      print "************************"
      self.train_barcodes = self.train_full_barcodes
    #
    # #pdb.set_trace()
    # # self.test_barcodes = self.usable_barcodes[self.test_id_query]
    #
    # # correct view of observed for train and test
    # self.test_observed  = self.data_store[self.OBSERVED_key][ self.test_id_query ]
    # self.train_observed = self.data_store[self.OBSERVED_key][ self.train_id_query ]
    #
    # # use these barcodes to locate sources
    # self.test_barcodes  = self.test_observed.index
    # self.train_barcodes = self.train_observed.index
  
    #pdb.set_trace()
    assert len(np.intersect1d( self.test_barcodes, self.train_barcodes)) == 0, "train and test are not mutually exclusive!!"
    assert len(np.intersect1d( self.test_barcodes, self.validation_barcodes)) == 0, "train and test are not mutually exclusive!!"
    assert len(np.intersect1d( self.train_barcodes, self.validation_barcodes)) == 0, "train and test are not mutually exclusive!!"
    
    print "TEST BARCODES: "
    print self.test_barcodes
    self.test_tissue  = self.data_store[self.TISSUE_key].loc[ self.test_barcodes ]
    self.train_tissue = self.data_store[self.TISSUE_key].loc[ self.train_barcodes ]
    self.val_tissue   = self.data_store[self.TISSUE_key].loc[ self.validation_barcodes ]
  

    #pdb.set_trace()
    self.n_train = len(self.train_barcodes)
    self.n_test  = len(self.test_barcodes)
  
    self.data_dict[N_TRAIN] = self.n_train
    self.data_dict[N_TEST]  = self.n_test
    
    print "** n_train = ", self.n_train
    print "** n_test  = ", self.n_test

    #pdb.set_trace()
    
    if self.include_brca is True:
      # do nothing
      if self.include_validation is True:
        # mode validation to train, validation is empty
        assert False, "this is probably only for when include brca is false"
      pass
    else:
      print "+++ removing BRCA from train and test putting into validation"
      # self.fully_barcodes
      # self.usable_barcodes
      all_brcas = []
      full_brcas = []
      for bc in self.fully_barcodes:
        if bc.split("_")[0] == "brca":
          full_brcas.append( bc )
      for bc in self.usable_barcodes:
        if bc.split("_")[0] == "brca":
          all_brcas.append( bc )
      
      brca_train_barcodes = []
      for bc in self.train_barcodes:
        if bc.split("_")[0]=="brca":
          brca_train_barcodes.append( bc )
      brca_test_barcodes = []
      for bc in self.test_barcodes:
        if bc.split("_")[0]=="brca":
          brca_test_barcodes.append( bc )
          
      np.random.seed(0)
      ids_brcas = np.random.permutation( len(full_brcas) )[:self.nbr_brca_train]
      self.min_bcra_bcs = [full_brcas[idx] for idx in ids_brcas]
      self.all_other_brcas = np.setdiff1d(all_brcas, self.min_bcra_bcs )
      #self.min_bcra_bcs = brca_train_barcodes[:self.nbr_brca_train]
      #brca_train_barcodes = brca_train_barcodes[self.nbr_brca_train:]
      self.brca_barcodes = all_brcas #np.union1d( brca_train_barcodes, brca_test_barcodes )
      
      self.train_barcodes = np.setdiff1d( self.train_barcodes, self.brca_barcodes )
      self.train_barcodes = np.union1d(self.min_bcra_bcs, self.train_barcodes )
      self.test_barcodes = np.setdiff1d( self.test_barcodes, self.brca_barcodes )
      
      if self.include_validation is True:
        # move validation to train
        self.train_barcodes = np.union1d( self.train_barcodes, self.validation_barcodes )
        self.validation_barcodes = self.all_other_brcas
      else:
        # add BRCA to validation
        #self.validation_barcodes = np.union1d( self.brca_barcodes, self.validation_barcodes )
        self.validation_barcodes = self.all_other_brcas

    self.test_tissue  = self.data_store[self.TISSUE_key].loc[ self.test_barcodes ]
    self.train_tissue = self.data_store[self.TISSUE_key].loc[ self.train_barcodes ]
    self.val_tissue   = self.data_store[self.TISSUE_key].loc[ self.validation_barcodes ]

    self.n_train = len(self.train_barcodes)
    self.n_test  = len(self.test_barcodes)
  
    self.data_dict[N_TRAIN] = self.n_train
    self.data_dict[N_TEST]  = self.n_test
    
    print "** n_train = ", self.n_train
    print "** n_test  = ", self.n_test
    
    print "TEST: " 
    print self.test_tissue.sum()
    print "TRAIN: " 
    print self.train_tissue.sum()
    print "VAL: " 
    print self.val_tissue.sum()
    #pdb.set_trace()
    
    self.viz_filename_z_rec_on_z_gen         =  os.path.join( self.savedir, "z_rec_on_z_gen.png" )
    self.viz_filename_rna_prediction_scatter =  os.path.join( self.savedir, "rna_prediction_scatter.png" )
    self.viz_filename_dna_batch_target       =  os.path.join( self.savedir, "dna_batch_target" )
    self.viz_filename_dna_batch_predict      =  os.path.join( self.savedir, "dna_batch_predict" )
    self.viz_filename_dna_aucs               =  os.path.join( self.savedir, "dna_aucs.png" )
    self.viz_filename_weights_rec_rna        =  os.path.join( self.savedir, "weights_rec_rna.png" )
    self.viz_filename_lower_bound            =  os.path.join( self.savedir, "lower_bound.png" )
    self.viz_filename_log_pdf_sources        = os.path.join( self.savedir, "log_pdf_sources_z.png" )
    self.viz_filename_log_pdf_sources_per_gene = os.path.join( self.savedir, "log_pdf_batch.png" )
    self.viz_filename_log_pdf_sources_per_gene_fill = os.path.join( self.savedir, "log_pdf_fill.png" )
    self.viz_filename_error_sources_per_gene_fill = os.path.join( self.savedir, "errors_fill.png" )
    self.viz_filename_log_pdf_sources_per_gene_fill_all = os.path.join( self.savedir, "log_pdf_sources_z_per_gene_fill_all.png" )
    self.viz_filename_error_sources_per_gene_fill_all = os.path.join( self.savedir, "errors_sources_z_per_gene_fill_all.png" )


  def Train( self, batch ):
    print "$$$$$$$ TRAINING SVD $$$$$$$$$"
    X = self.MakeX( batch )
    
    #Xtrain = self.MakeTrain()
    n_z = self.n_z
    
    Utrain,s,VT = sp.linalg.svd( X )
    s = s[:n_z]
    VT = VT[:n_z,:]
    Utrain = Utrain[:,:n_z]; S = np.diag( s )

    self.Ztrain = Utrain
    self.Z_eigvals = S
    
    FT = np.dot(S,VT)
  
    self.S  = S
    self.s = s
    self.VT = VT
    self.FT = np.dot(S,VT)
    
  def Test( self, batch ):
    print "$$$$$$$ TESTING SVD $$$$$$$$$"
    X = self.MakeX( batch )
    bcs = batch[BARCODES]
    Utest = np.dot( np.diag( (1.0/self.s)**2), np.dot( self.FT, X.T ) ).T
    X_pred = np.dot( Utest, self.FT) 
    X_pred   = np.maximum(X_pred,0)
    X_pred   = np.minimum(X_pred,1)
    
    #pdb.set_trace()
    rna_fill, \
    dna_fill, \
    meth_fill = self.split_result( X_pred  )
    
    return rna_fill, dna_fill, meth_fill, Utest

  def split_result( self, X ):
    cum_dims = self.cum_dims
    dims = self.dims
    
    rna_predict   = X[:,:dims[0]]

    dna_predict = X[:,cum_dims[0]:cum_dims[1] ]
    meth_predict  = X[:,cum_dims[1]:]
    
    return rna_predict, dna_predict, meth_predict
        
  def CallBack( self, function_name, cb_info ):
    if function_name == BATCH_EPOCH:
      self.BatchEpoch( cb_info )

    elif function_name == TEST_EPOCH:
      self.TestEpoch( cb_info )
      self.ValEpoch( cb_info )

    #elif function_name == EPOCH_VIZ:
    #  self.VizEpochs(  cb_info )

    elif function_name == SAVE_MODEL:
      self.SaveModel( sess, cb_info )

    elif function_name == MODEL_VIZ:
      self.VizModel( sess, cb_info )

    elif function_name == SAVE_LATENT:
      self.SaveTestLatent( sess, cb_info )

    elif function_name == LATENT_VIZ:
      self.VizLatent( sess, cb_info )

    elif function_name == TEST_FILL:
      self.TestFill2( sess, cb_info )
  
  def TestFill2( self, info_dict ):

    print "$$$$$$$ TESTING SVD $$$$$$$$$"
    epoch       = 0 #info_dict[EPOCH]
    #feed_dict   = info_dict[TEST_FEED_DICT]
    impute_dict = info_dict[TEST_FEED_IMPUTATION]
    
    self.RunFill2( epoch, impute_dict, mode="TEST" )
    self.RunFillZ( epoch, impute_dict, mode="TEST" )

    print "$$$$$$$ VALIDATING SVD $$$$$$$$$"
    #feed_dict   = info_dict[VAL_FEED_DICT]
    impute_dict = info_dict[VAL_FEED_IMPUTATION]
    
    self.RunFill2( epoch, impute_dict, mode="VAL" )
    self.RunFillZ( epoch, impute_dict, mode="VAL" )
  
  def MakeX( self, d ):
    #dims = np.array([rna_test.shape[1],dna_0_test.shape[1],dna_1_test.shape[1],dna_2_test.shape[1],dna_3_test.shape[1],meth_test.shape[1]])
    #cum_dims = np.cumsum(dims)
    #Xtrain = np.hstack((rna_train,dna_0_train,dna_1_train,dna_2_train,dna_3_train,meth_train))
    
    
    X = np.hstack(( d[RNA_INPUT], \
                    d[DNA_INPUT], \
                    d[METH_INPUT] ) )
    return X
      
  def RunFill2( self, epoch, impute_dict, mode ):
    print "COMPUTE Z-SPACE"
          
    barcodes = impute_dict[BARCODES]
    batch = self.FillBatch( impute_dict[BARCODES], mode )
    #not_observed = np.setdiff1d( self.input_sources, inputs2use )
        
    #rec_z_space_tensors       = self.network.GetTensor( "rec_z_space" )
    #rna_rec_z_space_tensors   = self.network.GetTensor( "rec_z_space_rna" )
    #dna_rec_z_space_tensors   = self.network.GetTensor( "rec_z_space_dna" )
    #meth_rec_z_space_tensors  = self.network.GetTensor( "rec_z_space_meth" )
  
    #rna_expectation_tensor = self.network.GetLayer( "gen_rna_space" ).expectation
    #dna_expectation_tensor = self.network.GetLayer( "gen_dna_space" ).expectation
    #meth_expectation_tensor = self.network.GetLayer( "gen_meth_space" ).expectation
    
    dna_data = np.zeros( (len(barcodes),self.dna_dim) )
    for idx,DNA_key in zip(range(len(self.DNA_keys)),self.DNA_keys):
      batch_data = self.data_store[DNA_key].loc[ barcodes ].fillna( 0 ).values
      dna_data += batch_data
      
    dna_data = np.minimum(1.0,dna_data)
      # matrix tensors for each target source
    #loglikes_data_as_matrix = self.network.loglikes_data_as_matrix
  
    # tensors = []
    # tensors.extend(rec_z_space_tensors)
    # tensors.extend(rna_rec_z_space_tensors)
    # #tensors.extend(dna_rec_z_space_tensors)
    # tensors.extend(meth_rec_z_space_tensors)
    # tensors.extend([rna_expectation_tensor,dna_expectation_tensor,meth_expectation_tensor])
    #
    # tensor_names = ["z_mu","z_var",\
    #                 "z_mu_rna","z_var_rna",\
    #                 #"z_mu_dna","z_var_dna",\
    #                 "z_mu_meth","z_var_meth",\
    #                 "rna_expecation","dna_expectation","meth_expectation"]
  
    #assert len(tensor_names)==len(tensors), "should be same number"
    #self.network.FillFeedDict( feed_dict, impute_dict )

    rna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[RNA]] == 1
    dna_observed_query = batch[ DNA_OBSERVATIONS ] == 1
    meth_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[METH]] == 1
        
    rna_expectation = np.zeros( (len(barcodes), self.dims_dict[RNA] ), dtype=float )
    rna_loglikelihood  = np.zeros( (np.sum(rna_observed_query), self.dims_dict[RNA] ), dtype=float )
    meth_expectation = np.zeros( (len(barcodes), self.dims_dict[METH] ), dtype=float )
    meth_loglikelihood  = np.zeros( (np.sum(meth_observed_query), self.dims_dict[METH] ), dtype=float )
        
      #drop_likelihoods = np.zeros( rna_dim )
    dna_dim = self.dims_dict[DNA] #/self.n_dna_channels
    
    dna_expectation = np.zeros( (len(barcodes),dna_dim), dtype=float )
    dna_loglikelihood = np.zeros( (np.sum(dna_observed_query),dna_dim), dtype=float )
        
    #like_observed = batch[ INPUT_OBSERVATIONS ] 
    

    nbr_splits = 10
    #tensor2fill = []
    drop_factor = float(nbr_splits)/float(nbr_splits-1)
    for drop_idx in range(nbr_splits):
      drop_rna_ids = np.arange(drop_idx,self.dims_dict[RNA],nbr_splits, dtype=int)
      drop_dna_ids = np.arange(drop_idx,dna_dim,nbr_splits, dtype=int)
      drop_meth_ids = np.arange(drop_idx,self.dims_dict[METH],nbr_splits, dtype=int)
      
      
      
      # ------
      # RNA
      # -----
      batch_data = self.data_store[self.RNA_key].loc[ barcodes ]
      nans = np.isnan( batch_data.values )
      batch[ RNA_INPUT ] = drop_factor*self.NormalizeRnaInput( batch_data.fillna( 0 ).values )
      batch[ RNA_INPUT ][nans] = 0
      
      batch[ RNA_INPUT][:,drop_rna_ids] = 0
      
      #tensor2fill.extend( [rna_expectation_tensor, loglikes_data_as_matrix["gen_rna_space"] ] )
        
      
      # ------
      # DNA
      # -----
      dna_data_inputs = np.minimum(1.0,dna_data)
      #for idx,DNA_key in zip(range(len(self.DNA_keys)-1),self.DNA_keys[:-1]):
      dna_data_inputs[:,drop_dna_ids] = 0
      batch[ DNA_INPUT ] = drop_factor*dna_data_inputs
      #tensor2fill.extend( [dna_expectation_tensor, loglikes_data_as_matrix["gen_dna_space"] ] )
      # columns = self.dna_genes
      # observations = dna_data
      # batch[ INPUT_OBSERVATIONS ]
        
      
      # ------
      # DNA
      # -----
      batch_data = self.data_store[self.METH_key].loc[ barcodes ]
      batch[ METH_INPUT ] = drop_factor*batch_data.fillna( 0 ).values
      batch[ METH_INPUT][:,drop_meth_ids] = 0
        
      
      # ---------
      # RUN SESS
      # ---------
      rna_fill, dna_fill, meth_fill, z_fill = self.Test( batch )
      
      # ------
      # FILL EVALUATION 
      # -----      
      rna_expectation[:,drop_rna_ids]   = rna_fill[:,drop_rna_ids]
      #rna_loglikelihood[:,drop_rna_ids] = tensor2fill_eval[1][:,drop_rna_ids]
      
      #for idx,DNA_key in zip(range(len(self.DNA_keys)-1),self.DNA_keys[:-1]):
      dna_expectation[:,drop_dna_ids] = dna_fill[:,drop_dna_ids]
      #dna_loglikelihood[:,drop_dna_ids] = tensor2fill_eval[3][:,drop_dna_ids]
      
      meth_expectation[:,drop_meth_ids]   = meth_fill[:,drop_meth_ids]
      #meth_loglikelihood[:,drop_meth_ids] = tensor2fill_eval[5][:,drop_meth_ids]
    
    #pdb.set_trace()      
    self.WriteRunFillExpectation( epoch, RNA, barcodes, self.rna_genes, rna_observed_query, rna_expectation, self.data_store[self.RNA_key].loc[ barcodes ].values, mode )
    self.WriteRunFillExpectation( epoch, METH, barcodes, self.meth_genes, meth_observed_query, meth_expectation, self.data_store[self.METH_key].loc[ barcodes ].values, mode )
    self.WriteRunFillExpectation( epoch, DNA, barcodes, self.dna_genes, dna_observed_query, dna_expectation, dna_data, mode )

    
    #self.WriteRunFillLoglikelihood( epoch, RNA, barcodes[rna_observed_query], self.rna_genes, rna_loglikelihood, mode )
    #self.WriteRunFillLoglikelihood( epoch, METH, barcodes[meth_observed_query], self.meth_genes, meth_loglikelihood, mode )
    #self.WriteRunFillLoglikelihood( epoch, DNA, barcodes[dna_observed_query], self.dna_genes, dna_loglikelihood, mode )

    #self.WriteRunFillLoglikelihood( epoch, target, inputs2use, barcodes, columns, target_loglikelihood, is_test )
    #pdb.set_trace()

  def RunFillZ( self, epoch, impute_dict, mode ):
    print "FILL Z"
          
    barcodes = impute_dict[BARCODES]
    batch = self.FillBatch( impute_dict[BARCODES], mode )
  
    rna_fill, dna_fill, meth_fill, z_fill = self.Test( batch )

    self.WriteZ( epoch, z_fill, barcodes, self.z_columns, mode)
    
  def RunFillZ_old( self, epoch, impute_dict, mode ):
    print "RunFill Z"
    batch = impute_dict
    barcodes = impute_dict[BARCODES]
    #batch = self.FillBatch( impute_dict[BARCODES], mode )
    
    dna_data = np.zeros( (len(barcodes), self.n_dna_channels,self.dna_dim) )
    for idx,DNA_key in zip(range(len(self.DNA_keys)),self.DNA_keys):
      batch_data = self.data_store[DNA_key].loc[ barcodes ].fillna( 0 ).values
      dna_data[:,idx,:] = batch_data

    rna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[RNA]] == 1
    dna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[DNA]] == 1
    meth_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[METH]] == 1
        
    rna_expectation = np.zeros( (len(barcodes), self.dims_dict[RNA] ), dtype=float )
    rna_loglikelihood  = np.zeros( (np.sum(rna_observed_query), self.dims_dict[RNA] ), dtype=float )
    meth_expectation = np.zeros( (len(barcodes), self.dims_dict[METH] ), dtype=float )
    meth_loglikelihood  = np.zeros( (np.sum(meth_observed_query), self.dims_dict[METH] ), dtype=float )
        
    dna_dim = self.dims_dict[DNA]/self.n_dna_channels
    
    dna_expectation = np.zeros( (len(barcodes), self.n_dna_channels,dna_dim), dtype=float )
    dna_loglikelihood = np.zeros( (np.sum(dna_observed_query), self.n_dna_channels,dna_dim), dtype=float )

    # ------
    # RNA
    # -----
    batch_data = self.data_store[self.RNA_key].loc[ barcodes ]
    nans = np.isnan( batch_data.value )
    batch[ RNA_INPUT ] = self.NormalizeRnaInput( batch_data.fillna( 0 ).values )
    batch[ RNA_INPUT ][nans] = 0

    # ------
    # DNA
    # -----
    dna_data_inputs = dna_data.copy()
    batch[ DNA_INPUT ] = dna_data_inputs
    
    # ------
    # METH
    # -----
    batch_data = self.data_store[self.METH_key].loc[ barcodes ]
    batch[ METH_INPUT ] = batch_data.fillna( 0 ).values
    
    # ---------
    # RUN SESS
    # ---------
    rna_fill, dna_fill, meth_fill, z_fill = self.Test( batch )

    self.WriteZ( epoch, z_fill, barcodes, self.z_columns, mode)

  def WriteZ( self, epoch, X, barcodes, columns, mode ):

    self.fill_store.open()
    
    #s = "/Z/%s/mu/"%(mode )
    
    self.fill_store[ "/Z/%s/Z/mu/"%(mode ) ] = pd.DataFrame( X, index = barcodes, columns = columns )
    self.fill_store.close()
        
  def WriteRunFillExpectation( self, epoch, target, barcodes, columns, obs_query, X, Y, mode ):

    self.fill_store.open()
    if target == DNA:
      for channel in range(self.n_dna_channels):
        s = "/Fill/%s/%s/%d/"%(mode,target, channel )
        self.fill_store[ s ] = pd.DataFrame( X, index = barcodes, columns = columns )
      x = X[obs_query,:].flatten()
      y = Y[obs_query,:].flatten()
      auc = roc_auc_score(y,x)
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
    
  # ---------------------------------- #
  # ------- EPOCH CALLBACKS
  # ---------------------------------- #
    
  def BatchEpoch( self, info_dict ):
    epoch       = info_dict[EPOCH]
    impute_dict = info_dict[BATCH_FEED_IMPUTATION]
    
    self.Epoch( "Batch", sess, info_dict, epoch, feed_dict, impute_dict, mode = "BATCH" )

  def TestEpoch( self, sess, info_dict ):
    epoch       = info_dict[EPOCH]
    impute_dict = info_dict[TEST_FEED_IMPUTATION]
    
    self.Epoch( "Test", sess, info_dict, epoch, feed_dict, impute_dict, mode = "TEST" )

  def TrainEpoch( self, sess, info_dict ):    
    epoch       = info_dict[EPOCH]
    impute_dict = info_dict[VAL_FEED_IMPUTATION]
    
    self.Epoch( "Train", sess, info_dict, epoch, feed_dict, impute_dict, mode = "TRAIN" )


  def ValEpoch( self, sess, info_dict ):    
    epoch       = info_dict[EPOCH]
    impute_dict = info_dict[VAL_FEED_IMPUTATION]
    
    self.Epoch( "Val", sess, info_dict, epoch, feed_dict, impute_dict, mode = "VAL" )
            
  def VizEpochs(self, sess, info_dict ):
    print "** VIZ Epochs"
    self.epoch_store.open()
  
    f = pp.figure()
  
    pp.plot( self.epoch_store["Batch"]["Epoch"].values, self.epoch_store["Batch"]["Lower Bound"], 'bo-', lw=2 )
    pp.plot( self.epoch_store["Test"]["Epoch"].values, self.epoch_store["Test"]["Lower Bound"], 'ro-', lw=2 )
    pp.legend( ["Batch","Test"], loc="lower right")
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
      pp.plot( self.epoch_store[TEST_SOURCE_LOGPDF]["Epoch"].values, \
               self.epoch_store[TEST_SOURCE_LOGPDF][target_source]/self.dims_dict[target_source], 'o-', \
               color=self.source2mediumcolor[target_source],\
               mec=self.source2darkcolor[target_source], mew=2, \
               mfc=self.source2lightcolor[target_source], lw=3, \
               ms = 8, \
               label="Test  (%0.4f)"%(self.epoch_store[TEST_SOURCE_LOGPDF][target_source].values[-1]/self.dims_dict[target_source])  )

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
      query1 = self.epoch_store[TEST_FILL_LOGLIK]["Target"] == target_source
      query = query1#&query2
      loglik_df = self.epoch_store[TEST_FILL_LOGLIK][query]
      epochs = loglik_df["Epoch"].values
      loglik = loglik_df["LogLik"].values
      pp.plot( epochs, loglik, 'o-', \
               color=self.source2darkcolor[target_source],\
               mec=self.source2darkcolor[target_source], mew=1, \
               mfc=self.source2lightcolor[target_source], lw=2, \
               ms = 8, \
               label="Test (%0.4f)"%(loglik[-1]) )

      query1 = self.epoch_store[VAL_FILL_LOGLIK]["Target"] == target_source
      query = query1#&query2
      loglik_df = self.epoch_store[VAL_FILL_LOGLIK][query]
      epochs = loglik_df["Epoch"].values
      loglik = loglik_df["LogLik"].values
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
      query1 = self.epoch_store[TEST_FILL_ERROR]["Target"] == target_source
      #query2 = self.epoch_store[TEST_FILL_ERROR]["Inputs"] == inputs
      query = query1#&query2
      df = self.epoch_store[TEST_FILL_ERROR][query]
      epochs = df["Epoch"].values
      loglik = df["Error"].values
      pp.plot( epochs, loglik, 'o-', \
               color=self.source2darkcolor[target_source],\
               mec=self.source2darkcolor[target_source], mew=1, \
               mfc=self.source2lightcolor[target_source], lw=2, \
               ms = 8, \
               label="Test (%0.6f)"%(loglik[-1]) )
      query1 = self.epoch_store[VAL_FILL_ERROR]["Target"] == target_source
      #query2 = self.epoch_store[TEST_FILL_ERROR]["Inputs"] == inputs
      query = query1#&query2
      df = self.epoch_store[VAL_FILL_ERROR][query]
      epochs = df["Epoch"].values
      loglik = df["Error"].values
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
  
    try:
      rec_rna_weights = self.model_store[ "/rec_hidden1/W/0" ].values.flatten()
      f = pp.figure()
      pp.hist(  rec_rna_weights, 50, normed=True, alpha=0.5 )      
      pp.grid('on')
      pp.savefig( self.viz_filename_weights_rec_rna, dpi = 300, fmt="png", bbox_inches = "tight")
      pp.close(f)
    except:
      print "** could not viz any model"
    self.model_store.close()
    pp.close('all')
      
  def SaveTestLatent( self, sess, info_dict ):
    print "** SAVE Latent"
    self.latent_store.open() # = OpenHdfStore(self.savedir, self.store_name, mode="a" )
    
    feed_dict          = info_dict[ TEST_FEED_DICT ]
    feed_imputation    = info_dict[ TEST_FEED_IMPUTATION ]
    barcodes           = feed_imputation[BARCODES]
    z_rec_space_tensor = self.network.GetLayer( REC_Z_SPACE ).tensor
    z_gen_space_tensor = self.network.GetLayer( GEN_Z_SPACE ).tensor

  
    z_rec_space        = sess.run( z_rec_space_tensor, feed_dict = feed_dict )
    z_gen_space        = sess.run( z_gen_space_tensor, feed_dict = feed_dict )
    
    
    for idx, z_s in zip( range(len(z_rec_space)),z_rec_space ):
      self.latent_store[ REC_Z_SPACE + "/s%d/"%(idx)] = pd.DataFrame( z_s, index = barcodes, columns=self.z_columns)

    for idx, z_s in zip( range(len(z_gen_space)),z_gen_space ):
      self.latent_store[ GEN_Z_SPACE + "/s%d/"%(idx)] = pd.DataFrame( z_s, index = barcodes, columns=self.z_columns)
            
    # if self.network.HasLayer( "gen_rna_space" ):
    #   rna_gen_space_tensor = self.network.GetLayer( "gen_rna_space" ).tensor
    #   rna_target_tensor = self.network.GetLayer( RNA_TARGET ).tensor
    #
    #   rna_gen_space        = sess.run( rna_gen_space_tensor, feed_dict = info_dict[ TEST_FEED_DICT ] )
    #   rna_target           = sess.run( rna_target_tensor, feed_dict = info_dict[ TEST_FEED_DICT ] )
    #
    #   self.latent_store[ RNA + "/test/predict"] = pd.DataFrame( rna_gen_space[0]/(rna_gen_space[0]+rna_gen_space[1]), index = barcodes, columns=self.rna_genes)
    #   self.latent_store[ RNA + "/test/target"]  = pd.DataFrame( rna_target, index = barcodes, columns=self.rna_genes)
    #
    #   rna_gen_space        = sess.run( rna_gen_space_tensor, feed_dict = info_dict[BATCH_FEED_DICT] )
    #   rna_target           = sess.run( rna_target_tensor, feed_dict = info_dict[BATCH_FEED_DICT] )
    #
    #   self.latent_store[ RNA + "/batch/predict"] = pd.DataFrame( rna_gen_space[0]/(rna_gen_space[0]+rna_gen_space[1]), columns=self.rna_genes)
    #   self.latent_store[ RNA + "/batch/target"]  = pd.DataFrame( rna_target, columns=self.rna_genes)
    
    # if self.network.HasLayer( "gen_dna_space" ):
    #   dna_gen_space_tensor = self.network.GetLayer( "gen_dna_space" ).tensor
    #   dna_target_tensor = self.network.GetLayer( DNA_TARGET ).tensor
    #
    #   dna_gen_space        = sess.run( dna_gen_space_tensor, feed_dict = info_dict[ TEST_FEED_DICT ] )
    #   dna_target           = sess.run( dna_target_tensor, feed_dict = info_dict[ TEST_FEED_DICT ] )
    #
    #
    #   for channel_idx in range(self.n_dna_channels):
    #     self.latent_store[ DNA + "/test/predict/c%d/"%channel_idx] = pd.DataFrame( dna_gen_space[:,channel_idx,:], index = barcodes, columns=self.dna_genes)
    #     self.latent_store[ DNA + "/test/target/c%d/"%channel_idx]  = pd.DataFrame( dna_target[:,channel_idx,:], index = barcodes, columns=self.dna_genes)
    #
    #
    #   dna_gen_space        = sess.run( dna_gen_space_tensor, feed_dict = info_dict[BATCH_FEED_DICT] )
    #   dna_target           = sess.run( dna_target_tensor, feed_dict = info_dict[BATCH_FEED_DICT] )
    #
    #   for channel_idx in range(self.n_dna_channels):
    #
    #     self.latent_store[ DNA + "/batch/predict/c%d/"%channel_idx] = pd.DataFrame( dna_gen_space[:,channel_idx,:], columns=self.dna_genes)
    #     self.latent_store[ DNA + "/batch/target/c%d/"%channel_idx]  = pd.DataFrame( dna_target[:,channel_idx,:], columns=self.dna_genes)
    
    self.latent_store.close()
    
  def VizLatent( self, sess, info_dict ): 
    print "** VIZ Latent"
    self.latent_store.open()
  
    mean_rec_z = self.latent_store[ REC_Z_SPACE + "/s%d/"%(0)]
    mean_gen_z = self.latent_store[ GEN_Z_SPACE + "/s%d/"%(0)]

    f = pp.figure()
    
    mean_gen_z_mean = mean_gen_z.mean().values
    mean_gen_z_std  = mean_gen_z.std().values
    
    pp.plot( mean_gen_z_mean, "wo", lw=1 )
    pp.plot( mean_gen_z_mean + 2*mean_gen_z_std, "k-", lw=0.5 )
    pp.plot( mean_gen_z_mean - 2*mean_gen_z_std, "k-", lw=0.5 )
    pp.fill_between( np.arange(len(mean_gen_z_mean)), mean_gen_z_mean - 2*mean_gen_z_std, mean_gen_z_mean + 2*mean_gen_z_std, color="k", alpha=0.5 )
    
    sns.violinplot( x=None, y=None, data=mean_rec_z)
    pp.grid('on')
    pp.savefig( self.viz_filename_z_rec_on_z_gen, dpi = 300, fmt="png", bbox_inches = "tight")
    ##
    pp.close(f)
    
    pp.close('all')
    
    rna_predict_test  = self.latent_store[ RNA + "/test/predict"].values
    rna_target_test   = self.latent_store[ RNA + "/test/target"].values
    rna_predict_batch = self.latent_store[ RNA + "/batch/predict"].values
    rna_target_batch  = self.latent_store[ RNA + "/batch/target"].values
    
    f = pp.figure()
    pp.plot( rna_target_test.flatten(), rna_predict_test.flatten(), 'r.', alpha = 0.3)
    pp.plot( rna_target_batch.flatten(), rna_predict_batch.flatten(), 'b.', alpha = 0.3)
    pp.xlim(0,1)
    pp.ylim(0,1)
    pp.legend( ["Test", "Batch"])
    pp.xlabel( "TARGET" ); pp.ylabel("PREDICTION")
    pp.title("Test RNA Prediction")
    pp.grid('on')
    pp.savefig( self.viz_filename_rna_prediction_scatter, dpi = 300, fmt="png", bbox_inches = "tight")
    ##
    pp.close(f)
    
    aucs = []
    rocs = []
    
    for channel_idx in range(self.n_dna_channels):
      f = pp.figure()
      pp.matshow(self.latent_store[ DNA + "/batch/target/c%d/"%channel_idx])
      pp.title("Batch DNA Target")
      pp.grid('on')
      pp.savefig( self.viz_filename_dna_batch_target+"%d.png"%(channel_idx), dpi = 300, fmt="png", bbox_inches = "tight")
      pp.close()  
      f = pp.figure()
      pp.matshow(self.latent_store[ DNA + "/batch/predict/c%d"%channel_idx])
      pp.title("Batch DNA Prediction")
      pp.grid('on')
      pp.savefig( self.viz_filename_dna_batch_predict+"%d.png"%(channel_idx), dpi = 300, fmt="png", bbox_inches = "tight")
      pp.close()    
      
      flattened_dna_predict = self.latent_store[ DNA + "/test/predict/c%d"%channel_idx].values.flatten()
      flattened_dna_target = self.latent_store[ DNA + "/test/target/c%d"%channel_idx].values.flatten()
      test_auc = roc_auc_score(flattened_dna_target,flattened_dna_predict)
      aa,cc,dd = roc_curve(flattened_dna_target,flattened_dna_predict)
      aucs.append(test_auc)
      rocs.append([aa,cc,dd])
      
    f_auc = pp.figure()
    s_auc = f_auc.add_subplot(111)
    title="DNA AUC: "
    for auc,roc in zip(aucs,rocs):
      pp.plot( roc[0],roc[1], '-',lw=4, alpha=1)
      title+= "%0.2f "%(auc)
    pp.legend(["channel %d"%idx for idx in range(self.n_dna_channels)], loc="lower right")
    pp.title(title )
    pp.plot( [0,1],[0,1], 'k--')
    pp.grid('on')
    pp.savefig( self.viz_filename_dna_aucs, dpi = 300, fmt="png", bbox_inches = "tight")
    pp.close()
      
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
   
    # HACK to get the sizes correct
    n_batch_size = len(d[BARCODES])
    counts[RNA] = n_rna
    counts[DNA] = n_dna
    counts[METH] = n_meth
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
    elif mode == "TEST":
      self.AddSeries(  self.epoch_store, TEST_SOURCE_LOGPDF, values = epoch_log_p_source_z_values, columns = epoch_source_columns )
      self.PrintRow( self.epoch_store, epoch_key )
    elif mode == "VAL":
      self.AddSeries(  self.epoch_store, VAL_SOURCE_LOGPDF, values = epoch_log_p_source_z_values, columns = epoch_source_columns )
     
    
    if mode == "TEST" or mode == "VAL":
      #print "!!!!!!!!"
      #print "testing product model"
      input_observations = impute_dict[INPUT_OBSERVATIONS]

      tensors = []
      tensors.extend( self.network.GetTensor("rec_z_space_rna") )
      tensors.extend( self.network.GetTensor("rec_z_space_dna") )
      tensors.extend( self.network.GetTensor("rec_z_space_meth") )
      tensors.extend( self.network.GetTensor("rec_z_space") )
      tensors.extend( self.network.GetTensor("gen_z_space") )

      tensor_evals = sess.run( tensors, feed_dict = feed_dict )

      rna_mean  = tensor_evals[0]
      rna_var   = tensor_evals[1]
      dna_mean  = tensor_evals[2]
      dna_var   = tensor_evals[3]
      meth_mean = tensor_evals[4]
      meth_var  = tensor_evals[5]
      z_mean    = tensor_evals[6]
      z_var     = tensor_evals[7]
      z_mean_g  = tensor_evals[8]
      z_var_g   = tensor_evals[9]
      
      self.fill_store.open()
      self.fill_store[ "%s/Z/%s/mu"%(mode,RNA)]   = pd.DataFrame( rna_mean, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/%s/var"%(mode,RNA)]  = pd.DataFrame( rna_var, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/%s/mu"%(mode,DNA)]   = pd.DataFrame( dna_mean, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/%s/var"%(mode,DNA)]  = pd.DataFrame( dna_var, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/%s/mu"%(mode,METH)]  = pd.DataFrame( meth_mean, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/%s/var"%(mode,METH)] = pd.DataFrame( meth_var, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/rec/mu"%mode]        = pd.DataFrame( z_mean, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/rec/var"%mode]       = pd.DataFrame( z_var, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/gen/mu"%mode]        = pd.DataFrame( z_mean_g, index = barcodes, columns = self.z_columns )
      self.fill_store[ "%s/Z/gen/var"%mode]       = pd.DataFrame( z_var_g, index = barcodes, columns = self.z_columns )
      self.fill_store.close()
     
    
  def NextBatch( self, batch_ids ):
    return self.FillBatch( self.train_barcodes[batch_ids], mode = "BATCH" )

  def TrainBatch( self ):
    return self.FillBatch( self.train_barcodes, mode = "TRAIN" )
        
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
    
    self.layer_names = [RNA_INPUT, DNA_INPUT, METH_INPUT, INPUT_OBSERVATIONS]
    for layer_name in self.layer_names: 
    #for layer_name, layer in self.network.layers.iteritems():
      #print layer_name, layer  
      if layer_name == RNA_INPUT:
        batch_data = self.data_store[self.RNA_key].loc[ batch_barcodes ]
        nans = np.isnan( batch_data.values )
        batch[ layer_name ] = self.NormalizeRnaInput( batch_data.fillna( 0 ).values )
        batch[ layer_name ][nans] = 0
        #pdb.set_trace()
        
      elif layer_name == DNA_INPUT:
        dna_data = np.zeros( (len(batch_barcodes), self.dna_dim) )
        #for idx,DNA_key in zip(range(len(self.DNA_keys)-1),self.DNA_keys[:-1]):
        for idx,DNA_key in zip(range(len(self.DNA_keys)),self.DNA_keys):
          batch_data = self.data_store[DNA_key].loc[ batch_barcodes ].fillna( 0 ).values
          dna_data += batch_data
        
        batch[ layer_name ] = np.minimum(1.0,dna_data)# np.array( dna_data )
        
      elif layer_name == METH_INPUT :
        batch_data = self.data_store[self.METH_key].loc[ batch_barcodes ]
        nans = np.isnan( batch_data.values )
        batch[ layer_name ] = self.NormalizeMethInput( batch_data.fillna( 0 ).values )
        batch[ layer_name ][nans] = 0
        
      elif layer_name == TISSUE_INPUT:
        batch_data = self.data_store[self.TISSUE_key].loc[ batch_barcodes ]
        batch[ layer_name ] = batch_data.fillna( 0 ).values
        
        
      elif layer_name == INPUT_OBSERVATIONS:
        batch_data = batch[ "observed_sources" ][:,self.observed_product_sources]
        batch[ layer_name ] = batch_data
        
    return batch            
    
  def NormalizeMethInput( self, X ):
    #return X
    return 2*X-1.0

  def NormalizeMethTarget( self, X ):
    return 0.0001+0.9999*X
    return X

  def NormalizeRnaInput( self, X ):
    #return X
    return 2*X-1.0

  def NormalizeRnaTarget( self, X ):
    return 0.0001+0.9999*X
    return X


  def FillBatch_old( self, batch_barcodes, mode = "BATCH" ):
    batch = OrderedDict()
    
    n = len(batch_barcodes)
    #observed_columns = self.store[self.OBSERVED_key].columns[1:]
    #obs2idx = OrderedDict()
    #for s,idx in zip( self.observed_order, range(len(self.observed_order)) ):
    #  obs2idx[ s ] = idx
      
    batch_observed = self.data_store[self.OBSERVED_key].loc[ batch_barcodes ].values
    batch[ "observed_sources" ] = batch_observed
    batch[ "barcodes" ]         = batch_barcodes
    self.layer_names = [RNA_INPUT, DNA_INPUT, METH_INPUT, INPUT_OBSERVATIONS]
    for layer_name in self.layer_names:
      #print layer_name, layer  
      if layer_name == RNA_INPUT:
        batch_data = self.data_store[self.RNA_key].loc[ batch_barcodes ]
        nans = batch_data.values==np.nan
        batch[ layer_name ] = self.NormalizeRnaInput( batch_data.fillna( 0 ).values )
        batch[ layer_name ][nans] = 0
        #pdb.set_trace()
        
      elif layer_name == RNA_TARGET:
        batch_data = self.data_store[self.RNA_key].loc[ batch_barcodes ]
        batch[ layer_name ] = self.NormalizeRnaTarget( batch_data.fillna( 0 ).values )
        
      elif layer_name == RNA_TARGET_MASK:
        #batch_data = self.store[self.RNA_key].loc[ batch_barcodes ]
        batch[ layer_name ] = batch_observed[:,self.observed_source2idx[ RNA ]].astype(bool)
        
        nbr_observed = batch_observed[:,self.observed_source2idx[ RNA ]].astype(bool).sum()
        #if nbr_observed < len(batch_barcodes):
        #  print "** batch has %d of %d observed RNA"%(nbr_observed, len(batch_barcodes))
        #print "-- RNA observed = %d"%(nbr_observed)
        #pdb.set_trace()
      elif layer_name == DNA_TARGET_MASK:
        #batch_data = self.store[self.RNA_key].loc[ batch_barcodes ]
        batch[ layer_name ] = batch_observed[:,self.observed_source2idx[ DNA ]].astype(bool)
        
        nbr_observed = batch_observed[:,self.observed_source2idx[ DNA ]].astype(bool).sum()
        
        #print "-- DNA observed = %d"%(nbr_observed)
        
      elif layer_name == DNA_INPUT or layer_name == DNA_TARGET:
        dna_data = np.zeros( (len(batch_barcodes), self.n_dna_channels,self.dna_dim) )
        for idx,DNA_key in zip(range(len(self.DNA_keys)),self.DNA_keys):
          batch_data = self.data_store[DNA_key].loc[ batch_barcodes ].fillna( 0 ).values
          if mode == "TEST" or mode == "VAL" or mode == "TRAIN" :
            dna_data[:,idx,:] = batch_data
          else:
            #if layer_name == DNA_TARGET:
            dna_data[:,idx,:] = self.AddNoise( batch_data )
          #
          #dna_data.append(batch_data.fillna( 0 ).values)
        batch[ layer_name ] = dna_data# np.array( dna_data )
        
      elif layer_name == METH_INPUT or layer_name == METH_TARGET:
        batch_data = self.data_store[self.METH_key].loc[ batch_barcodes ]
        batch[ layer_name ] = batch_data.fillna( 0 ).values

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
            
            
      
      
        #batch[ layer_name ] = w[:,np.newaxis]*batch_data
        batch[ layer_name ] = batch_data
    
      elif layer_name == INPUT_MISSING:
        batch_data = batch[ "observed_sources" ][:,self.observation_source_indices]
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
    
    # for layer_name, layer in self.network.dropouts.iteritems():
    #   #print "** Found dropout layer"
    #   if mode == "BATCH":
    #     #print "** setting dropout keep rate"
    #     batch[ layer_name ] = self.keep_rates[ layer_name ]
    return batch            
    
  def NormalizeRnaInput( self, X ):
    return X
    return 2*X-1.0

  def NormalizeRnaTarget( self, X ):
    #return 0.0005+0.999*X
    return X
    
  def AddNoise( self, X, rate1=0.01, rate2=0.001 ):
    return X
    
    a,b = X.shape

    x = X.flatten()

    I=pp.find(x>0)
    J=pp.find(x==0)

    r1 = np.random.rand(len(I)) < rate1
    r2 = np.random.rand(len(J)) < rate2

    x[I[r1]]=0
    x[J[r2]]=1

    return x.reshape((a,b))
     
  

    
      
      
      
            