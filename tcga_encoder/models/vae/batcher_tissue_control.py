from tcga_encoder.models.vae.batcher_ABC import *

class TCGABatcherTissueControl( TCGABatcherABC ):
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
    
    self.dims_dict = {miRNA:self.mirna_dim,RNA:self.rna_dim, DNA:self.dna_dim, TISSUE:self.tissue_dim, METH:self.meth_dim}
    self.data_store     = self.data_dict[DATASET].store
    self.batch_size     = self.algo_dict[BATCH_SIZE]
    #self.n_test         = self.algo_dict[N_TEST]
    #self.n_full_train = self.algo_dict[N_TRAIN_FULL]
    #self.n_non_full_train  = self.algo_dict[N_TRAIN_NON_FULL]
    #self.train_mode = self.algo_dict[TRAIN_MODE]
    
    # keep BRCA in train and test sets, if False, move them all to validation
    #self.include_brca = bool(self.algo_dict["include_brca"])
    #self.nbr_brca_train = int(self.algo_dict["nbr_brca_train"])
    # keep non-full tissues in validation, otherwise move to train
    #self.include_validation = bool(self.algo_dict["include_validation"])
    
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
    self.tissue2color = OrderedDict()
    self.tissue2shape = OrderedDict()
    cmap = pp.get_cmap('Set1')
    colors = cmap(np.linspace(0, 1, len(self.tissue_names)))
    for idx,t in zip( range(len(colors)),self.tissue_names):
      self.tissue2color[t] = colors[idx]
      self.tissue2shape[t] = 'o'
    
      

    
    self.source2darkcolor = {RNA:"darkblue", DNA:"darkgreen", METH:"darkred", miRNA:"darkorange"}
    self.source2lightcolor = {RNA:"lightblue", DNA:"palegreen", METH:"lightsalmon", miRNA:"moccasin"}
    self.source2mediumcolor = {RNA:"dodgerblue", DNA:"darksage", METH:"red", miRNA:"orange"}
    
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
          
        # if nbr == 2:
        #   self.input_combo2fillcolor[inputs] = self.source2mediumcolor[inputs]
        #   self.input_combo2shape[inputs] = 's'
        #   self.input_combo2markersize[inputs] = 8
        #
        # if nbr == 3:
        #   self.input_combo2fillcolor[inputs] = self.source2mediumcolor[inputs]
        #   self.input_combo2shape[inputs] = 'o'
        #   self.input_combo2markersize[inputs] = 10
          

        
    self.observation_source_indices = []
    for source in self.arch_dict["sources"]:
      self.observation_source_indices.append( self.observed_source2idx[source] )

    self.observation_source_indices_input = []
    for source in self.arch_dict[INPUT_SOURCES]:
      self.observation_source_indices_input.append( self.observed_source2idx[source] )

    self.observation_source_indices_target = []
    for source in self.arch_dict[TARGET_SOURCES]:
      self.observation_source_indices_target.append( self.observed_source2idx[source] )                

    self.observed_batch_order = OrderedDict()
    #self.observed_batch_order[RNA] = 0
    #self.observed_batch_order[METH] = 1
    #self.observed_batch_order[miRNA] = 2

    
    self.observed_product_sources = []
    source_idx = 0
    for source in self.arch_dict["product_sources"]:
      self.observed_product_sources.append( self.observed_source2idx[source] )     
      self.observed_batch_order[ source ] = source_idx
      source_idx+=1
       
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
      #self.validation_obs_query[i.values] = True
      for bc in self.validation_tissue2barcodes[ tissue ]:
        self.validation_obs_query[ self.obs_store_bc_2idx[bc] ] = True
      #self.validation_obs_query[ids] = True
      print tissue, self.validation_obs_query.sum()
      #if tissue=="coad":
      #  pdb.set_trace()
      
    #coad_bc = "coad_tcga-t9-a92h"
    
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
  
    #if self.n_fully_observed <= self.n_full_train:
    #  print "==> setting test set from %d to %d"%(self.n_full_train,self.n_fully_observed)
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
    # if self.train_mode == ALL:
    #   print "************************"
    #   print "************************"
    #   print "ALL ALL ALL ALL ALL ALL "
    #   print "************************"
    #   print "************************"
    #   self.train_barcodes = np.union1d( self.train_full_barcodes, self.train_non_full_barcodes )
    # else:
    #   print "************************"
    #   print "************************"
    #   print "FULL FULL FULL FULL FULL"
    #   print "************************"
    #   print "************************"
    #   self.train_barcodes = self.train_full_barcodes
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
    
    self.fill_store.open()
    z_columns = ["z%d"%zidx for zidx in range(self.n_z)]

    self.fill_store["Z/TRAIN/Z/mu"]  = pd.DataFrame( np.zeros( (len(self.train_barcodes),self.n_z) ), index = self.train_barcodes, columns = z_columns )
    self.fill_store["Z/TRAIN/Z/var"] = pd.DataFrame( np.zeros( (len(self.train_barcodes),self.n_z) ), index = self.train_barcodes, columns = z_columns )
    
    #pdb.set_trace()
    self.fill_store.close()
    
    
    self.n_test = len(self.test_barcodes)
    self.n_val = len(self.validation_barcodes)
    
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
      
      
      
            