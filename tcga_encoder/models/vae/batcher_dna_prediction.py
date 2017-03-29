from tcga_encoder.models.vae.batcher_ABC import *


  
class DnaBatcher( TCGABatcherABC ):
  
  def CallBack( self, function_name, sess, cb_info ):
    if function_name == "everything":
      self.FillDna( sess, cb_info )
      #self.TestFillZ( sess, cb_info )
      #self.TrainFillZ( sess, cb_info )
      self.SaveModel( sess, cb_info )
      
      self.BatchEpoch( sess, cb_info )
      self.TestEpoch( sess, cb_info )
      self.ValEpoch( sess, cb_info )
      self.VizEpochs( sess, cb_info )
      self.VizModel( sess, cb_info )
  
  
  def FillDna( self, sess, info_dict ):
    epoch       = info_dict[EPOCH]
    # feed_dict   = info_dict[TEST_FEED_DICT]
    # impute_dict = info_dict[TEST_FEED_IMPUTATION]
    #
    # self.RunFillZ( epoch, sess, feed_dict, impute_dict, mode="TEST" )
    
    feed_dict   = info_dict[VAL_FEED_DICT]
    impute_dict = info_dict[VAL_FEED_IMPUTATION]
    
    self.RunFillDna( epoch, sess, feed_dict, impute_dict, mode="VAL" )
    
    feed_dict   = info_dict[BATCH_FEED_DICT]
    impute_dict = info_dict[BATCH_FEED_IMPUTATION]
    self.batch_ids = info_dict["batch_ids"]
    self.RunFillDna( epoch, sess, feed_dict, impute_dict, mode="BATCH" )
    
    for batch_ids in chunks( np.arange(len(self.train_barcodes)), 5000 ):
      barcodes = self.train_barcodes[batch_ids]
      impute_dict = self.FillBatch( barcodes, mode = "TRAIN" ) #self.NextBatch(batch_ids)
      #impute_dict[BARCODES] = barcodes
      self.batch_ids = batch_ids
      
      
      
      train_feed_dict={}
      self.network.FillFeedDict( train_feed_dict, impute_dict )
      #batch = self.FillBatch( impute_dict[BARCODES], mode )
      self.RunFillDna( epoch, sess, train_feed_dict, impute_dict, mode="TRAIN" )
      
        
  def GetAlgoDictStuff(self):
    pass
   
  def SummarizeData(self):
    print "Running: SummarizeData()"
    self.dna_mean = self.data_store[self.DNA_keys[0]].loc[self.train_barcodes].mean(0)
    self.dna_std = self.data_store[self.DNA_keys[0]].loc[self.train_barcodes].std(0)

    self.dna_order = np.argsort( self.dna_mean.values )
    
    self.tissue_statistics = {}
    
    #pdb.set_trace()
    tissue_names = self.train_tissue.columns
    stats = np.zeros( (5,len(tissue_names)))
    for t_idx, tissue in zip( range(len(tissue_names)),tissue_names ):
      bcs = self.train_tissue.loc[self.train_tissue[tissue]==1].index.values
      
      #pdb.set_trace()
      
      #mirna=self.data_store[self.miRNA_key].loc[ bcs ]
      
      
      self.tissue_statistics[ tissue ] = {}
      self.tissue_statistics[ tissue ][ DNA ] = {}
      self.tissue_statistics[ tissue ][ DNA ][ "mean"]   = self.data_store[self.DNA_keys[0]].mean(0).fillna(0)
      self.tissue_statistics[ tissue ][ DNA ][ "var"]   = self.data_store[self.DNA_keys[0]].var(0).fillna(0)

      try:
        dna=self.data_store[self.DNA_keys[0]].loc[ bcs ]
        self.tissue_statistics[ tissue ][ DNA ][ "mean"]   = dna.mean(0).fillna(0)
        self.tissue_statistics[ tissue ][ DNA ][ "var"]   = dna.var(0).fillna(0)
      except:
        print "No DNA for %s"%(tissue)   
  
  def InitializeAnythingYouWant(self, sess, network ):
    print "Running : InitializeAnythingYouWant"
    input_sources = ["DNA"] 
    layers = ["dna_predictions"]
    
    n_tissues = len(self.data_store[self.TISSUE_key].columns)
    #self.data_store[self.TISSUE_key].loc[ batch_barcodes ]
    m = self.dna_mean.values + 1e-5
    beta_0 = np.log( m ) - np.log( 1.0 - m )
    
    if np.any(np.isnan(beta_0)) or np.any(np.isinf(beta_0)):
      pdb.set_trace()
    # get log_alpha and log_beta values
    for layer_name, input_name in zip( layers, input_sources ):
      n_dims = self.dims_dict[ input_name ]
      
      alpha = np.zeros( (self.n_z, n_dims ), dtype = float )
      beta  = np.zeros( (n_tissues, n_dims ), dtype = float )
      
      for t_idx, tissue in zip( range( n_tissues), self.data_store[self.TISSUE_key].columns):
        
        n_samples = self.train_tissue[ tissue ].sum()
        m = self.tissue_statistics[ tissue ][ DNA ][ "mean"].values + 1e-5
        
        beta[t_idx,:] = np.log( m ) - np.log( 1.0 - m )
        if np.any(np.isnan(beta[t_idx,:])) or np.any(np.isinf(beta[t_idx,:])):
          pdb.set_trace()
        
      
      #log_alpha = np.log( alpha + 0.001 ).astype(np.float32)
      #log_beta = np.log( beta + 0.001).astype(np.float32)
      
      #layer = network.GetLayer( layer_name )
      
      #sess.run( tf.assign(layer.weights[0][0], log_alpha) )
      #sess.run( tf.assign(layer.weights[1][0], log_beta) )
      if len(network.GetLayer( layer_name ).weights) == 2:
        # 
        print "initialize as if log reg and tissue specific biases"
        #pdb.set_trace()
        network.GetLayer( layer_name ).SetWeights( sess, [alpha, beta ])
      else:
        if network.GetLayer( layer_name ).biases is not None:
          print "initialize with tissue specific biases"
          network.GetLayer( layer_name ).SetBiases( sess, [beta_0])
         
  def StoreNames(self):
    #self.latent_store_name = self.network_name + "_" + LATENT
    #self.latent_store = OpenHdfStore(self.savedir, self.latent_store_name, mode=self.default_store_mode )
    self.model_store_name = self.network_name + "_DNA_" + MODEL
    #self.survival_store_name = self.network_name + "_" + SURVIVAL
    
    # open in "a" mode
    self.model_store = OpenHdfStore(self.savedir, self.model_store_name, mode="a" )
    
    self.epoch_store_name = self.network_name + "_DNA_" + EPOCH
    self.epoch_store = OpenHdfStore(self.savedir, self.epoch_store_name, mode=self.default_store_mode )
    
    self.fill_store_name = self.network_name + "_DNA_" + FILL
    self.fill_store = OpenHdfStore(self.savedir, self.fill_store_name, mode="a")
    
    self.fill_source_store_name = self.network_name + "_" +  FILL
    self.fill_source_store = OpenHdfStore(self.savedir, self.fill_source_store_name, mode="r")

    
    self.fill_source_store.close()
    
    self.fill_store.close()
    #self.latent_store.close()
    self.model_store.close()
    self.epoch_store.close()
  
  def CloseAll(self):
    self.data_store.close()
    self.fill_source_store.close()
    self.model_store.close()
    self.epoch_store.close()
      
  def MakeVizFilenames(self):
    self.viz_filename_dna_batch_target       =  os.path.join( self.savedir, "dna_batch_target" )
    self.viz_filename_dna_batch_predict      =  os.path.join( self.savedir, "dna_batch_predict" )
    self.viz_filename_dna_aucs               =  os.path.join( self.savedir, "dna_aucs.png" )
    self.viz_filename_lower_bound            =  os.path.join( self.savedir, "dna_lower_bound.png" )
    self.viz_filename_error_sources_per_gene_fill = os.path.join( self.savedir, "dna_errors_fill.png" )
    self.viz_dna_weights = os.path.join( self.savedir, "dna_weights" )
    
  def PlotLogPdf(self):
    f = pp.figure()
    #pdb.set_trace()
    pp.plot( self.epoch_store["Batch"]["Epoch"].values, self.epoch_store["Batch"]["log p(x)"], 'bo-', lw=2 , label="Batch")
    if self.n_test > 0:
      pp.plot( self.epoch_store["Test"]["Epoch"].values, self.epoch_store["Test"]["log p(x)"], 'ro-', lw=2, label="Test" )
    if self.n_val > 0:
      pp.plot( self.epoch_store["Val"]["Epoch"].values, self.epoch_store["Val"]["log p(x)"], 'ro-', lw=2, label="Val" )
    pp.legend( loc="lower right")
    pp.xlabel("Epoch")
    pp.ylabel("log p(x)")
    pp.grid('on')

    pp.savefig( self.viz_filename_lower_bound, dpi = 300, fmt="png", bbox_inches = "tight")
    pp.close(f)

  def FillDerivedPlaceholder( self, batch, layer_name, mode ):
    
    if layer_name == "Z_input":
      self.fill_source_store.open()
      if mode == "BATCH" or mode == "TRAIN":
        #pdb.set_trace()
        batch_data = self.fill_source_store["/Z/TRAIN/Z/mu"].loc[ batch["barcodes"] ]

      else:
        #pdb.set_trace()
        batch_data = self.fill_source_store["/Z/VAL/Z/mu"].loc[ batch["barcodes"] ]
        
        
      nans = np.isnan( batch_data.values )
      batch_data_values = batch_data.values
      # if mode == "BATCH":
      #  batch_data_values = self.AddmiRnaNoise( batch_data.values, rate = 0.1 )
      #   
      # batch[ layer_name ] = self.NormalizemiRnaInput( batch_data_values )
      batch[ layer_name ] = batch_data
      batch[ layer_name ][nans] = 0
      self.fill_source_store.close()
      
    
  # def PlotLogPdf(self, main_sources, prior_sources ):
  #   f = pp.figure()
  #   legends  = []
  #   colours = "bgr"
  #   fill_colours = ["lightblue","lightgreen","lightred"]
  #   n_sources = len(main_sources)
  #   for idx,target_source, prior_source in zip( range(n_sources),main_sources,prior_sources):
  #     s = f.add_subplot(1,n_sources,idx+1)
  #     pp.plot( self.epoch_store[BATCH_SOURCE_LOGPDF]["Epoch"].values,
  #              self.epoch_store[BATCH_SOURCE_LOGPDF][target_source]/self.dims_dict[target_source], 's--', \
  #              color=self.source2mediumcolor[target_source], \
  #              mec=self.source2darkcolor[target_source], mew=1, \
  #              mfc=self.source2lightcolor[target_source], lw=1, \
  #              ms = 5, \
  #              alpha=0.75, \
  #              label="Batch (%0.4f)"%(self.epoch_store[BATCH_SOURCE_LOGPDF][prior_source].values[-1]/self.dims_dict[target_source]) )
  #     pp.plot( self.epoch_store[BATCH_SOURCE_LOGPDF]["Epoch"].values,
  #              self.epoch_store[BATCH_SOURCE_LOGPDF][prior_source]/self.dims_dict[prior_source], 's--', \
  #              color=self.source2mediumcolor[prior_source], \
  #              mec=self.source2darkcolor[prior_source], mew=1, \
  #              mfc=self.source2lightcolor[prior_source], lw=1, \
  #              ms = 5, \
  #              alpha=0.75, \
  #              label="Batch prior (%0.4f)"%(self.epoch_store[BATCH_SOURCE_LOGPDF][prior_source].values[-1]/self.dims_dict[prior_source]) )
  #
  #     if self.n_test > 0:
  #       pp.plot( self.epoch_store[TEST_SOURCE_LOGPDF]["Epoch"].values, \
  #              self.epoch_store[TEST_SOURCE_LOGPDF][target_source]/self.dims_dict[target_source], 'o-', \
  #              color=self.source2mediumcolor[target_source],\
  #              mec=self.source2darkcolor[target_source], mew=2, \
  #              mfc=self.source2lightcolor[target_source], lw=3, \
  #              ms = 8, \
  #              label="Test  (%0.4f)"%(self.epoch_store[TEST_SOURCE_LOGPDF][target_source].values[-1]/self.dims_dict[target_source])  )
  #       pp.plot( self.epoch_store[TEST_SOURCE_LOGPDF]["Epoch"].values, \
  #              self.epoch_store[TEST_SOURCE_LOGPDF][prior_source]/self.dims_dict[prior_source], 'o-', \
  #              color=self.source2mediumcolor[prior_source],\
  #              mec=self.source2darkcolor[prior_source], mew=2, \
  #              mfc=self.source2lightcolor[prior_source], lw=3, \
  #              ms = 8, \
  #              label="Test prior (%0.4f)"%(self.epoch_store[TEST_SOURCE_LOGPDF][prior_source].values[-1]/self.dims_dict[prior_source])  )
  #
  #
  #     if self.n_val > 0:
  #       pp.plot( self.epoch_store[VAL_SOURCE_LOGPDF]["Epoch"].values, \
  #              self.epoch_store[VAL_SOURCE_LOGPDF][target_source]/self.dims_dict[target_source], 'v-', \
  #              color=self.source2darkcolor[target_source],\
  #              mec=self.source2darkcolor[target_source], mew=2, \
  #              mfc=self.source2lightcolor[target_source], lw=3, \
  #              ms = 8, \
  #              label="Val  (%0.4f)"%(self.epoch_store[VAL_SOURCE_LOGPDF][target_source].values[-1]/self.dims_dict[target_source])  )
  #       pp.plot( self.epoch_store[VAL_SOURCE_LOGPDF]["Epoch"].values, \
  #              self.epoch_store[VAL_SOURCE_LOGPDF][prior_source]/self.dims_dict[prior_source], 'v-', \
  #              color=self.source2darkcolor[prior_source],\
  #              mec=self.source2darkcolor[prior_source], mew=2, \
  #              mfc=self.source2lightcolor[prior_source], lw=3, \
  #              ms = 8, \
  #              label="Val prior (%0.4f)"%(self.epoch_store[VAL_SOURCE_LOGPDF][prior_source].values[-1]/self.dims_dict[prior_source])  )
  #
  #
  #     if idx==0:
  #       pp.ylabel("log p(x|z)") #%(target_source))
  #     pp.legend(loc="lower right")
  #     pp.title( "%s"%(target_source))
  #     pp.xlabel("Epoch")
  #
  #   pp.grid('on')
  #   #pdb.set_trace()
  #
  #   pp.savefig( self.viz_filename_log_pdf_sources_per_gene, dpi = 300, fmt="png", bbox_inches = "tight")
  #   pp.close(f)


  # def PlotFillLogPdf(self,main_sources,prior_sources):
  #   f = pp.figure()
  #   legends  = []
  #   colours = "bgr"
  #   fill_colours = ["lightblue","lightgreen","lightred"]
  #   n_sources = len(main_sources)
  #   for idx,target_source, prior_source in zip( range(n_sources),main_sources,prior_sources):
  #     s = f.add_subplot(1,n_sources,idx+1)
  #     pp.plot( self.epoch_store[BATCH_SOURCE_LOGPDF]["Epoch"].values,
  #              self.epoch_store[BATCH_SOURCE_LOGPDF][target_source]/self.dims_dict[target_source], 's--', \
  #              color=self.source2mediumcolor[target_source], \
  #              mec=self.source2darkcolor[target_source], mew=1, \
  #              mfc=self.source2lightcolor[target_source], lw=1, \
  #              ms = 5, \
  #              alpha=0.75, \
  #              label="Batch (%0.4f)"%(self.epoch_store[BATCH_SOURCE_LOGPDF][target_source].values[-1]/self.dims_dict[target_source]) )
  #     pp.plot( self.epoch_store[BATCH_SOURCE_LOGPDF]["Epoch"].values,
  #              self.epoch_store[BATCH_SOURCE_LOGPDF][prior_source]/self.dims_dict[prior_source], 's--', \
  #              color=self.source2mediumcolor[prior_source], \
  #              mec=self.source2darkcolor[prior_source], mew=1, \
  #              mfc=self.source2lightcolor[prior_source], lw=1, \
  #              ms = 5, \
  #              alpha=0.75, \
  #              label="Batch prior (%0.4f)"%(self.epoch_store[BATCH_SOURCE_LOGPDF][prior_source].values[-1]/self.dims_dict[prior_source]) )
  #
  #     if self.n_test > 0:
  #       query1 = self.epoch_store[TEST_FILL_LOGLIK]["Target"] == target_source
  #       query2 = self.epoch_store[TEST_FILL_LOGLIK]["Target"] == prior_source
  #       query = query1#&query2
  #       loglik_df = self.epoch_store[TEST_FILL_LOGLIK][query]
  #       epochs = loglik_df["Epoch"].values
  #       loglik = loglik_df["LogLik"].values
  #       if len(loglik) == 0:
  #         continue
  #       pp.plot( epochs, loglik, 'o-', \
  #              color=self.source2darkcolor[target_source],\
  #              mec=self.source2darkcolor[target_source], mew=1, \
  #              mfc=self.source2lightcolor[target_source], lw=2, \
  #              ms = 8, \
  #              label="Test (%0.4f)"%(loglik[-1]) )
  #
  #       query = query2#&query2
  #       loglik_df = self.epoch_store[TEST_FILL_LOGLIK][query]
  #       epochs = loglik_df["Epoch"].values
  #       loglik = loglik_df["LogLik"].values
  #       if len(loglik) == 0:
  #         continue
  #       pp.plot( epochs, loglik, 'o-', \
  #              color=self.source2darkcolor[prior_source],\
  #              mec=self.source2darkcolor[prior_source], mew=1, \
  #              mfc=self.source2lightcolor[prior_source], lw=2, \
  #              ms = 8, \
  #              label="Test prior (%0.4f)"%(loglik[-1]) )
  #
  #     if self.n_val > 0:
  #       query1 = self.epoch_store[VAL_FILL_LOGLIK]["Target"] == target_source
  #       query2 = self.epoch_store[VAL_FILL_LOGLIK]["Target"] == prior_source
  #       query = query1#&query2
  #       loglik_df = self.epoch_store[VAL_FILL_LOGLIK][query]
  #       epochs = loglik_df["Epoch"].values
  #       loglik = loglik_df["LogLik"].values
  #       if len(loglik) == 0:
  #         continue
  #
  #       pp.plot( epochs, loglik, 'v-', \
  #              color=self.source2mediumcolor[target_source],\
  #              mec=self.source2lightcolor[target_source], mew=1, \
  #              mfc=self.source2darkcolor[target_source], lw=2, \
  #              ms = 8, \
  #              label="Val (%0.4f)"%(loglik[-1]) )
  #       query = query2#&query2
  #       loglik_df = self.epoch_store[VAL_FILL_LOGLIK][query]
  #       epochs = loglik_df["Epoch"].values
  #       loglik = loglik_df["LogLik"].values
  #       if len(loglik) == 0:
  #         continue
  #
  #       pp.plot( epochs, loglik, 'v-', \
  #              color=self.source2mediumcolor[prior_source],\
  #              mec=self.source2lightcolor[prior_source], mew=1, \
  #              mfc=self.source2darkcolor[prior_source], lw=2, \
  #              ms = 8, \
  #              label="Val prior (%0.4f)"%(loglik[-1]) )
  #
  #
  #     if idx==0:
  #       pp.ylabel("log p(x|z)") #%(target_source))
  #     pp.legend(loc="lower right")
  #     pp.title( "%s"%(target_source))
  #     pp.xlabel("Epoch")
  #
  #   pp.grid('on')
  #   pp.savefig( self.viz_filename_log_pdf_sources_per_gene_fill, dpi = 300, fmt="png", bbox_inches = "tight")
  #   pp.close(f)


  def PlotFillError(self,main_sources):
    f = pp.figure(figsize=(12,10))
    legends  = []
    n_sources = len(main_sources)
    for idx,target_source in zip( range(n_sources),main_sources):
      s = f.add_subplot(1,n_sources,idx+1)
      inputs = "RNA+DNA+METH"
      # query1 = self.epoch_store[BATCH_FILL_ERROR]["Target"] == target_source
      # query = query1#&query2
      # df = self.epoch_store[BATCH_FILL_ERROR][query]
      # epochs = df["Epoch"].values
      # loglik = df["Error"].values
      # if len(loglik) == 0:
      #   continue
      # pp.plot( epochs, loglik, 'o-', \
      #          color=self.source2lightcolor[target_source],\
      #          mec=self.source2mediumcolor[target_source], mew=1, \
      #          mfc=self.source2lightcolor[target_source], lw=2, \
      #          ms = 8, \
      #          label="Batch (%0.6f)"%(loglik[-1]) )
      
      
      if self.n_test > 0:
        query1 = self.epoch_store[TEST_FILL_ERROR]["Target"] == target_source
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
    pp.close(f)
              
  def VizEpochs(self, sess, info_dict ):
    print "** VIZ Epochs"
    main_sources = [DNA]
    #prior_sources = [miRNA+"_b", RNA+"_b", METH+"_b"]

    self.epoch_store.open()
  
    self.PlotLogPdf()

    #self.PlotLogPdf(main_sources,prior_sources)
    
    #self.PlotFillLogPdf(main_sources,prior_sources)
    
    self.PlotFillError(main_sources)
    
    self.PlotAucs()
    
    self.epoch_store.close()
    pp.close('all')
      
  def RunFillDna( self, epoch, sess, feed_dict, impute_dict, mode ):
    print "COMPUTE Z-SPACE"
    use_dna = False
    use_rna = True
    use_meth = True
    use_mirna = True
          
    barcodes = impute_dict[BARCODES]
    batch = self.FillBatch( impute_dict[BARCODES], mode )
    #not_observed = np.setdiff1d( self.input_sources, inputs2use )
    #pdb.set_trace()     
    dna_expectation_tensor = self.network.GetLayer( "dna_predictions" ).expectation
    dna_data = np.zeros( (len(barcodes),self.dna_dim) )
    for idx,DNA_key in zip(range(len(self.DNA_keys)),self.DNA_keys):
      batch_data = self.data_store[DNA_key].loc[ barcodes ].fillna( 0 ).values
      dna_data += batch_data
    
    dna_data = np.minimum(1.0,dna_data)
      
    loglikes_data_as_matrix = self.network.loglikes_data_as_matrix
  
    tensors = [dna_expectation_tensor]
    tensor_names = ["dna_predictions"]
  
    assert len(tensor_names)==len(tensors), "should be same number"
    self.network.FillFeedDict( feed_dict, impute_dict )

    #pdb.set_trace()
    # rna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[RNA]] == 1
    # meth_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[METH]] == 1
    # mirna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[miRNA]] == 1
    dna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[DNA]] == 1
    
    tensor2fill = []
    tensor2fill.extend( [dna_expectation_tensor, loglikes_data_as_matrix["dna_predictions"] ] )
    z_ids = [0,1]        
      
    # ---------
    # RUN SESS
    # ---------
    self.network.FillFeedDict( feed_dict, batch )
    tensor2fill_eval = sess.run( tensor2fill, feed_dict = feed_dict )

    # ------
    # FILL EVALUATION 
    # -----    
    dna_expectation = tensor2fill_eval[0]
    dna_loglikelihood = tensor2fill_eval[1]

    #pdb.set_trace()
    self.WriteRunFillExpectation( epoch, DNA, barcodes, self.dna_genes, dna_observed_query, dna_expectation, dna_data, mode )
    self.WriteRunFillLoglikelihood( epoch, DNA, barcodes[dna_observed_query], self.dna_genes, dna_loglikelihood, mode )
    
    self.WriteAucs( epoch, DNA, barcodes, self.dna_genes, dna_observed_query, dna_expectation, dna_data, mode )

  def WriteAucs( self, epoch, target, barcodes, columns, obs_query, X, Y, mode ):
    #inputs = inputs2use[0]
    #for s in inputs2use[1:]:
    #  inputs += "+%s"%(s)
    print "Running: WriteAucs"
    self.fill_store.open()
    if target == DNA:
      #for channel in range(self.n_dna_channels):
      s = "/AUC/%s/%s/"%(mode,target )
      #self.fill_store[ s ] = pd.DataFrame( X, index = barcodes, columns = columns )
      x_obs = X[obs_query,:] #.flatten()
      y_obs = Y[obs_query,:] # .flatten()
      
      auc = np.zeros( x_obs.shape[1] )
      ok = np.zeros( x_obs.shape[1] )
      for d_idx in xrange( x_obs.shape[1] ):
        
        if y_obs[:,d_idx].sum()>0:
          auc[d_idx] = roc_auc_score(y_obs[:,d_idx],x_obs[:,d_idx])
          ok[d_idx] = 1
        # else:
        #   auc[d_idx] = 1.0
      #errors = 1.0-auc
      #pdb.set_trace()
       
      ok = pp.find(ok)
      auc = auc[ ok ]
      columns = columns[ ok ]
      self.fill_store[ s ] = pd.DataFrame( auc.reshape((1,len(auc))), columns = columns )

    
    self.fill_store.close()
  
  def PlotAucs( self ):
    self.fill_store.open()
    #pdb.set_trace()
    s = "/AUC/VAL/%s/"%(DNA )
    
    f = pp.figure(figsize=(12,10))
    ax=f.add_subplot(111)
    df = self.fill_store[s]

    I = np.argsort( np.squeeze(df.values))
    
    
    
    mean = self.tissue_statistics[ self.validation_tissues[0] ][ DNA ][ "mean"]
    sorted_mean = pd.DataFrame( np.squeeze(mean.values)[I].reshape((1,len(I))), columns = np.array(df.columns)[I] )
    sorted_all_mean = pd.DataFrame( np.squeeze(self.dna_mean.values)[I].reshape((1,len(I))), columns = np.array(df.columns)[I] )
    sorted = pd.DataFrame( np.squeeze(df.values)[I].reshape((1,len(I))), columns = np.array(df.columns)[I] )
    #pdb.set_trace()

    sorted_mean.T.plot(kind='bar',ax=ax, sharex=True)
    sorted.T.plot(ax=ax)
    sorted_all_mean.T.plot(kind='bar',ax=ax, fontsize=6, sharex=True)
    sorted_mean.T.plot(kind='bar',ax=ax, fontsize=6, sharex=True)
                 
    pp.title( "mean = %0.3f"%(df.values.mean()))
    pp.savefig( self.viz_filename_dna_aucs, fmt="png", bbox_inches = "tight")
    self.fill_store.close()
    
  def Epoch( self, epoch_key, sess, info_dict, epoch, feed_dict, impute_dict, mode ):  
    barcodes = impute_dict[BARCODES]
    batch_tensor_evals = sess.run( self.network.batch_log_tensors, feed_dict = feed_dict )
    
    # batch_counts = self.CountSourcesInDict( impute_dict )
    #
    # n_batch = []
    # for source in self.arch_dict[TARGET_SOURCES]:
    #   n_batch.append( batch_counts[source] )
    # n_batch = np.array(n_batch).astype(float)
    #
    n_batch_size = len(impute_dict[BARCODES])
    #
    # log_p_z         = batch_tensor_evals[2]/float(n_batch_size)
    # log_q_z         = batch_tensor_evals[3]/float(n_batch_size)
    #
    # # normalize by nbr observed for each source
    # log_p_source_z_values = batch_tensor_evals[4:]/n_batch
    #
    # #print np.sort(info_dict[BATCH_IDS])
    # new_log_p_x_given_z = log_p_source_z_values.sum()
    # lower_bound = log_p_z-log_q_z + new_log_p_x_given_z
    
    new_values = [epoch]
    new_values.extend( batch_tensor_evals )
    new_values[1]/=n_batch_size
    self.AddSeries(  self.epoch_store, epoch_key, values = new_values, columns = self.network.batch_log_columns )
    
    epoch_values = [epoch]
    epoch_values.extend( batch_tensor_evals )
    #epoch_columns = ['Epoch']
    epoch_columns = self.network.batch_log_columns
    
    #pdb.set_trace()
    if mode == "BATCH":
      self.AddSeries(  self.epoch_store, BATCH_SOURCE_LOGPDF, values = epoch_values, columns = epoch_columns )
      self.PrintRow( self.epoch_store, epoch_key )
    elif mode == "TEST" and self.n_test>0: 
      self.AddSeries(  self.epoch_store, TEST_SOURCE_LOGPDF, values = epoch_values, columns = epoch_columns )
      self.PrintRow( self.epoch_store, epoch_key )
    elif mode == "VAL" and self.n_val>0:
      self.AddSeries(  self.epoch_store, VAL_SOURCE_LOGPDF, values = epoch_values, columns = epoch_columns )
      self.PrintRow( self.epoch_store, epoch_key )
 
  def VizWeightsGeneric( self, sess, info_dict ):    
    print "  -> Generic Viz" 
    self.model_store.open()
    keys = self.model_store.keys()
    
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
      pp.savefig( self.viz_dna_weights + "%s.png"%old_layer, fmt="png", bbox_inches = "tight")
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
   

      
  def VizModel( self, sess, info_dict ): 
    print "** VIZ Model"    
    self.VizWeightsGeneric(sess, info_dict )


  # def InitializeAnythingYouWant(self, sess, network ):
  #   pass
    # print "Running : InitializeAnythingYouWant"
    # input_sources = ["METH","RNA","miRNA"]
    # layers = ["gen_meth_space_basic","gen_rna_space_basic","gen_mirna_space_basic"]
    #
    # n_tissues = len(self.data_store[self.TISSUE_key].columns)
    # #self.data_store[self.TISSUE_key].loc[ batch_barcodes ]
    #
    # # get log_alpha and log_beta values
    # for layer_name, input_name in zip( layers, input_sources ):
    #   n_dims = self.dims_dict[ input_name ]
    #
    #   alpha = np.zeros( (n_tissues, n_dims ), dtype = float )
    #   beta  = np.zeros( (n_tissues, n_dims ), dtype = float )
    #
    #   for t_idx, tissue in zip( range( n_tissues), self.data_store[self.TISSUE_key].columns):
    #
    #     n_samples = self.train_tissue[ tissue ].sum()
    #     alpha[t_idx,:] = self.tissue_statistics[ tissue ][ input_name ][ "alpha"]
    #     beta[t_idx,:] = self.tissue_statistics[ tissue ][ input_name ][ "beta"]
    #
    #   log_alpha = np.log( alpha + 0.001 ).astype(np.float32)
    #   log_beta = np.log( beta + 0.001).astype(np.float32)
    #
    #   #layer = network.GetLayer( layer_name )
    #
    #   #sess.run( tf.assign(layer.weights[0][0], log_alpha) )
    #   #sess.run( tf.assign(layer.weights[1][0], log_beta) )
    #   network.GetLayer( layer_name ).SetWeights( sess, [log_alpha, log_beta ])
    #   #pdb.set_trace()
    