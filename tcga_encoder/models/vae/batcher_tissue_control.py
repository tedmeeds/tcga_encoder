from tcga_encoder.models.vae.batcher_ABC import *


  
class TCGABatcher( TCGABatcherABC ):
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
    self.viz_scaled_weights = os.path.join( self.savedir, "weights_scaled_inputs.png" )
    
  def PlotLowerBound(self):
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
  
  def PlotLogPdf(self, main_sources, prior_sources ):
    f = pp.figure()
    legends  = []
    colours = "bgr"
    fill_colours = ["lightblue","lightgreen","lightred"]
    n_sources = len(main_sources)
    for idx,target_source, prior_source in zip( range(n_sources),main_sources,prior_sources):
      s = f.add_subplot(1,n_sources,idx+1)
      pp.plot( self.epoch_store[BATCH_SOURCE_LOGPDF]["Epoch"].values, 
               self.epoch_store[BATCH_SOURCE_LOGPDF][target_source]/self.dims_dict[target_source], 's--', \
               color=self.source2mediumcolor[target_source], \
               mec=self.source2darkcolor[target_source], mew=1, \
               mfc=self.source2lightcolor[target_source], lw=1, \
               ms = 5, \
               alpha=0.75, \
               label="Batch (%0.4f)"%(self.epoch_store[BATCH_SOURCE_LOGPDF][prior_source].values[-1]/self.dims_dict[target_source]) )
      pp.plot( self.epoch_store[BATCH_SOURCE_LOGPDF]["Epoch"].values, 
               self.epoch_store[BATCH_SOURCE_LOGPDF][prior_source]/self.dims_dict[prior_source], 's--', \
               color=self.source2mediumcolor[prior_source], \
               mec=self.source2darkcolor[prior_source], mew=1, \
               mfc=self.source2lightcolor[prior_source], lw=1, \
               ms = 5, \
               alpha=0.75, \
               label="Batch prior (%0.4f)"%(self.epoch_store[BATCH_SOURCE_LOGPDF][prior_source].values[-1]/self.dims_dict[prior_source]) )

      if self.n_test > 0:
        pp.plot( self.epoch_store[TEST_SOURCE_LOGPDF]["Epoch"].values, \
               self.epoch_store[TEST_SOURCE_LOGPDF][target_source]/self.dims_dict[target_source], 'o-', \
               color=self.source2mediumcolor[target_source],\
               mec=self.source2darkcolor[target_source], mew=2, \
               mfc=self.source2lightcolor[target_source], lw=3, \
               ms = 8, \
               label="Test  (%0.4f)"%(self.epoch_store[TEST_SOURCE_LOGPDF][target_source].values[-1]/self.dims_dict[target_source])  )
        pp.plot( self.epoch_store[TEST_SOURCE_LOGPDF]["Epoch"].values, \
               self.epoch_store[TEST_SOURCE_LOGPDF][prior_source]/self.dims_dict[prior_source], 'o-', \
               color=self.source2mediumcolor[prior_source],\
               mec=self.source2darkcolor[prior_source], mew=2, \
               mfc=self.source2lightcolor[prior_source], lw=3, \
               ms = 8, \
               label="Test prior (%0.4f)"%(self.epoch_store[TEST_SOURCE_LOGPDF][prior_source].values[-1]/self.dims_dict[prior_source])  )


      if self.n_val > 0:
        pp.plot( self.epoch_store[VAL_SOURCE_LOGPDF]["Epoch"].values, \
               self.epoch_store[VAL_SOURCE_LOGPDF][target_source]/self.dims_dict[target_source], 'v-', \
               color=self.source2darkcolor[target_source],\
               mec=self.source2darkcolor[target_source], mew=2, \
               mfc=self.source2lightcolor[target_source], lw=3, \
               ms = 8, \
               label="Val  (%0.4f)"%(self.epoch_store[VAL_SOURCE_LOGPDF][target_source].values[-1]/self.dims_dict[target_source])  )
        pp.plot( self.epoch_store[VAL_SOURCE_LOGPDF]["Epoch"].values, \
               self.epoch_store[VAL_SOURCE_LOGPDF][prior_source]/self.dims_dict[prior_source], 'v-', \
               color=self.source2darkcolor[prior_source],\
               mec=self.source2darkcolor[prior_source], mew=2, \
               mfc=self.source2lightcolor[prior_source], lw=3, \
               ms = 8, \
               label="Val prior (%0.4f)"%(self.epoch_store[VAL_SOURCE_LOGPDF][prior_source].values[-1]/self.dims_dict[prior_source])  )

      
      if idx==0:
        pp.ylabel("log p(x|z)") #%(target_source))
      pp.legend(loc="lower right")
      pp.title( "%s"%(target_source))
      pp.xlabel("Epoch")
    
    pp.grid('on')
    #pdb.set_trace()
  
    pp.savefig( self.viz_filename_log_pdf_sources_per_gene, dpi = 300, fmt="png", bbox_inches = "tight")
    pp.close(f)


  def PlotFillLogPdf(self,main_sources,prior_sources):
    f = pp.figure()
    legends  = []
    colours = "bgr"
    fill_colours = ["lightblue","lightgreen","lightred"]
    n_sources = len(main_sources)
    for idx,target_source, prior_source in zip( range(n_sources),main_sources,prior_sources):
      s = f.add_subplot(1,n_sources,idx+1)
      pp.plot( self.epoch_store[BATCH_SOURCE_LOGPDF]["Epoch"].values, 
               self.epoch_store[BATCH_SOURCE_LOGPDF][target_source]/self.dims_dict[target_source], 's--', \
               color=self.source2mediumcolor[target_source], \
               mec=self.source2darkcolor[target_source], mew=1, \
               mfc=self.source2lightcolor[target_source], lw=1, \
               ms = 5, \
               alpha=0.75, \
               label="Batch (%0.4f)"%(self.epoch_store[BATCH_SOURCE_LOGPDF][target_source].values[-1]/self.dims_dict[target_source]) )
      pp.plot( self.epoch_store[BATCH_SOURCE_LOGPDF]["Epoch"].values, 
               self.epoch_store[BATCH_SOURCE_LOGPDF][prior_source]/self.dims_dict[prior_source], 's--', \
               color=self.source2mediumcolor[prior_source], \
               mec=self.source2darkcolor[prior_source], mew=1, \
               mfc=self.source2lightcolor[prior_source], lw=1, \
               ms = 5, \
               alpha=0.75, \
               label="Batch prior (%0.4f)"%(self.epoch_store[BATCH_SOURCE_LOGPDF][prior_source].values[-1]/self.dims_dict[prior_source]) )

      if self.n_test > 0:
        query1 = self.epoch_store[TEST_FILL_LOGLIK]["Target"] == target_source
        query2 = self.epoch_store[TEST_FILL_LOGLIK]["Target"] == prior_source
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

        query = query2#&query2
        loglik_df = self.epoch_store[TEST_FILL_LOGLIK][query]
        epochs = loglik_df["Epoch"].values
        loglik = loglik_df["LogLik"].values
        if len(loglik) == 0:
          continue
        pp.plot( epochs, loglik, 'o-', \
               color=self.source2darkcolor[prior_source],\
               mec=self.source2darkcolor[prior_source], mew=1, \
               mfc=self.source2lightcolor[prior_source], lw=2, \
               ms = 8, \
               label="Test prior (%0.4f)"%(loglik[-1]) )
               
      if self.n_val > 0:
        query1 = self.epoch_store[VAL_FILL_LOGLIK]["Target"] == target_source
        query2 = self.epoch_store[VAL_FILL_LOGLIK]["Target"] == prior_source
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
        query = query2#&query2
        loglik_df = self.epoch_store[VAL_FILL_LOGLIK][query]
        epochs = loglik_df["Epoch"].values
        loglik = loglik_df["LogLik"].values
        if len(loglik) == 0:
          continue
        
        pp.plot( epochs, loglik, 'v-', \
               color=self.source2mediumcolor[prior_source],\
               mec=self.source2lightcolor[prior_source], mew=1, \
               mfc=self.source2darkcolor[prior_source], lw=2, \
               ms = 8, \
               label="Val prior (%0.4f)"%(loglik[-1]) )
        
                     
      if idx==0:
        pp.ylabel("log p(x|z)") #%(target_source))
      pp.legend(loc="lower right")
      pp.title( "%s"%(target_source))
      pp.xlabel("Epoch")
    
    pp.grid('on')
    pp.savefig( self.viz_filename_log_pdf_sources_per_gene_fill, dpi = 300, fmt="png", bbox_inches = "tight")
    pp.close(f)


  def PlotFillError(self,main_sources,prior_sources):
    f = pp.figure(figsize=(12,10))
    legends  = []
    n_sources = len(main_sources)
    for idx,target_source, prior_source in zip( range(n_sources),main_sources,prior_sources):
      s = f.add_subplot(1,n_sources,idx+1)
      inputs = "RNA+DNA+METH"
      if self.n_test > 0:
        query1 = self.epoch_store[TEST_FILL_ERROR]["Target"] == target_source
        query2 = self.epoch_store[TEST_FILL_ERROR]["Target"] == prior_source
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
        query = query2#&query2
        df = self.epoch_store[TEST_FILL_ERROR][query]
        epochs = df["Epoch"].values
        loglik = df["Error"].values
        if len(loglik) == 0:
          continue
        pp.plot( epochs, loglik, 'o-', \
                 color=self.source2darkcolor[prior_source],\
                 mec=self.source2darkcolor[prior_source], mew=1, \
                 mfc=self.source2lightcolor[prior_source], lw=2, \
                 ms = 8, \
                 label="Test prior (%0.6f)"%(loglik[-1]) )
      if self.n_val > 0:
        query1 = self.epoch_store[VAL_FILL_ERROR]["Target"] == target_source
        query2 = self.epoch_store[VAL_FILL_ERROR]["Target"] == prior_source
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
                 
        query = query2#&query2
        df = self.epoch_store[VAL_FILL_ERROR][query]
        epochs = df["Epoch"].values
        loglik = df["Error"].values
        if len(loglik) == 0:
          continue
        pp.plot( epochs, loglik, 'v-', \
                 color=self.source2mediumcolor[prior_source],\
                 mec=self.source2darkcolor[prior_source], mew=1, \
                 mfc=self.source2lightcolor[prior_source], lw=2, \
                 ms = 8, \
                 label="Val prior (%0.6f)"%(loglik[-1]) )
        
      
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
    main_sources = [miRNA, RNA, METH]
    prior_sources = [miRNA+"_b", RNA+"_b", METH+"_b"]

    self.epoch_store.open()
  
    self.PlotLowerBound()

    self.PlotLogPdf(main_sources,prior_sources)
    
    self.PlotFillLogPdf(main_sources,prior_sources)
    
    self.PlotFillError(main_sources,prior_sources)
    
    self.epoch_store.close()
    pp.close('all')
      
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
    meth_rec_z_space_tensors  = self.network.GetTensor( "rec_z_space_meth" )
    
  
    rna_expectation_tensor = self.network.GetLayer( "gen_rna_space" ).expectation
    mirna_expectation_tensor = self.network.GetLayer( "gen_mirna_space" ).expectation
    meth_expectation_tensor = self.network.GetLayer( "gen_meth_space" ).expectation
    
    rna_basic_expectation_tensor = self.network.GetLayer( "gen_rna_space_basic" ).expectation
    mirna_basic_expectation_tensor = self.network.GetLayer( "gen_mirna_space_basic" ).expectation
    meth_basic_expectation_tensor = self.network.GetLayer( "gen_meth_space_basic" ).expectation
    
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
    tensors.extend(meth_rec_z_space_tensors)
    tensors.extend([rna_expectation_tensor,mirna_expectation_tensor,meth_expectation_tensor])
    tensors.extend([rna_basic_expectation_tensor,mirna_basic_expectation_tensor,meth_basic_expectation_tensor])
  
    tensor_names = ["z_mu","z_var",\
                    "z_mu_rna","z_var_rna",\
                    "z_mu_mirna","z_var_mirna",\
                    "z_mu_meth","z_var_meth",\
                    "rna_expecation","mirna_expectation","meth_expectation",\
                    "rna_basic_expecation","mirna_basic_expectation","meth_basic_expectation"]
  
    assert len(tensor_names)==len(tensors), "should be same number"
    self.network.FillFeedDict( feed_dict, impute_dict )

    #pdb.set_trace()
    rna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[RNA]] == 1
    meth_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[METH]] == 1
    mirna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[miRNA]] == 1
        
    rna_expectation                  = np.zeros( (len(barcodes), self.dims_dict[RNA] ), dtype=float )
    rna_basic_expectation            = np.zeros( (len(barcodes), self.dims_dict[RNA] ), dtype=float )
    rna_loglikelihood                = np.zeros( (np.sum(rna_observed_query), self.dims_dict[RNA] ), dtype=float )
    rna_basic_loglikelihood          = np.zeros( (np.sum(rna_observed_query), self.dims_dict[RNA] ), dtype=float )
    meth_expectation                 = np.zeros( (len(barcodes), self.dims_dict[METH] ), dtype=float )
    meth_loglikelihood               = np.zeros( (np.sum(meth_observed_query), self.dims_dict[METH] ), dtype=float )
    meth_basic_expectation           = np.zeros( (len(barcodes), self.dims_dict[METH] ), dtype=float )
    meth_basic_loglikelihood         = np.zeros( (np.sum(meth_observed_query), self.dims_dict[METH] ), dtype=float )
    mirna_expectation                = np.zeros( (len(barcodes), self.dims_dict[miRNA] ), dtype=float )
    mirna_loglikelihood              = np.zeros( (np.sum(mirna_observed_query), self.dims_dict[miRNA] ), dtype=float )
    mirna_basic_expectation          = np.zeros( (len(barcodes), self.dims_dict[miRNA] ), dtype=float )
    mirna_basic_loglikelihood        = np.zeros( (np.sum(mirna_observed_query), self.dims_dict[miRNA] ), dtype=float )
        
      #drop_likelihoods = np.zeros( rna_dim )
    # dna_dim = self.dims_dict[DNA] #/self.n_dna_channels
    # dna_expectation = np.zeros( (len(barcodes),dna_dim), dtype=float )
    # dna_loglikelihood = np.zeros( (np.sum(dna_observed_query),dna_dim), dtype=float )
    
    nbr_splits = 50
    tensor2fill = []
    drop_factor = 1.0 #float(nbr_splits)/float(nbr_splits-1)
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
        tensor2fill.extend( [rna_expectation_tensor, rna_basic_expectation_tensor, loglikes_data_as_matrix["gen_rna_space"], loglikes_data_as_matrix["gen_rna_space_basic"] ] )
        rna_ids = [id_start,id_start+1,id_start+2,id_start+3]
        id_start+=4

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
        tensor2fill.extend( [mirna_expectation_tensor, mirna_basic_expectation_tensor, loglikes_data_as_matrix["gen_mirna_space"], loglikes_data_as_matrix["gen_mirna_space_basic"] ] )
        mirna_ids = [id_start,id_start+1,id_start+2,id_start+3]
        id_start+=4
             
      
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
        tensor2fill.extend( [meth_expectation_tensor, meth_basic_expectation_tensor, loglikes_data_as_matrix["gen_meth_space"], loglikes_data_as_matrix["gen_meth_space_basic"] ] )
        meth_ids = [id_start,id_start+1,id_start+2,id_start+3]
        id_start+=4
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
        rna_expectation[:,drop_rna_ids]         = tensor2fill_eval[rna_ids[0]][:,drop_rna_ids]
        rna_basic_expectation[:,drop_rna_ids]   = tensor2fill_eval[rna_ids[1]][:,drop_rna_ids]
        rna_loglikelihood[:,drop_rna_ids]       = tensor2fill_eval[rna_ids[2]][:,drop_rna_ids]
        rna_basic_loglikelihood[:,drop_rna_ids] = tensor2fill_eval[rna_ids[3]][:,drop_rna_ids]

      if use_mirna:
        mirna_expectation[:,drop_mirna_ids]         = tensor2fill_eval[mirna_ids[0]][:,drop_mirna_ids]
        mirna_basic_expectation[:,drop_mirna_ids]   = tensor2fill_eval[mirna_ids[1]][:,drop_mirna_ids]
        mirna_loglikelihood[:,drop_mirna_ids]       = tensor2fill_eval[mirna_ids[2]][:,drop_mirna_ids]
        mirna_basic_loglikelihood[:,drop_mirna_ids] = tensor2fill_eval[mirna_ids[3]][:,drop_mirna_ids]
      
      if use_dna:
        for idx,DNA_key in zip(range(len(self.DNA_keys)-1),self.DNA_keys[:-1]):
          dna_expectation[:,drop_dna_ids] = tensor2fill_eval[dna_ids[0]][:,drop_dna_ids]
          dna_loglikelihood[:,drop_dna_ids] = tensor2fill_eval[dna_ids[1]][:,drop_dna_ids]
        
      if use_meth:
        meth_expectation[:,drop_meth_ids]         = tensor2fill_eval[meth_ids[0]][:,drop_meth_ids]
        meth_basic_expectation[:,drop_meth_ids]   = tensor2fill_eval[meth_ids[1]][:,drop_meth_ids]
        meth_loglikelihood[:,drop_meth_ids]       = tensor2fill_eval[meth_ids[2]][:,drop_meth_ids]
        meth_basic_loglikelihood[:,drop_meth_ids] = tensor2fill_eval[meth_ids[3]][:,drop_meth_ids]
    
    #pdb.set_trace()   
    if use_rna:   
      self.WriteRunFillExpectation( epoch, RNA, barcodes, self.rna_genes, rna_observed_query, rna_expectation, self.data_store[self.RNA_key].loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, RNA, barcodes[rna_observed_query], self.rna_genes, rna_loglikelihood, mode )
      
      self.WriteRunFillExpectation( epoch, RNA+"_b", barcodes, self.rna_genes, rna_observed_query, rna_basic_expectation, self.data_store[self.RNA_key].loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, RNA+"_b", barcodes[rna_observed_query], self.rna_genes, rna_basic_loglikelihood, mode )

    if use_meth:
      self.WriteRunFillExpectation( epoch, METH, barcodes, self.meth_genes, meth_observed_query, meth_expectation, self.data_store[self.METH_key].loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, METH, barcodes[meth_observed_query], self.meth_genes, meth_loglikelihood, mode )

      self.WriteRunFillExpectation( epoch, METH+"_b", barcodes, self.meth_genes, meth_observed_query, meth_basic_expectation, self.data_store[self.METH_key].loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, METH+"_b", barcodes[meth_observed_query], self.meth_genes, meth_basic_loglikelihood, mode )


    if use_mirna:
      self.WriteRunFillLoglikelihood( epoch, miRNA, barcodes[mirna_observed_query], self.mirna_hsas, mirna_loglikelihood, mode )
      self.WriteRunFillExpectation( epoch, miRNA, barcodes, self.mirna_hsas, mirna_observed_query, mirna_expectation, self.data_store[self.miRNA_key].loc[ barcodes ].values, mode )

      self.WriteRunFillLoglikelihood( epoch, miRNA+"_b", barcodes[mirna_observed_query], self.mirna_hsas, mirna_basic_loglikelihood, mode )
      self.WriteRunFillExpectation( epoch, miRNA+"_b", barcodes, self.mirna_hsas, mirna_observed_query, mirna_basic_expectation, self.data_store[self.miRNA_key].loc[ barcodes ].values, mode )

    
    if use_dna:
      self.WriteRunFillExpectation( epoch, DNA, barcodes, self.dna_genes, dna_observed_query, dna_expectation, dna_data, mode )
      self.WriteRunFillLoglikelihood( epoch, DNA, barcodes[dna_observed_query], self.dna_genes, dna_loglikelihood, mode )
 
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
   
  def VizInputScales(self, sess, info_dict ):
    self.model_store.open()
    
    f = pp.figure()
    input_sources = ["METH","RNA","miRNA"]
    orders        = [self.meth_order,self.rna_order,self.mirna_order]
    data_means    = [self.meth_mean,self.rna_mean,self.mirna_mean]
    n_sources = len(input_sources)
    
    post_fix = "_scaled"
    idx=1
    for input_source in input_sources:
      # try and find...
      #pdb.set_trace()
      w_mean = self.model_store[ input_source + post_fix + "/W/w0"].values
      w_scale = self.model_store[ input_source + post_fix + "/W/w1"].values
      
      log_alpha = w_mean
      log_beta = w_scale
      
      alpha = np.exp( log_alpha )
      beta = np.exp( log_beta )
      
      mean = compute_mean( alpha, beta )
      variance = compute_variance( alpha, beta )
      
      n_dims, n_tissues = w_mean.shape
      
      precision = 1.0 / variance #np.exp( w_scale )
      std_dev = np.sqrt(variance+1e-12)
      
      ax = f.add_subplot( n_sources, 1, idx )
      
      w_i = orders[ idx-1 ] #np.arange( n_dims, dtype=int )
      w_0 = np.arange( n_dims, dtype=int )
      #pdb.set_trace()
      colors = "brgymcbrgymcbrgymcbrgymcbrgymcbrgymcbrgymcbrgkmcbrgymcbrgymcbrgymcbrgymcbrgymcbrgymcbrgymc"
      for t_idx in range(n_tissues):
        m = mean[:,t_idx][w_i]
        s = std_dev[:,t_idx][w_i]
        #pdb.set_trace()
        ax.fill_between( w_0, m - 0.5*s, m + 0.5*s, alpha=0.25, color=colors[t_idx] )
        ax.plot( w_0, m,colors[t_idx]+'-' )
      
      
      pp.plot( w_0, data_means[idx-1][w_i], 'k--', lw=2, alpha=0.5)
      pp.ylabel( input_source )
      pp.ylim(0,1)
      idx+=1
    
    #pp.show()
    #pdb.set_trace() 
    pp.savefig( self.viz_scaled_weights + ".png", fmt="png", bbox_inches = "tight")  
    self.model_store.close()
      
  def VizModel( self, sess, info_dict ): 
    print "** VIZ Model"
    
    #self.VizWeightsGeneric(sess, info_dict )
    self.VizInputScales(sess, info_dict )
    #self.model_store.open()
    #keys = self.model_store.keys()
    #print keys
    #pdb.set_trace()
    #self.model_store.close()

  def InitializeAnythingYouWant(self, sess, network ):
    print "Running : InitializeAnythingYouWant"
    input_sources = ["METH","RNA","miRNA"] 
    layers = ["gen_meth_space_basic","gen_rna_space_basic","gen_mirna_space_basic"]
    
    n_tissues = len(self.data_store[self.TISSUE_key].columns)
    #self.data_store[self.TISSUE_key].loc[ batch_barcodes ]
    
    # get log_alpha and log_beta values
    for layer_name, input_name in zip( layers, input_sources ):
      n_dims = self.dims_dict[ input_name ]
      
      alpha = np.zeros( (n_tissues, n_dims ), dtype = float )
      beta  = np.zeros( (n_tissues, n_dims ), dtype = float )
      
      for t_idx, tissue in zip( range( n_tissues), self.data_store[self.TISSUE_key].columns):
        
        n_samples = self.train_tissue[ tissue ].sum()
        alpha[t_idx,:] = self.tissue_statistics[ tissue ][ input_name ][ "alpha"]
        beta[t_idx,:] = self.tissue_statistics[ tissue ][ input_name ][ "beta"]
      
      log_alpha = np.log( alpha + 0.001 ).astype(np.float32)
      log_beta = np.log( beta + 0.001).astype(np.float32)
      
      #layer = network.GetLayer( layer_name )
      
      #sess.run( tf.assign(layer.weights[0][0], log_alpha) )
      #sess.run( tf.assign(layer.weights[1][0], log_beta) )
      network.GetLayer( layer_name ).SetWeights( sess, [log_alpha, log_beta ])
      #pdb.set_trace()
    