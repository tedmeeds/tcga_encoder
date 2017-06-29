#from tcga_encoder.models.vae.batcher_ABC import *
from tcga_encoder.models.vae.batcher_tissue_control import *
from sklearn.ensemble import RandomForestClassifier as rfc
from tcga_encoder.utils.models import GenerativeBinaryClassifier

def find_cohort( barcodes, cohorts ):
  found = np.zeros( (len(barcodes),1), dtype=bool )
  
  diseases = np.array( [s.split("_")[0] for s in barcodes], dtype = str)
  
  for cohort in cohorts:
    I = pp.find( diseases == cohort )
    found[I] = True
    
  return found
   
class TCGABatcherAdversarial( TCGABatcher ):
  def MakeVizFilenames(self):
    self.fill_z_input = True
    self.aucs_save = os.path.join( self.savedir, "dna_aucs.csv" )
    self.viz_filename_survival      =  os.path.join( self.savedir, "survival" )
    self.viz_filename_survival_lda  =  os.path.join( self.savedir, "survival__lda" )
    self.viz_filename_z_to_dna      =  os.path.join( self.savedir, "lda_dna" )
    self.viz_filename_z_rec_scatter          =  os.path.join( self.savedir, "z_rec_scatter.png" )
    self.viz_filename_z_rec_on_z_gen         =  os.path.join( self.savedir, "z_rec_on_z_gen.png" )
    self.viz_filename_rna_prediction_scatter =  os.path.join( self.savedir, "rna_prediction_scatter.png" )
    self.viz_filename_dna_batch_target       =  os.path.join( self.savedir, "dna_batch_target" )
    self.viz_filename_dna_batch_predict      =  os.path.join( self.savedir, "dna_batch_predict" )
    self.viz_filename_dna_aucs               =  os.path.join( self.savedir, "dna_aucs.png" )
    self.viz_filename_dna_aucs2               =  os.path.join( self.savedir, "dna_aucs_generative.png" )
    self.viz_filename_weights        =  os.path.join( self.savedir, "weights_" )
    self.viz_filename_lower_bound            =  os.path.join( self.savedir, "lower_bound.png" )
    self.viz_filename_log_pdf_sources        = os.path.join( self.savedir, "log_pdf_sources_z.png" )
    self.viz_filename_log_pdf_sources_per_gene = os.path.join( self.savedir, "log_pdf_batch.png" )
    self.viz_filename_log_pdf_sources_per_gene_fill = os.path.join( self.savedir, "log_pdf_fill.png" )
    self.viz_filename_error_sources_per_gene_fill = os.path.join( self.savedir, "errors_fill.png" )
    self.viz_filename_log_pdf_sources_per_gene_fill_all = os.path.join( self.savedir, "log_pdf_sources_z_per_gene_fill_all.png" )
    self.viz_filename_error_sources_per_gene_fill_all = os.path.join( self.savedir, "errors_sources_z_per_gene_fill_all.png" )
    self.viz_scaled_weights = os.path.join( self.savedir, "weights_scaled_inputs.png" )
    self.viz_tissue_predictions = os.path.join( self.savedir, "tissue_predictions" )
    self.viz_hidden_weights = os.path.join( self.savedir, "hidden_weights" )
    
  def FillDerivedPlaceholder( self, batch, layer_name, mode ):
    #pdb.set_trace()
    if layer_name == "Z_rec_input" and self.fill_z_input is True:
      #print "Getting Z for batch for ids ", batch["barcodes"][:5]
      
      self.fill_store.open()
      if mode == "BATCH" or mode == "TRAIN":
        # #pdb.set_trace()
        # try:
        #   batch_data_mu = self.fill_store["/Z/BATCH/Z/mu"].loc[ batch["barcodes"] ]
        #   batch_data_var = self.fill_store["/Z/BATCH/Z/var"].loc[ batch["barcodes"] ]
        # except:
        #   print "getting from train..."
        batch_data_mu = self.fill_store["/Z/TRAIN/Z/mu"].loc[ batch["barcodes"] ]
        batch_data_var = self.fill_store["/Z/TRAIN/Z/var"].loc[ batch["barcodes"] ]
        #pdb.set_trace()
        #batch_data_mu = self.fill_store["/Z/BATCH/Z/mu"].loc[ batch["barcodes"] ]
        
        #batch_data_var = self.fill_store["/Z/BATCH/Z/var"].loc[ batch["barcodes"] ]
        
        n,d = batch_data_mu.values.shape
        if mode == "BATCH":
          #pdb.set_trace()
          batch_data_values = batch_data_mu.values #+ np.sqrt(batch_data_var.values)*batch['u_z']
        else:
          batch_data_values = batch_data_mu.values
          
        #batch_data = self.fill_source_store["/Z/TRAIN/Z/mu"].loc[ batch["barcodes"] ]
        batch_data = pd.DataFrame(batch_data_values, index=batch_data_mu.index, columns=batch_data_mu.columns)
      else:
        #pdb.set_trace()
        batch_data = self.fill_store["/Z/VAL/Z/mu"].loc[ batch["barcodes"] ]
        
        
      nans = np.isnan( batch_data.values )
      batch_data_values = batch_data.values
      # if mode == "BATCH":
      #  batch_data_values = self.AddmiRnaNoise( batch_data.values, rate = 0.1 )
      #   
      # batch[ layer_name ] = self.NormalizemiRnaInput( batch_data_values )
      batch[ layer_name ] = batch_data
      batch[ layer_name ][nans] = 0
      #pdb.set_trace()
      self.fill_store.close()
    elif layer_name == "Z_rec_input":
      pdb.set_trace()
    elif layer_name == "DNA_target_mask_special":
      
      batch_barcodes = batch[ "barcodes" ]
      batch_observed = self.data_store[self.OBSERVED_key].loc[ batch_barcodes ].values
      cohort_observed = find_cohort( batch_barcodes, self.validation_tissues )
      if len(cohort_observed) != len(batch_observed):
        pdb.set_trace()
      batch_observed *= cohort_observed
      #
      
      batch[ layer_name ] = batch_observed[:,self.observed_source2idx[ DNA ]].astype(bool)
      self.dna_uses_cohorts = True
      #nbr_observed = batch_observed[:,self.observed_source2idx[ DNA ]].astype(bool).sum()
    elif layer_name == "dropped_source_observations":
      ids = [self.observed_source2idx[ source ] for source in self.arch_dict["input_sources"]]
      batch_data = batch[ "observed_sources" ][:,ids]  
      batch[ layer_name ]  = batch_data
      #pdb.set_trace()
      
  def InitializeAnythingYouWant(self, sess, network ):
    print "Running : InitializeAnythingYouWant"
    self.dna_aucs_all = []
    self.dna_aucs_all2 = []
    self.fill_z_input = True
    self.dna_uses_cohorts = False
    input_sources = ["METH","RNA","miRNA"] 
    input_datas = [self.METH_data, self.RNA_data, self.miRNA_data]
    layers = ["gen_meth_space_basic","gen_rna_space_basic","gen_mirna_space_basic"]
    
    n_tissues = len(self.data_store[self.TISSUE_key].columns)
    #self.data_store[self.TISSUE_key].loc[ batch_barcodes ]
    
    self.source2expectation_by_tissue = {}
    # get log_alpha and log_beta values
    for layer_name, input_name, input_data in zip( layers, input_sources, input_datas ):
      n_dims = self.dims_dict[ input_name ]
      
      if input_data == "FAIR":
        alpha = np.zeros( (n_tissues, n_dims ), dtype = float )
        beta  = np.zeros( (n_tissues, n_dims ), dtype = float )
      
        self.source2expectation_by_tissue[input_name] = 0.5*np.ones((n_tissues, n_dims ), dtype = float )
        for t_idx, tissue in zip( range( n_tissues), self.data_store[self.TISSUE_key].columns):
        
          n_samples = self.train_tissue[ tissue ].sum()
          alpha[t_idx,:] = self.tissue_statistics[ tissue ][ input_name ][ "alpha"]
          beta[t_idx,:] = self.tissue_statistics[ tissue ][ input_name ][ "beta"]
      
        log_alpha = np.log( alpha + 0.001 ).astype(np.float32)
        log_beta = np.log( beta + 0.001).astype(np.float32)
      
        self.source2expectation_by_tissue[input_name] = (alpha+0.001) / (alpha+beta+0.002)
        #layer = network.GetLayer( layer_name )
      
        #sess.run( tf.assign(layer.weights[0][0], log_alpha) )
        #sess.run( tf.assign(layer.weights[1][0], log_beta) )
        if network.HasLayer(layer_name ):
          #pdb.set_trace()
          network.GetLayer( layer_name ).SetWeights( sess, [log_alpha, log_beta ])
          
      elif input_data == "RSEM" or input_data == "READS" or input_data == "METH":      
          
        means     = np.zeros( (n_tissues, n_dims ), dtype = float )
        log_vars  = np.zeros( (n_tissues, n_dims ), dtype = float )
      
        self.source2expectation_by_tissue[input_name] = np.zeros((n_tissues, n_dims ), dtype = float )
        
        for t_idx, tissue in zip( range( n_tissues), self.data_store[self.TISSUE_key].columns):
        
          n_samples = self.train_tissue[ tissue ].sum()
          means[t_idx,:] = self.tissue_statistics[ tissue ][ input_name ][ "mean"]
          log_vars[t_idx,:] = np.log( self.tissue_statistics[ tissue ][ input_name ][ "var"] + 1e-12)
          
          # if tissue == "ucs":
          #   pdb.set_trace()
      
        self.source2expectation_by_tissue[input_name] = means #(alpha+0.001) / (alpha+beta+0.002)
        
        if network.HasLayer(layer_name ):
          #pdb.set_trace()
          network.GetLayer( layer_name ).SetWeights( sess, [means, log_vars ])

        
  def PreStepDoWhatYouWant( self, sess, epoch, network, cb_info, train_op ):
    train_ops = [train_op]
    
    if network.HasLayer( "Z_rec_input") or network.HasLayer( "Z_input"):
      # get the mean z space is wanted
      train_ops.extend( network.GetLayer( "rec_z_space" ).tensor )
      
    if network.HasLayer("gen_dna_space"):
      # get the predicted DNA
      train_ops.append( network.GetLayer("gen_dna_space").expectation )
    
    # self.fill_z_input = False
    # self.TestFillZ(sess,info_dict)
    #
    # self.BatchFillZ(sess,info_dict)
    # self.fill_z_input = True
    
    return train_ops
    
  def PostStepDoWhatYouWant( self, sess, epoch, network, cb_info, train_ops_evals ):
    network.GetLayer( "target_prediction_neg" ).SetWeights( sess, network.GetLayer( "target_prediction_pos" ).EvalWeights() )
    network.GetLayer( "target_prediction_neg" ).SetBiases( sess, network.GetLayer( "target_prediction_pos" ).EvalBiases() )
    
    self.fill_store.open()
    #pdb.set_trace()
    barcodes = cb_info[BATCH_FEED_IMPUTATION]["barcodes"]
    eval_idx = 1
    if network.HasLayer( "Z_rec_input") or network.HasLayer( "Z_input"):
      z_mu  = train_ops_evals[eval_idx]; eval_idx+=1
      z_var = train_ops_evals[eval_idx]; eval_idx+=1
      
      
      z_columns = ["z%d"%zidx for zidx in range(self.n_z)] 
      S = self.fill_store["/Z/TRAIN/Z/mu"]
      S.loc[ barcodes ] = z_mu 
      self.fill_store["/Z/TRAIN/Z/mu"] = S
      

      S = self.fill_store["/Z/TRAIN/Z/var"]
      S.loc[ barcodes ] = z_var
      self.fill_store["/Z/TRAIN/Z/var"] = S
    
      #pdb.set_trace()
      
    if network.HasLayer("gen_dna_space"):
      # get the predicted DNA
      p_of_dna = train_ops_evals[eval_idx]; eval_idx+=1
      S = self.fill_store["Fill/TRAIN/DNA"]
      S.loc[ barcodes ] = p_of_dna
      self.fill_store["Fill/TRAIN/DNA"] = S 
      #.loc[ barcodes ] = p_of_dna
    
    self.fill_store.close()
      
    
  # def DoWhatYouWantAtEpoch( self, sess, epoch, network, info_dict):
  #   pass
  #
  def InitFillStore(self):
    self.fill_store.open()
    z_columns = ["z%d"%zidx for zidx in range(self.n_z)]
    self.fill_store["Z/TRAIN/Z/mu"]  = pd.DataFrame( np.zeros( (len(self.train_barcodes),self.n_z)) , index = self.train_barcodes, columns = z_columns )
    self.fill_store["Z/TRAIN/Z/var"] = pd.DataFrame( np.ones( (len(self.train_barcodes),self.n_z) ), index = self.train_barcodes, columns = z_columns )
    self.fill_store["Z/VAL/Z/mu"]  = pd.DataFrame( np.zeros( (len(self.validation_barcodes),self.n_z)) , index = self.validation_barcodes, columns = z_columns )
    self.fill_store["Z/VAL/Z/var"] = pd.DataFrame( np.ones( (len(self.validation_barcodes),self.n_z) ), index = self.validation_barcodes, columns = z_columns )
    self.fill_store["Fill/TRAIN/DNA"] = pd.DataFrame( np.zeros( (len(self.train_barcodes),self.dna_dim) ), index = self.train_barcodes, columns = self.dna_genes )
    self.fill_store["Fill/VAL/DNA"] = pd.DataFrame( np.zeros( (len(self.validation_barcodes),self.dna_dim) ), index = self.validation_barcodes, columns = self.dna_genes )
    self.fill_store.close()

    
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
    #pdb.set_trace()
    log_p_source_z_values = batch_tensor_evals[4:]/n_batch
    
    #log_p_t_z_pos_values = batch_tensor_evals[6]/n_batch
    #log_p_t_z_neg_values = batch_tensor_evals[7]/n_batch
    
    #print np.sort(info_dict[BATCH_IDS])
    new_log_p_x_given_z = log_p_source_z_values.sum()
    lower_bound = log_p_z-log_q_z + new_log_p_x_given_z
    
    new_values = [epoch, lower_bound, new_log_p_x_given_z, log_p_z, log_q_z]
    new_values.extend( log_p_source_z_values )
    #new_values.extend( [log_p_t_z_pos_values,log_p_t_z_neg_values] )

    self.AddSeries(  self.epoch_store, epoch_key, values = new_values, columns = self.network.batch_log_columns )
    
    epoch_log_p_source_z_values = [epoch]
    epoch_log_p_source_z_values.extend( log_p_source_z_values )
    epoch_source_columns = ['Epoch']
    epoch_source_columns.extend(self.arch_dict[TARGET_SOURCES])
    epoch_source_columns[-4] = "T+"
    epoch_source_columns[-3] = "T-"
    epoch_source_columns[-2] = "acc T+"
    epoch_source_columns[-1] = "acc T-"
    if mode == "BATCH":
      #pdb.set_trace()
      self.AddSeries(  self.epoch_store, BATCH_SOURCE_LOGPDF, values = epoch_log_p_source_z_values, columns = epoch_source_columns )
      self.PrintRow( self.epoch_store, epoch_key )
    elif mode == "TEST" and self.n_test>0:
      self.AddSeries(  self.epoch_store, TEST_SOURCE_LOGPDF, values = epoch_log_p_source_z_values, columns = epoch_source_columns )
      self.PrintRow( self.epoch_store, epoch_key )
    elif mode == "VAL" and self.n_val>0:
      self.AddSeries(  self.epoch_store, VAL_SOURCE_LOGPDF, values = epoch_log_p_source_z_values, columns = epoch_source_columns )
      self.PrintRow( self.epoch_store, epoch_key )
    
    # if mode == "TEST" or mode == "VAL":
    #   #print "!!!!!!!!"
    #   #print "testing product model"
    #   input_observations = impute_dict[INPUT_OBSERVATIONS]
    #
    #   tensors = []
    #   tensors.extend( self.network.GetTensor("rec_z_space_rna") )
    #   #tensors.extend( self.network.GetTensor("rec_z_space_dna") )
    #   tensors.extend( self.network.GetTensor("rec_z_space_meth") )
    #   tensors.extend( self.network.GetTensor("rec_z_space") )
    #
    #   if self.network is tcga_encoder.models.networks.ConditionalVariationalAutoEncoder:
    #     tensors.extend( self.network.GetTensor("gen_z_space") )
    #
    #   tensor_evals = sess.run( tensors, feed_dict = feed_dict )
    #
    #   rna_mean  = tensor_evals[0]
    #   rna_var   = tensor_evals[1]
    #   #dna_mean  = tensor_evals[2]
    #   #dna_var   = tensor_evals[3]
    #   meth_mean = tensor_evals[2]
    #   meth_var  = tensor_evals[3]
    #   z_mean    = tensor_evals[4]
    #   z_var     = tensor_evals[5]
    #   if self.network is tcga_encoder.models.networks.ConditionalVariationalAutoEncoder:
    #     z_mean_g  = tensor_evals[6]
    #     z_var_g   = tensor_evals[7]
    #
    #   self.fill_store.open()
    #   self.fill_store[ "%s/Z/%s/mu"%(mode,RNA)]   = pd.DataFrame( rna_mean, index = barcodes, columns = self.z_columns )
    #   self.fill_store[ "%s/Z/%s/var"%(mode,RNA)]  = pd.DataFrame( rna_var, index = barcodes, columns = self.z_columns )
    #   #self.fill_store[ "%s/Z/%s/mu"%(mode,DNA)]   = pd.DataFrame( dna_mean, index = barcodes, columns = self.z_columns )
    #   #self.fill_store[ "%s/Z/%s/var"%(mode,DNA)]  = pd.DataFrame( dna_var, index = barcodes, columns = self.z_columns )
    #   self.fill_store[ "%s/Z/%s/mu"%(mode,METH)]  = pd.DataFrame( meth_mean, index = barcodes, columns = self.z_columns )
    #   self.fill_store[ "%s/Z/%s/var"%(mode,METH)] = pd.DataFrame( meth_var, index = barcodes, columns = self.z_columns )
    #   self.fill_store[ "%s/Z/rec/mu"%mode]        = pd.DataFrame( z_mean, index = barcodes, columns = self.z_columns )
    #   self.fill_store[ "%s/Z/rec/var"%mode]       = pd.DataFrame( z_var, index = barcodes, columns = self.z_columns )
    #   if self.network is tcga_encoder.models.networks.ConditionalVariationalAutoEncoder:
    #     self.fill_store[ "%s/Z/gen/mu"%mode]        = pd.DataFrame( z_mean_g, index = barcodes, columns = self.z_columns )
    #     self.fill_store[ "%s/Z/gen/var"%mode]       = pd.DataFrame( z_var_g, index = barcodes, columns = self.z_columns )
    #   self.fill_store.close()
                
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
               label="Batch (%0.4f)"%(self.epoch_store[BATCH_SOURCE_LOGPDF][target_source].values[-1]/self.dims_dict[target_source]) )
      if prior_source is not None:
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
        if prior_source is not None:
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
        if prior_source is not None:
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
      if prior_source is not None:
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
        if prior_source is not None:
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
        
        if prior_source is not None:
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
        if prior_source is not None:
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
                 
        if prior_source is not None:
          query = query2#&query2
          df = self.epoch_store[VAL_FILL_ERROR][query]
          epochs = df["Epoch"].values
          loglik = df["Error"].values
          
          if len(loglik) == 0:
            continue
          # pp.plot( epochs, loglik, 'v-', \
          #        color=self.source2mediumcolor[prior_source],\
          #        mec=self.source2darkcolor[prior_source], mew=1, \
          #        mfc=self.source2lightcolor[prior_source], lw=2, \
          #        ms = 8, \
          #        label="Val prior (%0.6f)"%(loglik[-1]) )
        
      
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

    if self.network.HasLayer("gen_dna_space"):
      main_sources.append( DNA )
      prior_sources.append(None)
      
    self.epoch_store.open()
  
    self.PlotLowerBound()

    self.PlotLogPdf(main_sources,prior_sources)
    
    self.PlotFillLogPdf(main_sources,prior_sources)
    
    self.PlotFillError(main_sources,prior_sources)
    
    self.PlotTissuePrediction(sess, info_dict)
    
    if self.network.HasLayer("gen_dna_space"):
      self.DnaGenerativeModel( sess, info_dict )
      self.PrintAndPlotDnaPredictions(sess, info_dict)
      
    self.epoch_store.close()
    pp.close('all')
  
  
  def DnaGenerativeModel( self, sess, info_dict ):
    self.fill_store.open()
    f=pp.figure()
    ax=f.add_subplot(111)
    ax.plot( [0.0,1.0], [0,1], 'k--')
    
    #self.TrainFillDna(sess, info_dict)
    
    dna_observed_query = self.data_store["/CLINICAL/observed"][DNA] == 1
    dna_observed_train = dna_observed_query.loc[self.train_barcodes]
    dna_observed_val   = dna_observed_query.loc[self.validation_barcodes]
    
    dna_observed_train_barcodes = dna_observed_train[dna_observed_train.values].index.values

    
    dna_observed_val_barcodes   = dna_observed_val[dna_observed_val.values].index.values
    val_cohorts = np.unique( np.array( [s.split("_")[0] for s in dna_observed_val_barcodes] ) )
    
    train_cohorts = np.array( [s.split("_")[0] for s in dna_observed_train_barcodes] )
    #if self.dna_uses_cohorts is True:
    new_cohorts = []
    for cohort in val_cohorts:
      I = pp.find( train_cohorts==cohort )
      new_cohorts.extend( dna_observed_train_barcodes[I] )
    dna_observed_train_barcodes = np.array(new_cohorts)
      
    #pdb.set_trace()
    # val_predictions   = self.fill_store["/Fill/VAL/DNA"].loc[dna_observed_val_barcodes]
    # train_predictions = self.fill_store["/Fill/TRAIN/DNA"].loc[dna_observed_train_barcodes]
    
    train_z = self.fill_store["/Z/TRAIN/Z/mu"].loc[dna_observed_train_barcodes]
    val_z   = self.fill_store["/Z/VAL/Z/mu"].loc[dna_observed_val_barcodes]
    
    val_targets   = self.dna_store.loc[dna_observed_val_barcodes]
    train_targets = self.dna_store.loc[dna_observed_train_barcodes]
    
    n_train = len(train_targets)
    n_val   = len(val_targets)
    aucs = []
    groups1 = []
    groups0 = []
    val_weighted_auc = 0.0
    val_weights = 0.0
    train_weighted_auc = 0.0
    train_weights = 0.0
  
    ok_val = []
    set_train_2_val = False
    train_predictions = []
    val_predictions = []
    for dna_gene in self.dna_genes:
      train_auc_rfc = 1.0
      train_cnt = np.sum(train_targets[dna_gene].values)
      n_train = len(train_targets[dna_gene].values)
      
      val_auc = 1.0
      val_auc_rfc = 1.0
      n_val = len(val_targets[dna_gene].values) 
      val_cnt = np.sum(val_targets[dna_gene].values)
      
      if train_cnt>0 and train_cnt < n_train:
        
        model = GenerativeBinaryClassifier( prior_std = 1.0 )
        model.fit( train_z.values, train_targets[dna_gene].values )
        train_prediction = model.predict( train_z.values )
        train_predictions.append(train_prediction)
        train_auc = roc_auc_score( train_targets[dna_gene].values, train_prediction )
        train_weighted_auc += train_cnt*train_auc
        train_weights += train_cnt
        
        if val_cnt>0 and val_cnt<n_val:
          ok_val.append(True)
          val_prediction = model.predict( val_z.values )
          val_predictions.append(val_prediction)
          val_auc = roc_auc_score( val_targets[dna_gene].values, val_prediction )
          val_auc_fpr, val_auc_tpr, thresholds = roc_curve( val_targets[dna_gene].values, val_prediction )
          val_weighted_auc += val_cnt*val_auc
          val_weights += val_cnt
          val_auc_rfc = 0#roc_auc_score( val_targets[dna_gene].values, val_rfc )
          if val_cnt>20:
            ax.plot( val_auc_fpr, val_auc_tpr, "k-", lw=1, alpha=0.5, label = "Val %s"%(dna_gene) )
          groups1.append(dna_gene)
        else:
          groups0.append(dna_gene)
          ok_val.append(False)
        if set_train_2_val is True:
          train_auc = val_auc
          train_cnt = val_cnt
          n_train = n_val
        #pdb.set_trace()
        if n_val>0:
          aucs.append([train_auc,val_auc, 1+1000*float(train_cnt)/n_train,1+1000*float(val_cnt)/n_val])
        else:
          aucs.append([train_auc,val_auc, 1+1000*float(train_cnt),1+1000*float(val_cnt)])
      else:
        set_train_2_val = True
      
    if len(val_predictions)==0:
      self.fill_store.close()
      return  
    aucs = np.array(aucs)
    train_predictions = np.array(train_predictions).T
    val_predictions = np.array(val_predictions).T
    
    val_predictions = pd.DataFrame( val_predictions, columns = self.dna_genes )
    train_predictions = pd.DataFrame( train_predictions, columns = self.dna_genes )
    ok_val.append(False)
    ok_val = np.array(ok_val)
    self.dna_aucs2 = pd.DataFrame( aucs.T, index = ["Train","Val","Frequency","Frequency2"], columns = self.dna_genes )
    
    #pdb.set_trace()
    val_auc = roc_auc_score( val_targets.values.flatten(), val_predictions.values.flatten() )
    val_weighted_auc /= val_weights
    val_auc_fpr, val_auc_tpr, thresholds = roc_curve( val_targets.values.flatten(), val_predictions.values.flatten() )
    I_val   = np.argsort( -val_predictions.values.flatten() )
    
    if set_train_2_val is True:
      train_auc = val_auc
      train_weighted_auc=val_weighted_auc
      
      train_weights=val_weights
      train_auc_fpr=val_auc_fpr
      train_auc_tpr=val_auc_tpr
      tr_auc_fpr=val_auc_fpr
      tr_auc_tpr=val_auc_tpr
      I_train=I_val
    else:
      train_auc = roc_auc_score( train_targets.values.flatten(), train_predictions.values.flatten() )
      train_weighted_auc /= train_weights
      tr_auc_fpr, tr_auc_tpr, thresholds = roc_curve( train_targets.values.flatten(), train_predictions.values.flatten() )
      I_train = np.argsort( -train_predictions.values.flatten() )
    
    self.dna_aucs2["ALL"] = pd.Series( [train_auc,val_auc, 1000.0,1000.0], index = ["Train","Val","Frequency","Frequency2"])  
    print self.dna_aucs2.T
    
    mean_aucs = self.dna_aucs2.T[ok_val].T.mean(1)
    self.dna_aucs_all2.append( [train_auc,val_auc, mean_aucs.loc["Train"], mean_aucs.loc["Val"], train_weighted_auc, val_weighted_auc]  )
    self.dna_aucs2[groups1].T.plot(ax=ax, kind='scatter', x='Train', y='Val', marker="o", color='White', s=self.dna_aucs2[groups1].T["Frequency2"].values, alpha=0.75, edgecolors='k')
        
    ax.plot( val_auc_fpr, val_auc_tpr, "r-", label = "Val ROC" )
    ax.plot( tr_auc_fpr, tr_auc_tpr, "b-", label = "Train ROC" )
    
    if self.data_dict.has_key("highlight_genes"):
      highlight_genes = self.data_dict[ "highlight_genes"]
      X = self.dna_aucs2[highlight_genes]
      X.T.plot(ax=ax, kind='scatter', x='Train', y='Val', marker="o", color='Yellow', s=self.dna_aucs2[highlight_genes].T["Frequency2"].values, alpha=0.75, edgecolors='k', linewidths=1)
      for x,y,dna_gene in zip( X.T.values[:,0], X.T.values[:,1], highlight_genes ):
        ax.text( x,y,dna_gene, fontsize=8 )
        val_auc_fpr, val_auc_tpr, thresholds = roc_curve( val_targets[dna_gene].values, val_predictions[dna_gene].values )
        ax.plot( val_auc_fpr, val_auc_tpr, "y-", lw=2, alpha=0.95, label = "Val %s"%(dna_gene) )
    
    X = np.array( self.dna_aucs_all2)
    ax.plot( X[:,0], X[:,1], 'r.-', alpha=0.75  )
    ax.plot( X[:,2], X[:,3], 'g.-', alpha=0.75  )
    ax.plot( X[:,4], X[:,5], 'm.-', alpha=0.75  )
    self.dna_aucs2[["ALL"]].T.plot(ax=ax, kind='scatter', x='Train', y='Val', marker="o", color='Red', s=self.dna_aucs2[["ALL"]].T["Frequency2"].values, alpha=0.25, edgecolors='k')
    ax.plot( X[-1,2], X[-1,3],  'go', ms=30,alpha=0.25,mec='k')
    ax.plot( X[-1,4], X[-1,5],  'mo', ms=30,alpha=0.25,mec='k')
    pp.xlim(0.0,1)
    pp.ylim(0.0,1)
    f.savefig( self.viz_filename_dna_aucs2, dpi=300,  )
    
    self.fill_store.close()
    
  def  PrintAndPlotDnaPredictions(self,sess, info_dict):
    self.fill_store.open()
    f=pp.figure()
    ax=f.add_subplot(111)
    ax.plot( [0.0,1.0], [0,1], 'k--')
    
    #self.TrainFillDna(sess, info_dict)
    
    dna_observed_query = self.data_store["/CLINICAL/observed"][DNA] == 1
    dna_observed_train = dna_observed_query.loc[self.train_barcodes]
    dna_observed_val   = dna_observed_query.loc[self.validation_barcodes]
    
    dna_observed_train_barcodes = dna_observed_train[dna_observed_train.values].index.values

    
    dna_observed_val_barcodes   = dna_observed_val[dna_observed_val.values].index.values
    val_cohorts = np.unique( np.array( [s.split("_")[0] for s in dna_observed_val_barcodes] ) )
    
    train_cohorts = np.array( [s.split("_")[0] for s in dna_observed_train_barcodes] )
    #if self.dna_uses_cohorts is True:
    new_cohorts = []
    for cohort in val_cohorts:
      I = pp.find( train_cohorts==cohort )
      new_cohorts.extend( dna_observed_train_barcodes[I] )
    dna_observed_train_barcodes = np.array(new_cohorts)
      
    #pdb.set_trace()
    val_predictions   = self.fill_store["/Fill/VAL/DNA"].loc[dna_observed_val_barcodes]
    train_predictions = self.fill_store["/Fill/TRAIN/DNA"].loc[dna_observed_train_barcodes]
    
    train_z = self.fill_store["/Z/TRAIN/Z/mu"].loc[dna_observed_train_barcodes]
    val_z   = self.fill_store["/Z/VAL/Z/mu"].loc[dna_observed_val_barcodes]
    
    val_targets   = self.dna_store.loc[dna_observed_val_barcodes]
    train_targets = self.dna_store.loc[dna_observed_train_barcodes]
    
    #pdb.set_trace()
    #pdb.set_trace()
    n_train = len(train_targets)
    n_val   = len(val_targets)
    aucs = []
    groups1 = []
    groups0 = []
    val_weighted_auc = 0.0
    val_weights = 0.0
    train_weighted_auc = 0.0
    train_weights = 0.0
    print "training random forests"
    ok_val = []
    set_train_2_val = False
    for dna_gene in self.dna_genes:
      train_auc_rfc = 1.0
      train_cnt = np.sum(train_targets[dna_gene].values)
      n_train = len(train_targets[dna_gene].values)
      if train_cnt>0 and train_cnt < n_train:
        train_auc = roc_auc_score( train_targets[dna_gene].values, train_predictions[dna_gene].values )
        train_weighted_auc += train_cnt*train_auc
        train_weights += train_cnt
        
        #M = rfc(n_estimators=1, max_depth=3, class_weight="balanced", bootstrap=True)
        
        #M.fit( train_z,  train_targets[dna_gene].values )
        train_rfc = 0#M.predict_proba(train_z)[:,1]
        val_rfc = 0#M.predict_proba(val_z)[:,1]
        train_auc_rfc = 0#roc_auc_score( train_targets[dna_gene].values, train_rfc )
        #pdb.set_trace()
      else:
        set_train_2_val = True
      
      val_auc = 1.0
      val_auc_rfc = 1.0
      n_val = len(val_targets[dna_gene].values) 
      val_cnt = np.sum(val_targets[dna_gene].values)
      if val_cnt>0 and val_cnt<n_val:
        ok_val.append(True)
        
        val_auc = roc_auc_score( val_targets[dna_gene].values, val_predictions[dna_gene].values )
        val_auc_fpr, val_auc_tpr, thresholds = roc_curve( val_targets[dna_gene].values, val_predictions[dna_gene].values )
        val_weighted_auc += val_cnt*val_auc
        val_weights += val_cnt
        val_auc_rfc = 0#roc_auc_score( val_targets[dna_gene].values, val_rfc )
        if val_cnt>20:
          ax.plot( val_auc_fpr, val_auc_tpr, "k-", lw=1, alpha=0.5, label = "Val %s"%(dna_gene) )
        groups1.append(dna_gene)
      else:
        groups0.append(dna_gene)
        ok_val.append(False)
      if set_train_2_val is True:
        train_auc = val_auc
        train_cnt = val_cnt
        n_train = n_val
        train_auc_rfc = val_auc_rfc
      #pdb.set_trace()
      if n_val>0:
        aucs.append([train_auc,val_auc, 1+1000*float(train_cnt)/n_train,1+1000*float(val_cnt)/n_val, train_auc_rfc, val_auc_rfc])
      else:
        aucs.append([train_auc,val_auc, 1+1000*float(train_cnt),1+1000*float(val_cnt), train_auc_rfc, val_auc_rfc])
      
    aucs = np.array(aucs)
    
    ok_val.append(False)
    ok_val = np.array(ok_val)
    self.dna_aucs = pd.DataFrame( aucs.T, index = ["Train","Val","Frequency","Frequency2","Train-rfc","Val-rfc"], columns = self.dna_genes )
    
    val_auc = roc_auc_score( val_targets.values.flatten(), val_predictions.values.flatten() )
    val_weighted_auc /= val_weights
    val_auc_fpr, val_auc_tpr, thresholds = roc_curve( val_targets.values.flatten(), val_predictions.values.flatten() )
    I_val   = np.argsort( -val_predictions.values.flatten() )
    
    if set_train_2_val is True:
      train_auc = val_auc
      train_weighted_auc=val_weighted_auc
      
      train_weights=val_weights
      train_auc_fpr=val_auc_fpr
      train_auc_tpr=val_auc_tpr
      tr_auc_fpr=val_auc_fpr
      tr_auc_tpr=val_auc_tpr
      I_train=I_val
    else:
      train_auc = roc_auc_score( train_targets.values.flatten(), train_predictions.values.flatten() )
      train_weighted_auc /= train_weights
      tr_auc_fpr, tr_auc_tpr, thresholds = roc_curve( train_targets.values.flatten(), train_predictions.values.flatten() )
      I_train = np.argsort( -train_predictions.values.flatten() )
    
    self.dna_aucs["ALL"] = pd.Series( [train_auc,val_auc, 1000.0,1000.0,0,0], index = ["Train","Val","Frequency","Frequency2","Train-rfc","Val-rfc"])  
    print self.dna_aucs.T
    #pdb.set_trace()
    
    mean_aucs = self.dna_aucs.T[ok_val].T.mean(1)
    #pdb.set_trace()
    self.dna_aucs_all.append( [train_auc,val_auc, mean_aucs.loc["Train"], mean_aucs.loc["Val"], train_weighted_auc, val_weighted_auc, mean_aucs.loc["Train-rfc"], mean_aucs.loc["Val-rfc"]]  )
    #pdb.set_trace()
    self.dna_aucs[groups1].T.plot(ax=ax, kind='scatter', x='Train', y='Val', marker="o", color='White', s=self.dna_aucs[groups1].T["Frequency2"].values, alpha=0.75, edgecolors='k')
    #self.dna_aucs[groups0].T.plot(ax=ax, kind='scatter', x='Train', y='Val', marker="s", color='Green',s=self.dna_aucs[groups0].T["Frequency2"].values, alpha=0.75, edgecolors='k')
    
    ax.plot( val_auc_fpr, val_auc_tpr, "r-", label = "Val ROC" )
    ax.plot( tr_auc_fpr, tr_auc_tpr, "b-", label = "Train ROC" )
    
    if self.data_dict.has_key("highlight_genes"):
      highlight_genes = self.data_dict[ "highlight_genes"]
      X = self.dna_aucs[highlight_genes]
      X.T.plot(ax=ax, kind='scatter', x='Train', y='Val', marker="o", color='Yellow', s=self.dna_aucs[highlight_genes].T["Frequency2"].values, alpha=0.75, edgecolors='k', linewidths=1)
      #pdb.set_trace()
      for x,y,dna_gene in zip( X.T.values[:,0], X.T.values[:,1], highlight_genes ):
        ax.text( x,y,dna_gene, fontsize=8 )
        val_auc_fpr, val_auc_tpr, thresholds = roc_curve( val_targets[dna_gene].values, val_predictions[dna_gene].values )
        ax.plot( val_auc_fpr, val_auc_tpr, "y-", lw=2, alpha=0.95, label = "Val %s"%(dna_gene) )
    
    X = np.array( self.dna_aucs_all)
    ax.plot( X[:,0], X[:,1], 'r.-', alpha=0.75  )
    ax.plot( X[:,2], X[:,3], 'g.-', alpha=0.75  )
    ax.plot( X[:,4], X[:,5], 'm.-', alpha=0.75  )
    #ax.plot( X[:,6], X[:,7], 'c.-', alpha=0.75  )
    self.dna_aucs[["ALL"]].T.plot(ax=ax, kind='scatter', x='Train', y='Val', marker="o", color='Red', s=self.dna_aucs[["ALL"]].T["Frequency2"].values, alpha=0.25, edgecolors='k')
    ax.plot( X[-1,2], X[-1,3],  'go', ms=30,alpha=0.25,mec='k')
    ax.plot( X[-1,4], X[-1,5],  'mo', ms=30,alpha=0.25,mec='k')
    #ax.plot( X[-1,6], X[-1,7],  'co', ms=30,alpha=0.25,mec='k')
    #pp.plot( [self.dna_aucs["ALL"].values[0]], [self.dna_aucs["ALL"].values[1]], 'ro' )
    pp.xlim(0.0,1)
    pp.ylim(0.0,1)
    f.savefig( self.viz_filename_dna_aucs, dpi=300,  )
    
    # f2 = pp.figure()
    # ax2 = f2.add_subplot(111)
    # ax2.semilogy( np.linspace(0,1,len(I_val)), val_predictions.values.flatten()[I_val], 'r', alpha=0.5 )
    # ax2.semilogy( np.linspace(0,1,len(I_val)), val_targets.values.flatten()[I_val][: : -1].cumsum()[: : -1]/val_weights, 'r--' )
    # ax2.semilogy( np.linspace(0,1,len(I_train)), train_predictions.values.flatten()[I_train], 'b', alpha=0.5 )
    # ax2.semilogy( np.linspace(0,1,len(I_train)), train_targets.values.flatten()[I_train][: : -1].cumsum()[: : -1]/train_weights, 'b--' )
    # f2.savefig( self.viz_filename_dna_aucs+"_2.png", dpi=300,  )
    # pp.close('all')
    # self.dna_aucs.T.to_csv(self.aucs_save,sep=",")
    self.fill_store.close()
    
  def PlotTissuePrediction(self,sess, info_dict):
    self.fill_store.open()
    mode = "VAL"
    target_pos = "TISSUE+"
    target_neg = "TISSUE-"
    target_pos_no_bias = TISSUE+"no_bias+"
    target_neg_no_bias = TISSUE+"no_bias-"
    s_pos = "/Fill/%s/%s/"%(mode,target_pos )
    s_neg = "/Fill/%s/%s/"%(mode,target_neg )
    s_pos_no_bias = "/Fill/%s/%s/"%(mode,target_pos_no_bias )
    s_neg_no_bias = "/Fill/%s/%s/"%(mode,target_neg_no_bias )
    
    pos_pred         = self.fill_store[s_pos]
    pos_pred_no_bias = self.fill_store[s_pos_no_bias]
    neg_pred         = self.fill_store[s_neg]
    neg_pred_no_bias     = self.fill_store[s_neg_no_bias]
    
    #pdb.set_trace()
    pos_predicted = np.argmax( pos_pred.values, 1 )
    neg_predicted = np.argmax( neg_pred.values, 1 )
    pos_predicted_no_bias = np.argmax( pos_pred_no_bias.values, 1 )
    neg_predicted_no_bias = np.argmax( neg_pred_no_bias.values, 1 )
    
    pos_predictions = np.zeros(pos_pred.shape,dtype=int)
    neg_predictions = np.zeros(neg_pred.shape,dtype=int)
    
    pos_predictions_no_bias = np.zeros(pos_pred.shape,dtype=int)
    neg_predictions_no_bias = np.zeros(neg_pred.shape,dtype=int)
    
    for idx in range(len(pos_predicted)):
      pos_predictions[idx, pos_predicted[idx]] = 1
      neg_predictions[idx, neg_predicted[idx]] = 1
      pos_predictions_no_bias[idx, pos_predicted_no_bias[idx]] = 1
      neg_predictions_no_bias[idx, neg_predicted_no_bias[idx]] = 1
      
    data = self.data_store[self.TISSUE_key].loc[pos_pred.index]
    
    order = np.argsort( np.argmax(data.values,1) )
    
    f = pp.figure( figsize=(14,6))
    
    ax_data=f.add_subplot(2,5,1)
    ax_pos             =f.add_subplot(2,5,2)
    ax_pos_pred        =f.add_subplot(2,5,3)
    ax_pos_no_bias     =f.add_subplot(2,5,4)
    ax_pos_pred_no_bias=f.add_subplot(2,5,5)

    ax_data2=f.add_subplot(2,5,6)
    ax_pos2             =f.add_subplot(2,5,7)
    ax_pos_pred2        =f.add_subplot(2,5,8)
    ax_pos_no_bias2     =f.add_subplot(2,5,9)
    ax_pos_pred_no_bias2=f.add_subplot(2,5,10)
        
    #ax_neg             =f.add_subplot(3,4,9)
    #ax_neg_pred        =f.add_subplot(3,4,10)
    #ax_neg_no_bias     =f.add_subplot(3,4,11)
    #ax_neg_pred_no_bias=f.add_subplot(3,4,12)
    
    ax_data.imshow(data, aspect='auto',interpolation='nearest',cmap='hot')
    ax_data2.plot(data.values.mean(0))
    ax_data.grid('off')
    
    if len(pos_pred) > 0:
      ax_pos.imshow(pos_pred, aspect='auto',interpolation='nearest',cmap='hot')
      ax_pos.grid('off')
      #pdb.set_trace()
    
    
      ax_pos2.plot(pos_pred.values.T, 'k.-', lw=0.5, alpha=0.5 )
      ax_pos2.plot(pos_pred.values.mean(0))
    
      #ax_neg.imshow(neg_pred, aspect='auto',interpolation='nearest')
      ax_pos_pred.imshow(pos_predictions, aspect='auto',interpolation='nearest',cmap='hot')
      ax_pos_pred.grid('off')
      ax_pos_pred2.plot(pos_predictions.mean(0))
      #ax_neg_pred.imshow(neg_predictions, aspect='auto',interpolation='nearest')
    
      ax_pos_no_bias.imshow(pos_pred_no_bias, aspect='auto',interpolation='nearest',cmap='hot')
      ax_pos_no_bias.grid('off')
      ax_pos_no_bias2.plot(pos_pred_no_bias.values.T, 'k.-', lw=0.5, alpha=0.5 )
      ax_pos_no_bias2.plot(pos_pred_no_bias.values.mean(0))
    
      #ax_neg_no_bias.imshow(neg_pred_no_bias, aspect='auto',interpolation='nearest')
      ax_pos_pred_no_bias.imshow(pos_predictions_no_bias, aspect='auto',interpolation='nearest',cmap='hot')
      ax_pos_pred_no_bias2.plot(pos_predictions_no_bias.mean(0))
      #ax_neg_pred_no_bias.imshow(neg_predictions_no_bias, aspect='auto',interpolation='nearest')
      ax_pos_pred_no_bias.grid('off')
    
    
      #f.savefig( self.viz_tissue_predictions + "_%d.png"%(info_dict["epoch"]))
    f.savefig( self.viz_tissue_predictions + ".png")
    pp.close()
    #pdb.set_trace()
    
      
  def RunFill2( self, epoch, sess, feed_dict, impute_dict, mode ):
    print "COMPUTE Z-SPACE"
    use_dna = False
    use_rna = True
    use_meth = True
    use_mirna = True
    use_tissue = True
    use_z = True
    
    if self.network.HasLayer( "gen_dna_space"):
      use_dna = True
    
    
    barcodes = impute_dict[BARCODES]
    batch = self.FillBatch( impute_dict[BARCODES], mode )
    #not_observed = np.setdiff1d( self.input_sources, inputs2use )
        
    rec_z_space_tensors       = self.network.GetTensor( "rec_z_space" )
    if self.network.HasLayer( "rec_z_space_rna" ):
      rna_rec_z_space_tensors   = self.network.GetTensor( "rec_z_space_rna" )
      mirna_rec_z_space_tensors   = self.network.GetTensor( "rec_z_space_mirna" )
      meth_rec_z_space_tensors  = self.network.GetTensor( "rec_z_space_meth" )
    
  
    rna_expectation_tensor = self.network.GetLayer( "gen_rna_space" ).expectation
    mirna_expectation_tensor = self.network.GetLayer( "gen_mirna_space" ).expectation
    meth_expectation_tensor = self.network.GetLayer( "gen_meth_space" ).expectation
    
    if self.network.HasLayer("gen_rna_space_basic"):
      rna_basic_expectation_tensor = self.network.GetLayer( "gen_rna_space_basic" ).expectation
      mirna_basic_expectation_tensor = self.network.GetLayer( "gen_mirna_space_basic" ).expectation
      meth_basic_expectation_tensor = self.network.GetLayer( "gen_meth_space_basic" ).expectation
    
    positive_tissue_prediction_tensor = self.network.GetLayer( "target_prediction_pos" ).expectation
    negative_tissue_prediction_tensor = self.network.GetLayer( "target_prediction_neg" ).expectation
    no_bias_positive_tissue_prediction_tensor = self.network.GetLayer( "target_prediction_pos" ).expectation_no_bias
    no_bias_negative_tissue_prediction_tensor = self.network.GetLayer( "target_prediction_neg" ).expectation_no_bias
    
    if use_dna:
      dna_expectation_tensor = self.network.GetLayer( "gen_dna_space" ).expectation
      dna_data = np.minimum( 1.0, self.dna_store.loc[ barcodes ].fillna( 0 ).values )
      dna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[DNA]] == 1
      
    loglikes_data_as_matrix = self.network.loglikes_data_as_matrix
  
    tensors = []; tensor_names=[]
    tensors.extend(rec_z_space_tensors); tensor_names.extend(["z_mu","z_var"])
    
    if self.network.HasLayer( "rec_z_space_rna" ):
      tensors.extend(rna_rec_z_space_tensors); tensor_names.extend(["z_mu_rna","z_var_rna"])
      tensors.extend(mirna_rec_z_space_tensors); tensor_names.extend(["z_mu_mirna","z_var_mirna"])
      tensors.extend(meth_rec_z_space_tensors); tensor_names.extend(["z_mu_meth","z_var_meth"])
    
    tensors.extend([rna_expectation_tensor,\
                    mirna_expectation_tensor,\
                    meth_expectation_tensor])
    tensor_names.extend(["rna_expecation","mirna_expectation","meth_expectation"])
                      
    if self.network.HasLayer("gen_rna_space_basic"):
      tensors.extend([rna_basic_expectation_tensor,\
                      mirna_basic_expectation_tensor,\
                      meth_basic_expectation_tensor])
      tensor_names.extend(["rna_basic_expecation","mirna_basic_expectation","meth_basic_expectation"])
      
    tensors.extend([positive_tissue_prediction_tensor,negative_tissue_prediction_tensor])
    tensor_names.extend(["target_prediction_pos","target_prediction_neg"])
    assert len(tensor_names)==len(tensors), "should be same number"
    self.network.FillFeedDict( feed_dict, impute_dict )

    #pdb.set_trace()
    rna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[RNA]] == 1
    meth_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[METH]] == 1
    mirna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[miRNA]] == 1
    tissue_observed_query = np.ones( (len(barcodes),), dtype=bool)
        
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

    pos_tissue_expectation          = np.zeros( (len(barcodes), self.dims_dict[TISSUE] ), dtype=float )
    neg_tissue_expectation          = np.zeros( (len(barcodes), self.dims_dict[TISSUE] ), dtype=float )
    pos_tissue_expectation_no_bias          = np.zeros( (len(barcodes), self.dims_dict[TISSUE] ), dtype=float )
    neg_tissue_expectation_no_bias          = np.zeros( (len(barcodes), self.dims_dict[TISSUE] ), dtype=float )
        
      #drop_likelihoods = np.zeros( rna_dim )
    if use_dna:
      dna_dim = self.dims_dict[DNA] #/self.n_dna_channels
      dna_expectation = np.zeros( (len(barcodes),dna_dim), dtype=float )
      dna_loglikelihood = np.zeros( (np.sum(dna_observed_query),dna_dim), dtype=float )
    
    nbr_splits = 1
    tensor2fill = []
    drop_factor = 1.0 #float(nbr_splits)/float(nbr_splits-1)
    for drop_idx in range(nbr_splits):
    
      id_start = 0
      
      if use_z:
        #z_mu = tensor2fill_eval[0]
        tensor2fill.extend( rec_z_space_tensors )
        z_ids = [id_start,id_start+1]
        id_start+=2
        
      # ------
      # RNA
      # -----
      if use_rna:
        drop_rna_ids = np.arange(drop_idx,self.dims_dict[RNA],nbr_splits, dtype=int)
        batch_data = self.rna_store.loc[ barcodes ].values #self.data_store[self.RNA_key].loc[ barcodes ]
        nans = np.isnan( batch_data )
        if np.sum(nans)>0: 
          batch_data = self.fill_nans( batch_data, nans, RNA, barcodes )
          
        #batch[ RNA_INPUT ] = drop_factor*self.NormalizeRnaInput( batch_data.fillna( 0 ).values )
        #batch[ RNA_INPUT ][nans] = 0
        #batch[ RNA_INPUT][:,drop_rna_ids] = 0
        batch[ RNA_INPUT] = batch_data #.values
        tensor2fill.extend( [rna_expectation_tensor, rna_basic_expectation_tensor, loglikes_data_as_matrix["gen_rna_space"], loglikes_data_as_matrix["gen_rna_space_basic"] ] )
        rna_ids = [id_start,id_start+1,id_start+2,id_start+3]
        id_start+=4

      # ------
      # miRNA
      # -----
      if use_mirna:
        drop_mirna_ids = np.arange(drop_idx,self.dims_dict[miRNA],nbr_splits, dtype=int)
        batch_data = self.mirna_store.loc[ barcodes ].values #self.data_store[self.miRNA_key].loc[ barcodes ]
        nans = np.isnan( batch_data )
        if np.sum(nans)>0: 
          #if mode == "VAL":
          #  pdb.set_trace()
          batch_data = self.fill_nans( batch_data, nans, miRNA, barcodes )
          
        batch[ miRNA_INPUT ] = batch_data #.values
        #else:
        #  batch[ miRNA_INPUT ] = batch_data.values
          
        #batch[ miRNA_INPUT ] = drop_factor*self.NormalizemiRnaInput( batch_data.fillna( 0 ).values )
        #batch[ miRNA_INPUT ][nans] = 0
        #batch[ miRNA_INPUT][:,drop_mirna_ids] = 0
        tensor2fill.extend( [mirna_expectation_tensor, mirna_basic_expectation_tensor, loglikes_data_as_matrix["gen_mirna_space"], loglikes_data_as_matrix["gen_mirna_space_basic"] ] )
        mirna_ids = [id_start,id_start+1,id_start+2,id_start+3]
        id_start+=4
             
      
      # ------
      # DNA
      # -----
      if use_dna:
        drop_dna_ids = np.arange(drop_idx,dna_dim,nbr_splits, dtype=int)
        dna_data_inputs = np.minimum(1.0,dna_data)
        #dna_data_inputs[:,drop_dna_ids] = 0
        #batch[ DNA_INPUT ] = drop_factor*dna_data_inputs
        batch[ DNA_INPUT ] = dna_data_inputs
        tensor2fill.extend( [dna_expectation_tensor, loglikes_data_as_matrix["gen_dna_space"] ] )
        dna_ids = [id_start,id_start+1]
        id_start+=2
        
      
      # ------
      # METH
      # -----
      if use_meth:
        drop_meth_ids = np.arange(drop_idx,self.dims_dict[METH],nbr_splits, dtype=int)
        batch_data = self.meth_store.loc[ barcodes ].values #self.data_store[self.METH_key].loc[ barcodes ]
        nans = np.isnan( batch_data )
        if np.sum(nans)>0: 
          batch_data = self.fill_nans( batch_data, nans, METH, barcodes )
          #pdb.set_trace()
        
        batch[ METH_INPUT ] = batch_data #.values
        #else:
        #  batch[ METH_INPUT ] = batch_data.values
          
        #batch[ METH_INPUT ] = drop_factor*batch_data.fillna( 0 ).values
        #batch[ METH_INPUT][:,drop_meth_ids] = 0
        tensor2fill.extend( [meth_expectation_tensor, meth_basic_expectation_tensor, loglikes_data_as_matrix["gen_meth_space"], loglikes_data_as_matrix["gen_meth_space_basic"] ] )
        meth_ids = [id_start,id_start+1,id_start+2,id_start+3]
        id_start+=4
      # columns = self.meth_genes
      # observations = self.data_store[self.METH_key].loc[ barcodes ].values
      
      if use_tissue:
        # ADD TISUSE PREDICTIONS
        tensor2fill.extend([negative_tissue_prediction_tensor,no_bias_negative_tissue_prediction_tensor])
        tissue_ids = [id_start,id_start+1]
        id_start+=2
      
      # ---------
      # RUN SESS
      # ---------
      self.network.FillFeedDict( feed_dict, batch )
      tensor2fill_eval = sess.run( tensor2fill, feed_dict = feed_dict )

      # ------
      # FILL EVALUATION 
      # -----      
      if use_rna:
        rna_expectation         = tensor2fill_eval[rna_ids[0]]
        rna_basic_expectation   = tensor2fill_eval[rna_ids[1]]
        rna_loglikelihood       = tensor2fill_eval[rna_ids[2]]
        rna_basic_loglikelihood = tensor2fill_eval[rna_ids[3]]

      if use_mirna:
        mirna_expectation         = tensor2fill_eval[mirna_ids[0]]
        mirna_basic_expectation   = tensor2fill_eval[mirna_ids[1]]
        mirna_loglikelihood       = tensor2fill_eval[mirna_ids[2]]
        mirna_basic_loglikelihood = tensor2fill_eval[mirna_ids[3]]
      
      if use_dna:
        dna_expectation = tensor2fill_eval[dna_ids[0]]
        dna_loglikelihood = tensor2fill_eval[dna_ids[1]]
        
      if use_meth:
        meth_expectation        = tensor2fill_eval[meth_ids[0]]
        meth_basic_expectation  = tensor2fill_eval[meth_ids[1]]
        meth_loglikelihood       = tensor2fill_eval[meth_ids[2]]
        meth_basic_loglikelihood = tensor2fill_eval[meth_ids[3]]
        
      if use_tissue:
        tissue_data = batch["TISSUE_input"]
        neg_tissue_expectation = tensor2fill_eval[tissue_ids[0]]
        neg_tissue_expectation_no_bias = tensor2fill_eval[tissue_ids[1]]
        
        
        z_mu = tensor2fill_eval[z_ids[0]]
        z_var = tensor2fill_eval[z_ids[1]]
        
        feed_dict[self.network.GetLayer( "Z_rec_input" ).tensor] = z_mu #+ np.sqrt(z_var)*batch["u_z"]
        
        evals = sess.run( [positive_tissue_prediction_tensor,no_bias_positive_tissue_prediction_tensor], feed_dict = feed_dict )
        pos_tissue_expectation = evals[0]
        pos_tissue_expectation_no_bias = evals[1]
        #pos_tissue_expectation = tensor2fill_eval[tissue_ids[0]]
        #pdb.set_trace()
        
    
    #pdb.set_trace()   
    if use_rna:   
      self.WriteRunFillExpectation( epoch, RNA, barcodes, self.rna_genes, rna_observed_query, rna_expectation, self.rna_store.loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, RNA, barcodes[rna_observed_query], self.rna_genes, rna_loglikelihood, mode )
      
      self.WriteRunFillExpectation( epoch, RNA+"_b", barcodes, self.rna_genes, rna_observed_query, rna_basic_expectation, self.rna_store.loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, RNA+"_b", barcodes[rna_observed_query], self.rna_genes, rna_basic_loglikelihood, mode )

    if use_meth:
      self.WriteRunFillExpectation( epoch, METH, barcodes, self.meth_genes, meth_observed_query, meth_expectation, self.meth_store.loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, METH, barcodes[meth_observed_query], self.meth_genes, meth_loglikelihood, mode )

      self.WriteRunFillExpectation( epoch, METH+"_b", barcodes, self.meth_genes, meth_observed_query, meth_basic_expectation, self.meth_store.loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, METH+"_b", barcodes[meth_observed_query], self.meth_genes, meth_basic_loglikelihood, mode )


    if use_mirna:
      self.WriteRunFillLoglikelihood( epoch, miRNA, barcodes[mirna_observed_query], self.mirna_hsas, mirna_loglikelihood, mode )
      self.WriteRunFillExpectation( epoch, miRNA, barcodes, self.mirna_hsas, mirna_observed_query, mirna_expectation, self.mirna_store.loc[ barcodes ].values, mode )

      self.WriteRunFillLoglikelihood( epoch, miRNA+"_b", barcodes[mirna_observed_query], self.mirna_hsas, mirna_basic_loglikelihood, mode )
      self.WriteRunFillExpectation( epoch, miRNA+"_b", barcodes, self.mirna_hsas, mirna_observed_query, mirna_basic_expectation, self.mirna_store.loc[ barcodes ].values, mode )

    
    if use_dna:
      self.WriteRunFillExpectation( epoch, DNA, barcodes, self.dna_genes, dna_observed_query, dna_expectation, dna_data, mode )
      self.WriteRunFillLoglikelihood( epoch, DNA, barcodes[dna_observed_query], self.dna_genes, dna_loglikelihood, mode )

    if use_tissue:
      self.WriteRunFillExpectation( epoch, TISSUE+"+", barcodes, self.tissue_names, tissue_observed_query, pos_tissue_expectation, tissue_data, mode )
      self.WriteRunFillExpectation( epoch, TISSUE+"-", barcodes, self.tissue_names, tissue_observed_query, neg_tissue_expectation, tissue_data, mode )
      self.WriteRunFillExpectation( epoch, TISSUE+"no_bias+", barcodes, self.tissue_names, tissue_observed_query, pos_tissue_expectation_no_bias, tissue_data, mode )
      self.WriteRunFillExpectation( epoch, TISSUE+"no_bias-", barcodes, self.tissue_names, tissue_observed_query, neg_tissue_expectation_no_bias, tissue_data, mode )

  def RunFill2_old( self, epoch, sess, feed_dict, impute_dict, mode ):
    print "COMPUTE Z-SPACE"
    use_dna = False
    use_rna = True
    use_meth = True
    use_mirna = True
    use_tissue = True
    use_z = True
    
    if self.network.HasLayer( "gen_dna_space"):
      use_dna = True
    
    
    barcodes = impute_dict[BARCODES]
    batch = self.FillBatch( impute_dict[BARCODES], mode )
    #not_observed = np.setdiff1d( self.input_sources, inputs2use )
        
    rec_z_space_tensors       = self.network.GetTensor( "rec_z_space" )
    if self.network.HasLayer( "rec_z_space_rna" ):
      rna_rec_z_space_tensors   = self.network.GetTensor( "rec_z_space_rna" )
      mirna_rec_z_space_tensors   = self.network.GetTensor( "rec_z_space_mirna" )
      meth_rec_z_space_tensors  = self.network.GetTensor( "rec_z_space_meth" )
    
  
    rna_expectation_tensor = self.network.GetLayer( "gen_rna_space" ).expectation
    mirna_expectation_tensor = self.network.GetLayer( "gen_mirna_space" ).expectation
    meth_expectation_tensor = self.network.GetLayer( "gen_meth_space" ).expectation
    
    if self.network.HasLayer("gen_rna_space_basic"):
      rna_basic_expectation_tensor = self.network.GetLayer( "gen_rna_space_basic" ).expectation
      mirna_basic_expectation_tensor = self.network.GetLayer( "gen_mirna_space_basic" ).expectation
      meth_basic_expectation_tensor = self.network.GetLayer( "gen_meth_space_basic" ).expectation
    
    positive_tissue_prediction_tensor = self.network.GetLayer( "target_prediction_pos" ).expectation
    negative_tissue_prediction_tensor = self.network.GetLayer( "target_prediction_neg" ).expectation
    no_bias_positive_tissue_prediction_tensor = self.network.GetLayer( "target_prediction_pos" ).expectation_no_bias
    no_bias_negative_tissue_prediction_tensor = self.network.GetLayer( "target_prediction_neg" ).expectation_no_bias
    
    if use_dna:
      dna_expectation_tensor = self.network.GetLayer( "gen_dna_space" ).expectation
      dna_data = np.minimum( 1.0, self.dna_store.loc[ barcodes ].fillna( 0 ).values )
      dna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[DNA]] == 1
      
    loglikes_data_as_matrix = self.network.loglikes_data_as_matrix
  
    tensors = []; tensor_names=[]
    tensors.extend(rec_z_space_tensors); tensor_names.extend(["z_mu","z_var"])
    
    if self.network.HasLayer( "rec_z_space_rna" ):
      tensors.extend(rna_rec_z_space_tensors); tensor_names.extend(["z_mu_rna","z_var_rna"])
      tensors.extend(mirna_rec_z_space_tensors); tensor_names.extend(["z_mu_mirna","z_var_mirna"])
      tensors.extend(meth_rec_z_space_tensors); tensor_names.extend(["z_mu_meth","z_var_meth"])
    
    tensors.extend([rna_expectation_tensor,\
                    mirna_expectation_tensor,\
                    meth_expectation_tensor])
    tensor_names.extend(["rna_expecation","mirna_expectation","meth_expectation"])
                      
    if self.network.HasLayer("gen_rna_space_basic"):
      tensors.extend([rna_basic_expectation_tensor,\
                      mirna_basic_expectation_tensor,\
                      meth_basic_expectation_tensor])
      tensor_names.extend(["rna_basic_expecation","mirna_basic_expectation","meth_basic_expectation"])
      
    tensors.extend([positive_tissue_prediction_tensor,negative_tissue_prediction_tensor])
    tensor_names.extend(["target_prediction_pos","target_prediction_neg"])
    assert len(tensor_names)==len(tensors), "should be same number"
    self.network.FillFeedDict( feed_dict, impute_dict )

    #pdb.set_trace()
    rna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[RNA]] == 1
    meth_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[METH]] == 1
    mirna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[miRNA]] == 1
    tissue_observed_query = np.ones( (len(barcodes),), dtype=bool)
        
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

    pos_tissue_expectation          = np.zeros( (len(barcodes), self.dims_dict[TISSUE] ), dtype=float )
    neg_tissue_expectation          = np.zeros( (len(barcodes), self.dims_dict[TISSUE] ), dtype=float )
    pos_tissue_expectation_no_bias          = np.zeros( (len(barcodes), self.dims_dict[TISSUE] ), dtype=float )
    neg_tissue_expectation_no_bias          = np.zeros( (len(barcodes), self.dims_dict[TISSUE] ), dtype=float )
        
      #drop_likelihoods = np.zeros( rna_dim )
    if use_dna:
      dna_dim = self.dims_dict[DNA] #/self.n_dna_channels
      dna_expectation = np.zeros( (len(barcodes),dna_dim), dtype=float )
      dna_loglikelihood = np.zeros( (np.sum(dna_observed_query),dna_dim), dtype=float )
    
    nbr_splits = 50
    tensor2fill = []
    drop_factor = 1.0 #float(nbr_splits)/float(nbr_splits-1)
    for drop_idx in range(nbr_splits):
    
      id_start = 0
      
      if use_z:
        #z_mu = tensor2fill_eval[0]
        tensor2fill.extend( rec_z_space_tensors )
        z_ids = [id_start,id_start+1]
        id_start+=2
        
      # ------
      # RNA
      # -----
      if use_rna:
        drop_rna_ids = np.arange(drop_idx,self.dims_dict[RNA],nbr_splits, dtype=int)
        batch_data = self.rna_store.loc[ barcodes ] #self.data_store[self.RNA_key].loc[ barcodes ]
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
        batch_data = self.mirna_store.loc[ barcodes ] #self.data_store[self.miRNA_key].loc[ barcodes ]
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
        batch_data = self.meth_store.loc[ barcodes ] #self.data_store[self.METH_key].loc[ barcodes ]
        batch[ METH_INPUT ] = drop_factor*batch_data.fillna( 0 ).values
        batch[ METH_INPUT][:,drop_meth_ids] = 0
        tensor2fill.extend( [meth_expectation_tensor, meth_basic_expectation_tensor, loglikes_data_as_matrix["gen_meth_space"], loglikes_data_as_matrix["gen_meth_space_basic"] ] )
        meth_ids = [id_start,id_start+1,id_start+2,id_start+3]
        id_start+=4
      # columns = self.meth_genes
      # observations = self.data_store[self.METH_key].loc[ barcodes ].values
      
      if use_tissue:
        # ADD TISUSE PREDICTIONS
        tensor2fill.extend([negative_tissue_prediction_tensor,no_bias_negative_tissue_prediction_tensor])
        tissue_ids = [id_start,id_start+1]
        id_start+=2
      
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
        dna_expectation[:,drop_dna_ids] = tensor2fill_eval[dna_ids[0]][:,drop_dna_ids]
        dna_loglikelihood[:,drop_dna_ids] = tensor2fill_eval[dna_ids[1]][:,drop_dna_ids]
        
      if use_meth:
        meth_expectation[:,drop_meth_ids]         = tensor2fill_eval[meth_ids[0]][:,drop_meth_ids]
        meth_basic_expectation[:,drop_meth_ids]   = tensor2fill_eval[meth_ids[1]][:,drop_meth_ids]
        meth_loglikelihood[:,drop_meth_ids]       = tensor2fill_eval[meth_ids[2]][:,drop_meth_ids]
        meth_basic_loglikelihood[:,drop_meth_ids] = tensor2fill_eval[meth_ids[3]][:,drop_meth_ids]
        
      if use_tissue:
        tissue_data = batch["TISSUE_input"]
        neg_tissue_expectation = tensor2fill_eval[tissue_ids[0]]
        neg_tissue_expectation_no_bias = tensor2fill_eval[tissue_ids[1]]
        
        
        z_mu = tensor2fill_eval[z_ids[0]]
        z_var = tensor2fill_eval[z_ids[1]]
        
        feed_dict[self.network.GetLayer( "Z_rec_input" ).tensor] = z_mu #+ np.sqrt(z_var)*batch["u_z"]
        
        evals = sess.run( [positive_tissue_prediction_tensor,no_bias_positive_tissue_prediction_tensor], feed_dict = feed_dict )
        pos_tissue_expectation = evals[0]
        pos_tissue_expectation_no_bias = evals[1]
        #pos_tissue_expectation = tensor2fill_eval[tissue_ids[0]]
        #pdb.set_trace()
        
    
    #pdb.set_trace()   
    if use_rna:   
      self.WriteRunFillExpectation( epoch, RNA, barcodes, self.rna_genes, rna_observed_query, rna_expectation, self.rna_store.loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, RNA, barcodes[rna_observed_query], self.rna_genes, rna_loglikelihood, mode )
      
      self.WriteRunFillExpectation( epoch, RNA+"_b", barcodes, self.rna_genes, rna_observed_query, rna_basic_expectation, self.rna_store.loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, RNA+"_b", barcodes[rna_observed_query], self.rna_genes, rna_basic_loglikelihood, mode )

    if use_meth:
      self.WriteRunFillExpectation( epoch, METH, barcodes, self.meth_genes, meth_observed_query, meth_expectation, self.meth_store.loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, METH, barcodes[meth_observed_query], self.meth_genes, meth_loglikelihood, mode )

      self.WriteRunFillExpectation( epoch, METH+"_b", barcodes, self.meth_genes, meth_observed_query, meth_basic_expectation, self.meth_store.loc[ barcodes ].values, mode )
      self.WriteRunFillLoglikelihood( epoch, METH+"_b", barcodes[meth_observed_query], self.meth_genes, meth_basic_loglikelihood, mode )


    if use_mirna:
      self.WriteRunFillLoglikelihood( epoch, miRNA, barcodes[mirna_observed_query], self.mirna_hsas, mirna_loglikelihood, mode )
      self.WriteRunFillExpectation( epoch, miRNA, barcodes, self.mirna_hsas, mirna_observed_query, mirna_expectation, self.mirna_store.loc[ barcodes ].values, mode )

      self.WriteRunFillLoglikelihood( epoch, miRNA+"_b", barcodes[mirna_observed_query], self.mirna_hsas, mirna_basic_loglikelihood, mode )
      self.WriteRunFillExpectation( epoch, miRNA+"_b", barcodes, self.mirna_hsas, mirna_observed_query, mirna_basic_expectation, self.mirna_store.loc[ barcodes ].values, mode )

    
    if use_dna:
      self.WriteRunFillExpectation( epoch, DNA, barcodes, self.dna_genes, dna_observed_query, dna_expectation, dna_data, mode )
      self.WriteRunFillLoglikelihood( epoch, DNA, barcodes[dna_observed_query], self.dna_genes, dna_loglikelihood, mode )

    if use_tissue:
      self.WriteRunFillExpectation( epoch, TISSUE+"+", barcodes, self.tissue_names, tissue_observed_query, pos_tissue_expectation, tissue_data, mode )
      self.WriteRunFillExpectation( epoch, TISSUE+"-", barcodes, self.tissue_names, tissue_observed_query, neg_tissue_expectation, tissue_data, mode )
      self.WriteRunFillExpectation( epoch, TISSUE+"no_bias+", barcodes, self.tissue_names, tissue_observed_query, pos_tissue_expectation_no_bias, tissue_data, mode )
      self.WriteRunFillExpectation( epoch, TISSUE+"no_bias-", barcodes, self.tissue_names, tissue_observed_query, neg_tissue_expectation_no_bias, tissue_data, mode )


  def RunFillDna( self, epoch, sess, feed_dict, impute_dict, mode ):
    print "COMPUTE Z-SPACE"
    use_dna = False
    use_rna = True
    use_meth = True
    use_mirna = True
    use_tissue = True
    use_z = True
    
    if self.network.HasLayer( "gen_dna_space"):
      use_dna = True
    
    
    barcodes = impute_dict[BARCODES]
    batch = self.FillBatch( impute_dict[BARCODES], mode )
    
    rec_z_space_tensors       = self.network.GetTensor( "rec_z_space" )
    
    dna_expectation_tensor = self.network.GetLayer( "gen_dna_space" ).expectation
    
    
    dna_data = np.minimum( 1.0, self.dna_store.loc[ barcodes ].fillna( 0 ).values )
    dna_observed_query = batch[ INPUT_OBSERVATIONS ][:,self.observed_batch_order[DNA]] == 1

                    
    self.network.FillFeedDict( feed_dict, impute_dict )


    dna_dim = self.dims_dict[DNA] #/self.n_dna_channels
    dna_expectation = np.zeros( (len(barcodes),dna_dim), dtype=float )
    dna_loglikelihood = np.zeros( (np.sum(dna_observed_query),dna_dim), dtype=float )
    
    nbr_splits = 50
    tensor2fill = []

    id_start = 0
    dna_data_inputs = np.minimum(1.0,dna_data)
    batch[ DNA_INPUT ] = drop_factor*dna_data_inputs
    
    
    
    tensors = []; tensor_names=[]
    tensors.extend(rec_z_space_tensors); tensor_names.extend(["z_mu","z_var"])
    
    #tensor2fill1 = 
    
    tensor2fill.extend( [dna_expectation_tensor, loglikes_data_as_matrix["gen_dna_space"] ] )
    dna_ids = [id_start,id_start+1]
    id_start+=2
        
    
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
      dna_expectation[:,drop_dna_ids] = tensor2fill_eval[dna_ids[0]][:,drop_dna_ids]
      dna_loglikelihood[:,drop_dna_ids] = tensor2fill_eval[dna_ids[1]][:,drop_dna_ids]
      
    if use_meth:
      meth_expectation[:,drop_meth_ids]         = tensor2fill_eval[meth_ids[0]][:,drop_meth_ids]
      meth_basic_expectation[:,drop_meth_ids]   = tensor2fill_eval[meth_ids[1]][:,drop_meth_ids]
      meth_loglikelihood[:,drop_meth_ids]       = tensor2fill_eval[meth_ids[2]][:,drop_meth_ids]
      meth_basic_loglikelihood[:,drop_meth_ids] = tensor2fill_eval[meth_ids[3]][:,drop_meth_ids]
      
    if use_tissue:
      tissue_data = batch["TISSUE_input"]
      neg_tissue_expectation = tensor2fill_eval[tissue_ids[0]]
      neg_tissue_expectation_no_bias = tensor2fill_eval[tissue_ids[1]]
      
      
      z_mu = tensor2fill_eval[z_ids[0]]
      z_var = tensor2fill_eval[z_ids[1]]
      
      feed_dict[self.network.GetLayer( "Z_rec_input" ).tensor] = z_mu #+ np.sqrt(z_var)*batch["u_z"]
      
      evals = sess.run( [positive_tissue_prediction_tensor,no_bias_positive_tissue_prediction_tensor], feed_dict = feed_dict )
      pos_tissue_expectation = evals[0]
      pos_tissue_expectation_no_bias = evals[1]
      #pos_tissue_expectation = tensor2fill_eval[tissue_ids[0]]
      #pdb.set_trace()

    #if use_dna:
    self.WriteRunFillExpectation( epoch, DNA, barcodes, self.dna_genes, dna_observed_query, dna_expectation, dna_data, mode )
    self.WriteRunFillLoglikelihood( epoch, DNA, barcodes[dna_observed_query], self.dna_genes, dna_loglikelihood, mode )

    # if use_tissue:
    #   self.WriteRunFillExpectation( epoch, TISSUE+"+", barcodes, self.tissue_names, tissue_observed_query, pos_tissue_expectation, tissue_data, mode )
    #   self.WriteRunFillExpectation( epoch, TISSUE+"-", barcodes, self.tissue_names, tissue_observed_query, neg_tissue_expectation, tissue_data, mode )
    #   self.WriteRunFillExpectation( epoch, TISSUE+"no_bias+", barcodes, self.tissue_names, tissue_observed_query, pos_tissue_expectation_no_bias, tissue_data, mode )
    #   self.WriteRunFillExpectation( epoch, TISSUE+"no_bias-", barcodes, self.tissue_names, tissue_observed_query, neg_tissue_expectation_no_bias, tissue_data, mode )
 
       
       
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
      try:
        w_mean = self.model_store[ input_source + post_fix + "/W/w0"].values
        w_scale = self.model_store[ input_source + post_fix + "/W/w1"].values
      except:
        continue
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

  def VizRecHidden(self, sess, info_dict ):
    #return
    
    
    if self.network.HasLayer( "rec_hidden" ):
      layer = self.network.GetLayer( "rec_hidden")
      
      layer_specs = self.arch_dict["recognition"]["layers"]
      for layer_spec in layer_specs:
        if layer_spec["name"] == "rec_hidden":
          inputs = layer_spec["inputs"]
          input_sources = []
          for input in inputs:
            source = input.split("_")[0]
            input_sources.append(source)
          break
    
    else:
      print "Could not find rec_hidden"
      return
      
    self.model_store.open()
    
    f = pp.figure()
    #input_sources = ["METH","RNA","miRNA"]
    #orders        = [self.meth_order,self.rna_order,self.mirna_order]
    #data_means    = [self.meth_mean,self.rna_mean,self.mirna_mean]
    n_sources = len(input_sources)
    
    post_fix = "_scaled"
    idx=1
    
    W = {}
    for w_idx, input_source in zip( range(n_sources), input_sources ):
      w = self.model_store[ "rec_hidden" + "/W/w%d"%(w_idx)].values
      #pdb.set_trace()
      
      
      d,k = w.shape
      
      columns = np.array( ["h_%d"%i for i in range(k)])
      if input_source == "RNA":
        rows = self.rna_genes
        print input_source, w.shape, len(rows), len(columns)
        W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
        
      if input_source == "miRNA":
        rows = self.mirna_hsas
        print input_source, w.shape, len(rows), len(columns)
        W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
        
      if input_source == "METH":
        rows = np.array( [ "M-%s"%g for g in self.meth_genes], dtype=str )
        print input_source, w.shape, len(rows), len(columns)
        W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
        
      if input_source == "TISSUE":
        rows = self.tissue_names
        print input_source, w.shape, len(rows), len(columns)
        W[ input_source ] = pd.DataFrame( w, index=rows, columns = columns )
      
      #if input_source == "INPUT_observations": 
        
         
      
      # colors = "brgymcbrgymcbrgymcbrgymcbrgymcbrgymcbrgymcbrgkmcbrgymcbrgymcbrgymcbrgymcbrgymcbrgymcbrgymc"
      # for t_idx in range(n_tissues):
      #   m = mean[:,t_idx][w_i]
      #   s = std_dev[:,t_idx][w_i]
      #   #pdb.set_trace()
      #   ax.fill_between( w_0, m - 0.5*s, m + 0.5*s, alpha=0.25, color=colors[t_idx] )
      #   ax.plot( w_0, m,colors[t_idx]+'-' )
      #
      #
      # pp.plot( w_0, data_means[idx-1][w_i], 'k--', lw=2, alpha=0.5)
      # pp.ylabel( input_source )
      # pp.ylim(0,1)
      # idx+=1
    self.model_store.close()
    
    #pp.show()
    
    #pdb.set_trace()
    #cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
    
    # W_all = pd.concat( W.values(), axis=0 )
    # rownames = W_all.index.values
    #
    # W_corr_hidden = W_all.corr()
    # W_corr_inputs = W_all.T.corr()
    
    f = pp.figure( figsize=(24,12) )
    ax_rna=f.add_subplot(231)
    ax_rna_sort=f.add_subplot(234)
    ax_meth=f.add_subplot(232)
    ax_meth_sort=f.add_subplot(235)
    ax_mirna=f.add_subplot(233)
    ax_mirna_sort=f.add_subplot(236)
    
    ax_rna.plot( W["RNA"].values, 'r-', lw=0.2 )
    ax_meth.plot( W["METH"].values, 'b-', lw=0.2 )
    ax_mirna.plot( W["miRNA"].values, 'g-', lw=0.2 )
    
    ax_rna_sort.plot( np.sort( W["RNA"].values, 0), 'r-', lw=0.2 )
    ax_meth_sort.plot( np.sort( W["METH"].values, 0), 'b-', lw=0.2 )
    ax_mirna_sort.plot(np.sort(  W["miRNA"].values, 0), 'g-', lw=0.2 )
    
    #ax_rna.plot( W["RNA"].values, 'k-', lw=0.2 )
    # mask = np.zeros_like(W_corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    
    # htmap = sns.clustermap ( W_corr_hidden, cmap=cmap, square=True )
    # #htmap.set_yticklabels( list(rownames), rotation='horizontal', fontsize=8 )
    # #htmap.set_xticklabels( list(rownames), rotation='vertical', fontsize=8 )
    #
    # pp.setp(htmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    # pp.setp(htmap.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    # pp.setp(htmap.ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
    # pp.setp(htmap.ax_heatmap.xaxis.get_majorticklabels(), fontsize=8)
    
    pp.savefig( self.viz_hidden_weights + "_sorted_hidden.png", fmt="png", bbox_inches = "tight") 

    # f2 = pp.figure(figsize=(32,24))
    # ax2=f.add_subplot(111)
    # # mask = np.zeros_like(W_corr, dtype=np.bool)
    # # mask[np.triu_indices_from(mask)] = True
    #
    # htmap2 = sns.clustermap ( W_corr_inputs, cmap=cmap, square=True )
    # #htmap.set_yticklabels( list(rownames), rotation='horizontal', fontsize=8 )
    # #htmap.set_xticklabels( list(rownames), rotation='vertical', fontsize=8 )
    
    # pp.setp(htmap2.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    # pp.setp(htmap2.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    # pp.setp(htmap2.ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
    # pp.setp(htmap2.ax_heatmap.xaxis.get_majorticklabels(), fontsize=8)
    #
    # pp.savefig( self.viz_hidden_weights + "_corr_heatmap_inputs.png", fmt="png", bbox_inches = "tight")
    #
    # # f3 = pp.figure(figsize=(32,24))
    # # ax3=f.add_subplot(111)
    # # # mask = np.zeros_like(W_corr, dtype=np.bool)
    # # # mask[np.triu_indices_from(mask)] = True
    # #
    # # htmap3 = sns.clustermap ( W_all, cmap=cmap, square=False )
    # # #htmap.set_yticklabels( list(rownames), rotation='horizontal', fontsize=8 )
    # # #htmap.set_xticklabels( list(rownames), rotation='vertical', fontsize=8 )
    # #
    # # pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    # # pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    # # pp.setp(htmap3.ax_heatmap.yaxis.get_majorticklabels(), fontsize=8)
    # # pp.setp(htmap3.ax_heatmap.xaxis.get_majorticklabels(), fontsize=8)
    # #
    # # pp.savefig( self.viz_hidden_weights + "_weights_heatmap.png", fmt="png", bbox_inches = "tight")
    # #
    # #
    # #
    # #pdb.set_trace()
    # #
    #
          
  def VizModel( self, sess, info_dict ): 
    print "** VIZ Model"
    
    #self.VizWeightsGeneric(sess, info_dict )
    #self.VizInputScales(sess, info_dict )
    self.VizRecHidden( sess, info_dict )
    #self.model_store.open()
    #keys = self.model_store.keys()
    #print keys
    #pdb.set_trace()
    #self.model_store.close()


      #pdb.set_trace()
  
  def fill_nans( self, X, NaNs, source, barcodes ):
    n,d = X.shape
    
    nan_sum = NaNs.sum(1)
    
    I = pp.find( nan_sum > 0  )
    barcodes_with_nans = barcodes[I]
    
    expectation = self.source2expectation_by_tissue[source]
    
    tissues = self.data_store[self.TISSUE_key].loc[ barcodes_with_nans ]
    #pdb.set_trace()
    X[I,:] = np.dot( tissues, expectation )
    
    #pdb.set_trace()
    if np.sum( np.any(np.isnan(X)))>0:
      pdb.set_trace()
    return X
        
  def AddDnaNoise( self, X, rate=0.5 ):
    #return X
    #Y=X.copy()
    sums = X.sum(0)
    n,d = X.shape
    I1 = X==1
    I0 = X==0
    for j in xrange(d):
      
      x = X[:,j]
      y=x.copy()
      
      i1 = pp.find(I1[:,j])
      i0 = pp.find(I0[:,j])
      
      n1 = int(x.sum())
      
      if n1 <1:
        continue
      
      n0 = n-n1
      
      #print n1, max(1,int(rate*n1)), rate*n1
      I20 = np.random.permutation( n1 )[:max(1,int(rate*n1))]
      n1a = len(I20)
      #pdb.set_trace()
      x[i1[I20]] = 0
      
      i0 = pp.find(x==0)
      n01 = len(i0)
      
      I21 = np.random.permutation( n01 )[:n1a]
      
      x[i0[I21]]=1
      
      #pdb.set_trace()
      X[:,j] = x
    
    sums2 = X.sum(0)
    
    assert np.sum( sums==sums2)==d, "should be equal"
    #pdb.set_trace()
    return X
      
      
      
    # a,b = X.shape
    #
    # x = X.flatten()
    #
    # #I =
    # I=pp.find(x>0)
    # J=pp.find(x==0)
    #
    # #true_freq = float(len(I))/float(len(J))
    #
    # r1 = np.random.rand(len(I)) < rate
    # n0 = np.sum(r1)
    # r2 = np.random.permutation(J)[:n0]
    # # move n0 to J
    #
    # #r2 = np.random.rand(len(J)) < rate2
    # #pdb.set_trace()
    # x[I[r1]]=0
    # x[r2]=1
    #
    # return x.reshape((a,b))
    