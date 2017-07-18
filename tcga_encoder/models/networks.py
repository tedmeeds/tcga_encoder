# import tensorflow as tf
# #from tensorflow import *
# from models.layers import *
# from models.regularizers import *
# from utils.utils import *
# from utils.definitions import *

from tcga_encoder.utils.helpers import *

import tensorflow as tf
from tcga_encoder.models.layers import *
from tcga_encoder.models.regularizers import *
from tcga_encoder.definitions.nn import *

class NeuralNetwork(object):
  def AddLayer( self, layer ):
    self.layers[ layer.name ] = layer
  
  def GetLayer( self, name ):
    return self.layers[ name ]
  
  def HasLayer( self, name ):
    return self.layers.has_key(name)
    
  def HasDropout( self, name ):
    return self.dropouts.has_key(name)

  def GetDropout( self, name ):
    return self.dropouts[name]
    
  def GetTensor( self, name ):
    return self.layers[ name ].tensor
    
  def AddRegularizer(self, name ):
    return self.layers[ name ].AddRegularizer( name )
  
  def OrderedDictOp( self, op, odict ):
    newodict = OrderedDict()
    for k,v in odict.iteritems():
      newodict[k] = op(v)
    return newodict
    
  def GetFromVarDict( self, key, value, var_dict ):
    if value.__class__ == str:
      print "GetFromVarDict:: getting %s from var_dict for %s"%(key,value)
      return var_dict[value]
    else:
      print "GetFromVarDict:: setting %s as is: "%(key), value
      return value

  def GetLayerDType( self, layer_dict, var_dict ):
    return tf.float32
          
  def BuildDataShape( self, layer_dict, var_dict ):
    shape = []
    for s in layer_dict[SHAPE]:
      shape.append( self.GetFromVarDict( s, s, var_dict ) )
    return shape

  def BuildLayerSpecs( self, layer_specs, var_dict )  :
    #layer_specs = {}
    for key,value in layer_specs.iteritems():
      if key == SHAPE:
        #shape: ["n_z"] => shape = [10]
        shape = []
        for v in value:
          shape.append( self.GetFromVarDict(key,v,var_dict) )
        layer_specs[key] = shape
        
      elif key == N_UNITS:
        layer_specs[key] = self.GetFromVarDict(key,value,var_dict)
      
      elif key == TRANSFER:
        layer_specs[key] = value  
      
      elif key == MIN_VARIANCE:
        layer_specs[key] = value
        
      elif key == MAX_VARIANCE:
        layer_specs[key] = value
        
      elif key == PRIOR:
        layer_specs[key] = value
        
      elif key == OUTPUT:
        layer_specs[key] = value
        
      elif key == "weight_constant":
        layer_specs[key] = value
          
      elif key == LAYER:
        pass  
      
      elif key == NAME:
        pass
        
      elif key == INPUTS:
        pass
        
      elif key == SHARED:
        layer_specs[key] = value
        
      else:
        #elif key == OUTPUT:
        layer_specs[key] = value
        print "WARNING: ADDING layer specs for KEY=%s"%(key)
        
    return layer_specs
          
  def BuildLayer( self, layer_dict, data_dict, var_dict ):
    if layer_dict[LAYER] == DataLayer:
      print "Building Data Layer: ",layer_dict[NAME]
      is_sparse = False
      if layer_dict.has_key("is_sparse"):
        is_sparse = layer_dict["is_sparse"]
        print "ADDING SPARSE DATA LAYER"
      self.AddLayer( DataLayer( shape = self.BuildDataShape( layer_dict, var_dict ), \
                                dtype = self.GetLayerDType( layer_dict, var_dict ), \
                                is_sparse = is_sparse, \
                                name  = layer_dict[NAME] ) )
    elif layer_dict[LAYER] == MaskLayer:
      print "Building Mask Layer: ",layer_dict[NAME]
      self.AddLayer( MaskLayer( name  = layer_dict[NAME] ) )
    else:
      print "Building Connected Layer: ",layer_dict[NAME]
      layer_specs = self.BuildLayerSpecs( layer_dict, var_dict )
      shared_layers = None
      print layer_specs
      if layer_specs.has_key("shared"):
        shared_layers_list = layer_specs["shared"]
        shared_layers = []
        for x in shared_layers_list:
          layer_borrowed_from = x[1]
          weights_for_input = x[0]
          
          shared_layers.append( [weights_for_input, self.GetLayer(layer_borrowed_from ) ] )
        print "Found shared layers ", shared_layers
        #pdb.set_trace()
      self.AddLayer( Connect( layer_dict[LAYER], \
                              self.GetInputLayers( layer_dict, data_dict, var_dict ), \
                              layer_specs = layer_specs, \
                              shared_layers = shared_layers, \
                              name        = layer_dict[NAME] ) )
                              
      if layer_dict[LAYER] == DropoutLayer:
        self.AddDropout( self.GetLayer(layer_dict[NAME]) )      

  def AddDropout( self, layer ):
    self.dropouts[ layer.name ] = layer
    
  def GetInputLayers( self, layer_dict, data_dict, var_dict ):
    inputs = []
    tensor_ids = {}
    input_list = []
    if layer_dict.has_key(INPUTS):
      if layer_dict[INPUTS].__class__ == list:
        for name in layer_dict[INPUTS]:
          if self.HasLayer(name):
            inputs.append( self.GetLayer(name) )
            #tensor_ids.append( None )
            input_list.append(name)
          else:
            print "Could not find %s, so... looking for input_template"%(name)
            if layer_dict.has_key( "input_template" ):
              assert layer_dict.has_key( "input_template" ), "need a template if layer does not exist."
              input_list = []
              name_template = layer_dict["input_template"] # how to sub in name
              assert name_template.has_key( name ), "cannot find %s in template"%(name)
            
              for templ_name in data_dict[name]:
                input_list.append( name_template[name].replace("?",templ_name) )
                inputs.append(  self.GetLayer( input_list[-1] ) )
                #tensor_ids.append( None )
            else:
              # 
              s_name,tensor_idx = name.split("/")
              if self.HasLayer(s_name):
                print "Found %s for %s"%(s_name,name)
                tensor_idx = int(tensor_idx)
                inputs.append( self.GetLayer(s_name) )
                input_list.append(name)
                tensor_ids[ s_name] =  tensor_idx
      else:
        # assume is key to data_dict names
        input_list = []
        name_template = layer_dict["name_template"] # how to sub in name
        
        for name in data_dict[layer_dict[INPUTS]]:
          input_list.append( name_template.replace("?",name) )
          inputs.append(  self.GetLayer( input_list[-1] ) )
          #tensor_ids.append( None )
        
        #pdb.set_trace()
    else:
      print "** no inputs for ",layer_dict[NAME]
    layer_dict[INPUTS] = input_list
    layer_dict["tensor_ids"] = tensor_ids
    return inputs

  def BuildDataLayers( self, specs_list, data_dict, var_dict ):
      # data_layers:
      #   - source: "dna_genes"
      #     layer: !!python/name:models.layers.DataLayer
      #     name_template: "?_observations"
      #     shape_template: ["n_channels","n_input_?"]
    for layer_dict in specs_list:
      source = layer_dict["source"] # names in data_dict
      name_template = layer_dict["name_template"] # how to sub in name
      shape_template = layer_dict["shape_template"] # how to sub in name
      for name in data_dict[source]:
        layer_dict[NAME] = name_template.replace("?",name)
        shape = []
        for s in shape_template:
          if s.find("?")>=0:
            shape.append( s.replace("?",name))
          else:
            shape.append(s)
        layer_dict[SHAPE] = shape
        self.BuildLayer( layer_dict, data_dict, var_dict)
        self.layer_names.append( layer_dict[NAME] )
    return self.layer_names
    
  def BuildLayers( self, specs_dict, data_dict, var_dict ):
    self.layer_names = []
    if specs_dict.has_key( DATA_LAYERS ):
      self.BuildDataLayers( specs_dict[DATA_LAYERS], data_dict, var_dict )
      
    for layer_dict in specs_dict[LAYERS]:
      self.BuildLayer( layer_dict, data_dict, var_dict)
      self.layer_names.append( layer_dict[NAME] )
    return self.layer_names

  def ApplyRegularizers( self, regs, var_dict ):
    penalties = []
    for reg in regs:
      layer = self.GetLayer( reg[LAYER] )
      lam   = self.GetFromVarDict(LAMBDA,reg[LAMBDA],var_dict)
      weights = layer.weights
      regularizer = reg[REG]( lam )
      try:
        ww = [item for sublist in weights for item in sublist]
      except:
        ww = weights
        
      ok = np.ones( len(ww), dtype=int )
      if reg.has_key("avoid"):
        for idx in reg["avoid"]:
          ok[idx] = 0
      #pdb.set_trace()  
      idx = 0
      for w in ww:
        if ok[idx]:
          penalties.append( regularizer.Apply( w ) )
        idx += 1
    #pdb.set_trace()    
    return penalties
    
  def BuildLoglikelihoods( self, loglik_layers, data_dict, var_dict, as_matrix = False ):
    loglikes = OrderedDict()
    for loglik_dict in loglik_layers:
      layer        = self.GetLayer( loglik_dict[MODEL] )
      if loglik_dict[OBSERVATIONS].__class__==list:
        if len(loglik_dict[OBSERVATIONS])>1:
          observations = [self.GetLayer( o_layer) for o_layer in loglik_dict[OBSERVATIONS]]
        else:
          observations = self.GetLayer( loglik_dict[OBSERVATIONS][0])
      else:
        observations = self.GetLayer( loglik_dict[OBSERVATIONS] )
      mask_layer = None
      if loglik_dict.has_key( MASK ):
        print "** Applying mask to log likelihood"
        mask_layer = self.GetLayer( loglik_dict[MASK] ).tensor
        loglikes[layer.name] = tf.boolean_mask( layer.LogLikelihood( observations, as_matrix=as_matrix ), mask_layer )
      else:
        loglikes[layer.name] = layer.LogLikelihood( observations, as_matrix=as_matrix )
      # if layer.name == "gen_dna_space":
      #   loglikes[layer.name] *=0
    return loglikes
      
  def FillFeedDict( self, feed_dict, imputation_dict ):
    # use stuff from imputation_dict to fill feed_dict
    for name, imputed_values in imputation_dict.iteritems():
      if self.HasLayer( name ) and self.HasDropout( name ) is False:
        feed_dict[ self.GetTensor(name) ]  = imputed_values
      elif self.HasDropout( name ):
        feed_dict[ self.GetDropout(name).GetKeepRateTensor() ]  = imputed_values
      
  def SaveModelToFile(self, savedir ): 
    # turn into pandas df and save as hd5
    model_dir = os.path.join( savedir, MODEL )
    check_and_mkdir( model_dir )
    
    print "SaveModel: saving dataframes..."
    for layer_name, weight_dict in self.panda_weights.iteritems():
      layerdir = os.path.join( model_dir, layer_name )
      check_and_mkdir( layerdir )
      
      for weight_name, weight_df in weight_dict.iteritems():
        fname = os.path.join( layerdir, weight_name+".hd5" )
        weight_df.to_hdf( fname )
  
  # def OpenHdfStore(self,location, which_one = "model"):
  #   store_name = "%s.h5"%(which_one)
  #   check_and_mkdir( location )
  #   full_name = os.path.join( location, store_name )
  #
  #   # I think we can just open in 'a' mode for both
  #   if os.path.exists(full_name) is False:
  #     return pd.HDFStore(os.path.join( location, store_name ), "w" )
  #   else:
  #     return pd.HDFStore(os.path.join( location, store_name ), "r+" )
  #
  # def CloseHdfStore(self,store):
  #   return store.close()

  def CallBack( self, function_name ):
    print "Received callback for %s"%(function_name)
    
  def SaveModel( self, store, flush=False ):
    
    for layer_name, layer in self.layers.iteritems():
      weights = layer.EvalWeights()
      biases  = layer.EvalBiases()
      
      self.SaveWeights( store, layer_name, weights )
      self.SaveBiases( store, layer_name, biases )
      
      #pdb.set_trace()
    if flush is True:
      store.flush(fsync=True)
    store.close()
      
  def SaveWeights( self, store, name, weights, key = "W" ):
    if weights.__class__ == list:
      for w, idx in zip( weights, range(len(weights))):
        if len(w.shape)==2:
          store[ name + "/%s/w%d"%(key,idx) ] = pd.DataFrame( w )
        else:
          for idx2, ww in zip( range(len(w)), w ):
            store[ name + "/%s/f%d/w%d"%(key,idx,idx2) ] = pd.DataFrame( ww )
    else:
      if len(weights.shape)==2:
        store[ name + "/%s/w%d"%(key,0) ] = pd.DataFrame( weights )
      else:
        for idx2, ww in zip( range(len(weights)), weights ):
          store[ name + "/%s/f%d/w%d"%(key,0,idx2) ] = pd.DataFrame( ww )
    #pdb.set_trace()
      
  def SaveBiases( self, store, name, weights, key = "b" ):
    #print "SaveBiases", name, weights
    if weights.__class__ == list:
      for w, idx in zip( weights, range(len(weights))):
        store[ name + "/%s/b%d"%(key,idx) ] = pd.DataFrame( w )
    else:
      store[ name + "/%s/b%d"%(key,0) ] = pd.DataFrame( weights )

  # def SaveBiases( self, store, name, weights, key = "b" ):
  #   if weights.__class__ == list:
  #     for w, idx in zip( weights, range(len(weights))):
  #       store[ name + "/%s/%d"%(key,idx) ] = pd.DataFrame( w )
  #   else:
  #     store[ name + "/%s/%d"%(key,0) ] = pd.DataFrame( weights )
      
  def LoadModel( self, sess, savedir = None, model_store_name = None ):
    print "** LOADING MODEL"
    #pdb.set_trace()
    if savedir is None:
      print "** LoadModel: Trying to open model_store"
      self.model_store.open()
    else:
      print "** LoadModel: setting store name= %s"%model_store_name
      self.model_store_name = model_store_name
      #pdb.set_trace()
      self.model_store = OpenHdfStore( savedir, self.model_store_name, mode="a"  )

    print "** AFTER OPENING"
    print self.model_store
    
    print "** SETTING WEIGHTS AND BIASES"
    self.LoadWeightsAndBiases( sess, self.model_store )
     
    print "** AFTER SETTING"
    print self.model_store 
    CloseHdfStore(self.model_store)
    print self.model_store 
    return self.model_store
      
  def LoadWeightsAndBiases( self, sess, store, wkey = "W", bkey="b" ):
    weights = {}
    biases = {}
    idx = 0
    store_keys = store.keys()
    for k in store_keys:
      
      
      splits = k.split( "/")
      print "Loading key: %s"%k, splits
      layer_name = splits[1]
      #layer = self.GetLayer(layer_name)
      
      # weight or bias
      store_type = splits[2]
      #store_type = store_type_and_idx[0]
      if store_type == wkey:
        try:
          weights[layer_name].append( store[k].values )
        except:
          weights[layer_name] = [store[k].values]
      elif store_type == bkey:
        try:
          biases[layer_name].append( store[k].values.flatten() )
        except:
          biases[layer_name] = [store[k].values.flatten()]
    
    for layer_name, weight_values in weights.iteritems():
      if len(weight_values) == 1:
        self.GetLayer( layer_name ).SetWeights( sess, weight_values)
      else:
        self.GetLayer( layer_name ).SetWeights( sess, weight_values)
        
    for layer_name, weight_values in biases.iteritems():
      if len(weight_values) == 1:
        self.GetLayer( layer_name ).SetBiases( sess, weight_values[0])
      else:
        self.GetLayer( layer_name ).SetBiases( sess, weight_values)
      


      
class ConditionalVariationalAutoEncoder(NeuralNetwork):
  def __init__( self, arch_dict, data_dict ):
    self.layers   = OrderedDict()
    self.dropouts = OrderedDict()
    
    self.use_matrix = True
    
    self.arch = arch_dict
    self.beta       = tf.placeholder( tf.float32, [], name="beta" )
    self.free_bits = tf.placeholder( tf.float32, [], name="free_bits" )
    
    self.recognition_names = self.BuildLayers( arch_dict[RECOGNITION], data_dict, arch_dict[VARIABLES] )
    self.generative_names  = self.BuildLayers( arch_dict[GENERATIVE],  data_dict, arch_dict[VARIABLES] )
    
    self.weight_penalties = self.ApplyRegularizers( arch_dict[REGULARIZERS], arch_dict[VARIABLES] )
    if len(self.weight_penalties)>0:
      self.weight_penalty = tf.add_n( self.weight_penalties )
    else:
      self.weight_penalty = 0.0
      
    #pdb.set_trace()
    self.loglikes_data_as_matrix  = self.BuildLoglikelihoods( arch_dict[DATA_LOGLIK],    data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_rec_as_matrix   = self.BuildLoglikelihoods( arch_dict[REC_Z_LOGLIK],   data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_prior_as_matrix = self.BuildLoglikelihoods( arch_dict[PRIOR_Z_LOGLIK], data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    
    if arch_dict.has_key( INTER_Z_KL ) and self.arch[INTER_Z_KL_PENALTY]>0:
      self.z_kls = self.InterZ_kl( arch_dict[INTER_Z_KL], self.GetTensor( arch_dict[INTER_Z_KL_MASK] ))
      
    self.loglikes_data  = self.OrderedDictOp( tf.reduce_sum, self.loglikes_data_as_matrix )
    #pdb.set_trace()
    self.loglikes_rec   = self.OrderedDictOp( tf.reduce_sum, self.loglikes_rec_as_matrix )
    self.loglikes_prior = self.OrderedDictOp( tf.reduce_sum, self.loglikes_prior_as_matrix )
    
    self.log_p_source_given_z = self.loglikes_data.values()
    self.log_p_x_given_z  = tf.add_n( self.loglikes_data.values(), name = "log_p_x_given_z" )
    self.log_p_z          = tf.add_n( self.loglikes_prior.values(), name = "log_p_z" )
    self.log_q_z          = tf.add_n( self.loglikes_rec.values(), name = "log_q_z" )

    self.log_p_source_given_z = self.loglikes_data.values()
    self.log_p_x_given_z  = tf.add_n( self.loglikes_data.values(), name = "log_p_x_given_z" )
    self.log_p_z          = tf.add_n( self.loglikes_prior.values(), name = "log_p_z" )
    self.log_q_z          = tf.add_n( self.loglikes_rec.values(), name = "log_q_z" )    
    #self.beta*tf.nn.relu(-self.log_p_z + self.log_q_z - self.free_bits )
    
    self.lower_bound = self.log_p_x_given_z - self.beta*tf.maximum(self.free_bits, self.log_q_z - self.log_p_z )
  
    self.batch_log_tensors = [self.lower_bound,self.log_p_x_given_z,self.log_p_z,self.log_q_z]
    self.batch_log_tensors.extend( self.log_p_source_given_z )
    self.batch_log_columns = ["Epoch","Lower Bound","log p(x|z)", "log p(z)", "log q(z|x)"]
    source_names = ["log p(%s|z)"%specs[SHORT] for specs in arch_dict[DATA_LOGLIK] ]
    self.batch_log_columns.extend(source_names)

  def InterZ_kl( self, list_of_pairs, observation_mask ):
    #mask_tensor = self.GetLayer( observation_mask )
    kls = []
    for pair in list_of_pairs:
      a = pair[0]; b=pair[1]; idx_0 = int(pair[2]); idx_1 = int(pair[3])
      
      #pdb.set_trace()
      kls.append( self.GetLayer(a).KL_mat( self.GetLayer(b)) )
      
      #kls.append( tf.expand_dims(observation_mask[:,idx_0],1)*tf.expand_dims(observation_mask[:,idx_1],1)*self.GetLayer(a).KL_mat( self.GetLayer(b)))
    return kls
    
  def CostToMinimize(self):
    if self.arch.has_key( INTER_Z_KL ) and self.arch[INTER_Z_KL_PENALTY]>0:
      #pdb.set_trace()
      return -self.lower_bound + self.weight_penalty + self.arch[INTER_Z_KL_PENALTY]*tf.reduce_sum(tf.add_n( self.z_kls ))
    else:
      return -self.lower_bound + self.weight_penalty

  def FillFeedDict( self, feed_dict, imputation_dict ):
    # use stuff from imputation_dict to fill feed_dict
    for name, imputed_values in imputation_dict.iteritems():
      if self.HasLayer( name ) and self.HasDropout( name ) is False:
        feed_dict[ self.GetTensor(name) ]  = imputed_values
      elif self.HasDropout( name ):
        feed_dict[ self.GetDropout(name).GetKeepRateTensor() ]  = imputed_values
    if imputation_dict.has_key("beta"):
      #print "filling beta"
      feed_dict[self.beta] = imputation_dict["beta"]
    if imputation_dict.has_key("free_bits"):
      #print "filling beta"
      feed_dict[self.free_bits] = imputation_dict["free_bits"]
      
class VanillaVariationalAutoEncoder(NeuralNetwork):
  def __init__( self, arch_dict, data_dict ):
    self.layers   = OrderedDict()
    self.dropouts = OrderedDict()
    
    self.use_matrix = True
    
    self.arch = arch_dict
    self.beta       = tf.placeholder( tf.float32, [], name="beta" )
    self.free_bits = tf.placeholder( tf.float32, [], name="free_bits" )
    
    self.recognition_names = self.BuildLayers( arch_dict[RECOGNITION], data_dict, arch_dict[VARIABLES] )
    self.generative_names  = self.BuildLayers( arch_dict[GENERATIVE],  data_dict, arch_dict[VARIABLES] )
    
    self.weight_penalties = self.ApplyRegularizers( arch_dict[REGULARIZERS], arch_dict[VARIABLES] )
    if len(self.weight_penalties)>0:
      self.weight_penalty = tf.add_n( self.weight_penalties )
    else:
      self.weight_penalty = 0.0
      
    #pdb.set_trace()
    self.loglikes_data_as_matrix  = self.BuildLoglikelihoods( arch_dict[DATA_LOGLIK],    data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_rec_as_matrix   = self.BuildLoglikelihoods( arch_dict[REC_Z_LOGLIK],   data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_prior_as_matrix = self.BuildLoglikelihoods( arch_dict[PRIOR_Z_LOGLIK], data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )

    self.loglikes_data  = self.OrderedDictOp( tf.reduce_sum, self.loglikes_data_as_matrix )
    self.loglikes_rec   = self.OrderedDictOp( tf.reduce_sum, self.loglikes_rec_as_matrix )
    self.loglikes_prior = self.OrderedDictOp( tf.reduce_sum, self.loglikes_prior_as_matrix )
    
    self.log_p_source_given_z = self.loglikes_data.values()
    self.log_p_x_given_z  = tf.add_n( self.loglikes_data.values(), name = "log_p_x_given_z" )
    self.log_p_z          = tf.add_n( self.loglikes_prior.values(), name = "log_p_z" )
    self.log_q_z          = tf.add_n( self.loglikes_rec.values(), name = "log_q_z" )
    
    #self.lower_bound = self.log_p_x_given_z + self.log_p_z - self.log_q_z
    self.lower_bound = self.log_p_x_given_z - self.beta*tf.maximum(self.free_bits, self.log_q_z - self.log_p_z )
     
    self.batch_log_tensors = [self.lower_bound,self.log_p_x_given_z,self.log_p_z,self.log_q_z]
    self.batch_log_tensors.extend( self.log_p_source_given_z )
    self.batch_log_columns = ["Epoch","Lower Bound","log p(x|z)", "log p(z)", "log q(z|x)"]
    source_names = ["log p(%s|z)"%specs[SHORT] for specs in arch_dict[DATA_LOGLIK] ]
    self.batch_log_columns.extend(source_names)
     
  def CostToMinimize(self):
    return -self.lower_bound + self.weight_penalty

  def FillFeedDict( self, feed_dict, imputation_dict ):
    # use stuff from imputation_dict to fill feed_dict
    for name, imputed_values in imputation_dict.iteritems():
      if self.HasLayer( name ) and self.HasDropout( name ) is False:
        feed_dict[ self.GetTensor(name) ]  = imputed_values
      elif self.HasDropout( name ):
        feed_dict[ self.GetDropout(name).GetKeepRateTensor() ]  = imputed_values
    if imputation_dict.has_key("beta"):
      #print "filling beta"
      feed_dict[self.beta] = imputation_dict["beta"]
    if imputation_dict.has_key("free_bits"):
      #print "filling beta"
      feed_dict[self.free_bits] = imputation_dict["free_bits"]

class AdversarialVariationalAutoEncoder(NeuralNetwork):
  def __init__( self, arch_dict, data_dict ):
    self.layers   = OrderedDict()
    self.dropouts = OrderedDict()
    
    self.use_matrix = True
    
    self.arch = arch_dict
    self.anti_weight = arch_dict["anti_weight"]
    
    self.train_penalty_factor = tf.placeholder( tf.float32, [], name = "train_penalty_factor" )
    #self.beta       = tf.placeholder( tf.float32, [], name="beta" )
    #self.free_bits = tf.placeholder( tf.float32, [], name="free_bits" )
    
    self.recognition_names = self.BuildLayers( arch_dict[RECOGNITION], data_dict, arch_dict[VARIABLES] )
    self.generative_names  = self.BuildLayers( arch_dict[GENERATIVE],  data_dict, arch_dict[VARIABLES] )
    self.negative_names  = self.BuildLayers( arch_dict["negative_target"],  data_dict, arch_dict[VARIABLES] )
    self.positive_names  = self.BuildLayers( arch_dict["positive_target"],  data_dict, arch_dict[VARIABLES] )
    
    self.weight_penalties = self.ApplyRegularizers( arch_dict[REGULARIZERS], arch_dict[VARIABLES] )
    if len(self.weight_penalties)>0:
      self.weight_penalty = tf.add_n( self.weight_penalties )
    else:
      self.weight_penalty = 0.0
      
    #pdb.set_trace()
    self.loglikes_data_as_matrix  = self.BuildLoglikelihoods( arch_dict[DATA_LOGLIK],    data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_rec_as_matrix   = self.BuildLoglikelihoods( arch_dict[REC_Z_LOGLIK],   data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_prior_as_matrix = self.BuildLoglikelihoods( arch_dict[PRIOR_Z_LOGLIK], data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )

    self.loglikes_pos_target_as_matrix  = self.BuildLoglikelihoods( arch_dict["positive_target_data_loglik"],    data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_neg_target_as_matrix  = self.BuildLoglikelihoods( arch_dict["negative_target_data_loglik"],    data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )

    pos_model_layer     = self.GetLayer( arch_dict["positive_target_data_loglik"][0][MODEL] )
    pos_obs_layer       = self.GetLayer( arch_dict["positive_target_data_loglik"][0][OBSERVATIONS] )
    neg_model_layer     = self.GetLayer( arch_dict["negative_target_data_loglik"][0][MODEL] )
    neg_obs_layer       = self.GetLayer( arch_dict["negative_target_data_loglik"][0][OBSERVATIONS] )
    
    #self.correct_prediction_pos = tf.cast( tf.equal(tf.round(pos_model_layer.expectation), tf.round(pos_obs_layer.tensor)), tf.float32 )
    #self.correct_prediction_neg = tf.cast( tf.equal(tf.round(neg_model_layer.expectation), tf.round(neg_obs_layer.tensor)), tf.float32 )
    self.correct_prediction_pos = tf.equal(tf.argmax(pos_model_layer.expectation,1), tf.argmax(pos_obs_layer.tensor,1))
    self.correct_prediction_neg = tf.equal(tf.argmax(neg_model_layer.expectation,1), tf.argmax(neg_obs_layer.tensor,1))
    
    self.accuracy_pos = tf.reduce_sum(tf.cast(self.correct_prediction_pos, tf.float32))
    self.accuracy_neg = tf.reduce_sum(tf.cast(self.correct_prediction_neg, tf.float32))
    
    self.loglikes_data  = self.OrderedDictOp( tf.reduce_sum, self.loglikes_data_as_matrix )
    self.loglikes_rec   = self.OrderedDictOp( tf.reduce_sum, self.loglikes_rec_as_matrix )
    self.loglikes_prior = self.OrderedDictOp( tf.reduce_sum, self.loglikes_prior_as_matrix )
    
    self.log_p_source_given_z = self.loglikes_data.values()
    self.log_p_x_given_z  = tf.add_n( self.loglikes_data.values(), name = "log_p_x_given_z" )
    self.log_p_z          = tf.add_n( self.loglikes_prior.values(), name = "log_p_z" )
    self.log_q_z          = tf.add_n( self.loglikes_rec.values(), name = "log_q_z" )
    
    self.log_p_t_given_z_pos = tf.reduce_sum( self.loglikes_pos_target_as_matrix.values(), name = "log_p_t_given_z_pos" )
    self.log_p_t_given_z_neg = tf.reduce_sum( self.loglikes_neg_target_as_matrix.values(), name = "log_p_t_given_z_neg" )
    
    self.lower_bound = self.log_p_x_given_z + self.log_p_z - self.log_q_z
    #self.lower_bound = self.log_p_x_given_z + tf.minimum(2*self.train_penalty_factor,1.0)*(self.log_p_z - self.log_q_z)
     
    self.batch_log_tensors = [self.lower_bound,self.log_p_x_given_z,self.log_p_z,self.log_q_z]
    self.batch_log_tensors.extend( self.log_p_source_given_z )
    self.batch_log_tensors.extend( [self.log_p_t_given_z_pos, self.log_p_t_given_z_neg,self.accuracy_pos,self.accuracy_neg])
    self.batch_log_columns = ["Epoch","Lower Bound","log p(x|z)", "log p(z)", "log q(z|x)"]
    source_names = ["log p(%s|z)"%specs[SHORT] for specs in arch_dict[DATA_LOGLIK] ]
    self.batch_log_columns.extend(source_names)
    self.batch_log_columns.extend(["log p(t|z_copy) +", "log p(t|z_rec) -", "acc T+", "acc T-"])
     
  def CostToMinimize(self):
    if self.anti_weight > 0:
      print "ANTI WEIGHT = ",self.anti_weight
      #return -self.lower_bound + self.train_penalty_factor*self.weight_penalty + self.anti_weight*self.log_p_t_given_z_neg - self.log_p_t_given_z_pos
      return -self.lower_bound + self.train_penalty_factor*(self.weight_penalty + self.anti_weight*self.log_p_t_given_z_neg) - self.log_p_t_given_z_pos
    else:
      return -self.lower_bound + self.train_penalty_factor*self.weight_penalty - self.log_p_t_given_z_pos
      
 
  def FillFeedDict( self, feed_dict, imputation_dict ):
    # use stuff from imputation_dict to fill feed_dict
    for name, imputed_values in imputation_dict.iteritems():
      if self.HasLayer( name ) and self.HasDropout( name ) is False:
        feed_dict[ self.GetTensor(name) ]  = imputed_values
      elif self.HasDropout( name ):
        feed_dict[ self.GetDropout(name).GetKeepRateTensor() ]  = imputed_values
    if imputation_dict.has_key("train_penalty_factor"):
      #print "filling beta"
      feed_dict[self.train_penalty_factor] = imputation_dict["train_penalty_factor"]
    # if imputation_dict.has_key("free_bits"):
    #   #print "filling beta"
    #   feed_dict[self.free_bits] = imputation_dict["free_bits"]
            
class MixtureVariationalAutoEncoder(NeuralNetwork):
  def __init__( self, arch_dict, data_dict ):
    self.layers   = OrderedDict()
    self.dropouts = OrderedDict()
    
    self.use_matrix = True
    
    self.arch = arch_dict
    
    self.recognition_names = self.BuildLayers( arch_dict[RECOGNITION], data_dict, arch_dict[VARIABLES] )
    self.generative_names  = self.BuildLayers( arch_dict[GENERATIVE],  data_dict, arch_dict[VARIABLES] )
    
    self.weight_penalties = self.ApplyRegularizers( arch_dict[REGULARIZERS], arch_dict[VARIABLES] )
    if len(self.weight_penalties)>0:
      self.weight_penalty = tf.add_n( self.weight_penalties )
    else:
      self.weight_penalty = 0.0
      
    #pdb.set_trace()
    self.loglikes_data_as_matrix  = self.BuildLoglikelihoods( arch_dict[DATA_LOGLIK],    data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_rec_as_matrix   = self.BuildLoglikelihoods( arch_dict[REC_Z_LOGLIK],   data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_prior_as_matrix = self.BuildLoglikelihoods( arch_dict[PRIOR_Z_LOGLIK], data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )

    self.loglikes_data  = self.OrderedDictOp( tf.reduce_sum, self.loglikes_data_as_matrix )
    self.loglikes_rec   = self.OrderedDictOp( tf.reduce_sum, self.loglikes_rec_as_matrix )
    self.loglikes_prior = self.OrderedDictOp( tf.reduce_sum, self.loglikes_prior_as_matrix )
    
    self.log_p_source_given_z = self.loglikes_data.values()
    self.log_p_x_given_z  = tf.add_n( self.loglikes_data.values(), name = "log_p_x_given_z" )
    self.log_p_z          = tf.add_n( self.loglikes_prior.values(), name = "log_p_z" )
    self.log_q_z          = tf.add_n( self.loglikes_rec.values(), name = "log_q_z" )
    
    self.lower_bound = self.log_p_x_given_z + self.log_p_z - self.log_q_z
  
    self.batch_log_tensors = [self.lower_bound,self.log_p_x_given_z,self.log_p_z,self.log_q_z]
    self.batch_log_tensors.extend( self.log_p_source_given_z )
    self.batch_log_columns = ["Epoch","Lower Bound","log p(x|z)", "log p(z)", "log q(z|x)"]
    source_names = ["log p(%s|z)"%specs[SHORT] for specs in arch_dict[DATA_LOGLIK] ]
    self.batch_log_columns.extend(source_names)
     
  def CostToMinimize(self):
    return -self.lower_bound + self.weight_penalty
    
class GaussianLogNormalVariationalAutoEncoder(NeuralNetwork):
  def __init__( self, arch_dict, data_dict ):
    self.layers   = OrderedDict()
    self.dropouts = OrderedDict()

    self.use_matrix = True

    self.arch = arch_dict

    self.recognition_names = self.BuildLayers( arch_dict[RECOGNITION], data_dict, arch_dict[VARIABLES] )
    self.generative_names  = self.BuildLayers( arch_dict[GENERATIVE],  data_dict, arch_dict[VARIABLES] )

    self.weight_penalties = self.ApplyRegularizers( arch_dict[REGULARIZERS], arch_dict[VARIABLES] )
    if len(self.weight_penalties)>0:
      self.weight_penalty = tf.add_n( self.weight_penalties )
    else:
      self.weight_penalty = 0.0

    #pdb.set_trace()
    self.loglikes_data_as_matrix  = self.BuildLoglikelihoods( arch_dict[DATA_LOGLIK],    data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_rec_as_matrix   = self.BuildLoglikelihoods( arch_dict[REC_Z_LOGLIK],   data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_prior_as_matrix = self.BuildLoglikelihoods( arch_dict[PRIOR_Z_LOGLIK], data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )

    self.loglikes_data  = self.OrderedDictOp( tf.reduce_sum, self.loglikes_data_as_matrix )
    self.loglikes_rec   = self.OrderedDictOp( tf.reduce_sum, self.loglikes_rec_as_matrix )
    self.loglikes_prior = self.OrderedDictOp( tf.reduce_sum, self.loglikes_prior_as_matrix )

    self.log_p_source_given_z = self.loglikes_data.values()
    self.log_p_x_given_z  = tf.add_n( self.loglikes_data.values(), name = "log_p_x_given_z" )
    self.log_p_z          = tf.add_n( self.loglikes_prior.values(), name = "log_p_z" )
    self.log_q_z          = tf.add_n( self.loglikes_rec.values(), name = "log_q_z" )

    self.lower_bound = self.log_p_x_given_z + self.log_p_z - self.log_q_z

    self.batch_log_tensors = [self.lower_bound,self.log_p_x_given_z,self.log_p_z,self.log_q_z]
    self.batch_log_tensors.extend( self.log_p_source_given_z )
    self.batch_log_columns = ["Epoch","Lower Bound","log p(x|z)", "log p(z)", "log q(z|x)"]
    source_names = ["log p(%s|z)"%specs[SHORT] for specs in arch_dict[DATA_LOGLIK] ]
    self.batch_log_columns.extend(source_names)

  def CostToMinimize(self):
    return -self.lower_bound + self.weight_penalty
#

class HouseholderVariationalAutoEncoder(NeuralNetwork):
  def __init__( self, arch_dict, data_dict ):
    self.layers   = OrderedDict()
    self.dropouts = OrderedDict()

    self.use_matrix = True
    self.beta       = tf.placeholder( tf.float32, [], name="beta" )
    self.free_bits = tf.placeholder( tf.float32, [], name="free_bits" )
    
    #self.beta = tf.placeholder_with_default( 1.0, [], name="beta" )
    self.arch = arch_dict

    self.recognition_names = self.BuildLayers( arch_dict[RECOGNITION], data_dict, arch_dict[VARIABLES] )
    self.generative_names  = self.BuildLayers( arch_dict[GENERATIVE],  data_dict, arch_dict[VARIABLES] )

    self.weight_penalties = self.ApplyRegularizers( arch_dict[REGULARIZERS], arch_dict[VARIABLES] )
    if len(self.weight_penalties)>0:
      self.weight_penalty = tf.add_n( self.weight_penalties )
    else:
      self.weight_penalty = 0.0

    #pdb.set_trace()
    self.loglikes_data_as_matrix  = self.BuildLoglikelihoods( arch_dict[DATA_LOGLIK],    data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_rec_as_matrix   = self.BuildLoglikelihoods( arch_dict[REC_Z_LOGLIK],   data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_prior_as_matrix = self.BuildLoglikelihoods( arch_dict[PRIOR_Z_LOGLIK], data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )

    self.loglikes_data  = self.OrderedDictOp( tf.reduce_sum, self.loglikes_data_as_matrix )
    self.loglikes_rec   = self.OrderedDictOp( tf.reduce_sum, self.loglikes_rec_as_matrix )
    self.loglikes_prior = self.OrderedDictOp( tf.reduce_sum, self.loglikes_prior_as_matrix )

    self.log_p_source_given_z = self.loglikes_data.values()
    self.log_p_x_given_z  = tf.add_n( self.loglikes_data.values(), name = "log_p_x_given_z" )
    self.log_p_z          = tf.add_n( self.loglikes_prior.values(), name = "log_p_z" )
    self.log_q_z          = tf.add_n( self.loglikes_rec.values(), name = "log_q_z" )

    self.lower_bound = self.log_p_x_given_z - self.beta*tf.maximum(self.free_bits, self.log_q_z - self.log_p_z )

    self.batch_log_tensors = [self.lower_bound,self.log_p_x_given_z,self.log_p_z,self.log_q_z]
    self.batch_log_tensors.extend( self.log_p_source_given_z )
    self.batch_log_columns = ["Epoch","Lower Bound","log p(x|z)", "log p(z)", "log q(z|x)"]
    source_names = ["log p(%s|z)"%specs[SHORT] for specs in arch_dict[DATA_LOGLIK] ]
    self.batch_log_columns.extend(source_names)

  def CostToMinimize(self):
    return -self.lower_bound + self.weight_penalty
    
  def FillFeedDict( self, feed_dict, imputation_dict ):
    # use stuff from imputation_dict to fill feed_dict
    for name, imputed_values in imputation_dict.iteritems():
      if self.HasLayer( name ) and self.HasDropout( name ) is False:
        feed_dict[ self.GetTensor(name) ]  = imputed_values
      elif self.HasDropout( name ):
        feed_dict[ self.GetDropout(name).GetKeepRateTensor() ]  = imputed_values
    if imputation_dict.has_key("beta"):
      #print "filling beta"
      feed_dict[self.beta] = imputation_dict["beta"]
    if imputation_dict.has_key("free_bits"):
      #print "filling beta"
      feed_dict[self.free_bits] = imputation_dict["free_bits"]
      
class GaussianLogNormalVariationalAutoEncoder(NeuralNetwork):
  def __init__( self, arch_dict, data_dict ):
    self.layers   = OrderedDict()
    self.dropouts = OrderedDict()

    self.use_matrix = True

    self.arch = arch_dict

    self.recognition_names = self.BuildLayers( arch_dict[RECOGNITION], data_dict, arch_dict[VARIABLES] )
    self.generative_names  = self.BuildLayers( arch_dict[GENERATIVE],  data_dict, arch_dict[VARIABLES] )

    self.weight_penalties = self.ApplyRegularizers( arch_dict[REGULARIZERS], arch_dict[VARIABLES] )
    if len(self.weight_penalties)>0:
      self.weight_penalty = tf.add_n( self.weight_penalties )
    else:
      self.weight_penalty = 0.0

    #pdb.set_trace()
    self.loglikes_data_as_matrix  = self.BuildLoglikelihoods( arch_dict[DATA_LOGLIK],    data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_rec_as_matrix   = self.BuildLoglikelihoods( arch_dict[REC_Z_LOGLIK],   data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_prior_as_matrix = self.BuildLoglikelihoods( arch_dict[PRIOR_Z_LOGLIK], data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )

    self.loglikes_data  = self.OrderedDictOp( tf.reduce_sum, self.loglikes_data_as_matrix )
    self.loglikes_rec   = self.OrderedDictOp( tf.reduce_sum, self.loglikes_rec_as_matrix )
    self.loglikes_prior = self.OrderedDictOp( tf.reduce_sum, self.loglikes_prior_as_matrix )

    self.log_p_source_given_z = self.loglikes_data.values()
    self.log_p_x_given_z  = tf.add_n( self.loglikes_data.values(), name = "log_p_x_given_z" )
    self.log_p_z          = tf.add_n( self.loglikes_prior.values(), name = "log_p_z" )
    self.log_q_z          = tf.add_n( self.loglikes_rec.values(), name = "log_q_z" )

    self.lower_bound = self.log_p_x_given_z + self.log_p_z - self.log_q_z

    self.batch_log_tensors = [self.lower_bound,self.log_p_x_given_z,self.log_p_z,self.log_q_z]
    self.batch_log_tensors.extend( self.log_p_source_given_z )
    self.batch_log_columns = ["Epoch","Lower Bound","log p(x|z)", "log p(z)", "log q(z|x)"]
    source_names = ["log p(%s|z)"%specs[SHORT] for specs in arch_dict[DATA_LOGLIK] ]
    self.batch_log_columns.extend(source_names)

  def CostToMinimize(self):
    return -self.lower_bound + self.weight_penalty
#

class LoopyVariationalAutoEncoder(NeuralNetwork):
  def __init__( self, arch_dict, data_dict ):
    self.layers   = OrderedDict()
    self.dropouts = OrderedDict()

    self.use_matrix = True
    self.beta       = tf.placeholder( tf.float32, [], name="beta" )
    self.free_bits = tf.placeholder( tf.float32, [], name="free_bits" )
    
    #self.beta = tf.placeholder_with_default( 1.0, [], name="beta" )
    self.arch = arch_dict

    self.recognition_names = self.BuildLayers( arch_dict[RECOGNITION], data_dict, arch_dict[VARIABLES] )
    self.generative_names  = self.BuildLayers( arch_dict[GENERATIVE],  data_dict, arch_dict[VARIABLES] )

    self.weight_penalties = self.ApplyRegularizers( arch_dict[REGULARIZERS], arch_dict[VARIABLES] )
    if len(self.weight_penalties)>0:
      self.weight_penalty = tf.add_n( self.weight_penalties )
    else:
      self.weight_penalty = 0.0

    #pdb.set_trace()
    self.loglikes_data_as_matrix  = self.BuildLoglikelihoods( arch_dict[DATA_LOGLIK],    data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_rec_as_matrix   = self.BuildLoglikelihoods( arch_dict[REC_Z_LOGLIK],   data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    self.loglikes_prior_as_matrix = self.BuildLoglikelihoods( arch_dict[PRIOR_Z_LOGLIK], data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )

    self.loglikes_data  = self.OrderedDictOp( tf.reduce_sum, self.loglikes_data_as_matrix )
    self.loglikes_rec   = self.OrderedDictOp( tf.reduce_sum, self.loglikes_rec_as_matrix )
    self.loglikes_prior = self.OrderedDictOp( tf.reduce_sum, self.loglikes_prior_as_matrix )

    self.log_p_source_given_z = self.loglikes_data.values()
    self.log_p_x_given_z  = tf.add_n( self.loglikes_data.values(), name = "log_p_x_given_z" )
    self.log_p_z          = tf.add_n( self.loglikes_prior.values(), name = "log_p_z" )
    self.log_q_z          = tf.add_n( self.loglikes_rec.values(), name = "log_q_z" )

    self.lower_bound = self.log_p_x_given_z - self.beta*tf.nn.relu(-self.log_p_z + self.log_q_z - self.free_bits )

    self.batch_log_tensors = [self.lower_bound,self.log_p_x_given_z,self.log_p_z,self.log_q_z]
    self.batch_log_tensors.extend( self.log_p_source_given_z )
    self.batch_log_columns = ["Epoch","Lower Bound","log p(x|z)", "log p(z)", "log q(z|x)"]
    source_names = ["log p(%s|z)"%specs[SHORT] for specs in arch_dict[DATA_LOGLIK] ]
    self.batch_log_columns.extend(source_names)
    
    self.mu_z_given_y = self.GetLayer( arch_dict["mu_z_given_y"] ).z_mu
    self.var_z_given_y = self.GetLayer( arch_dict["mu_z_given_y"] ).z_var
    self.mu_z_given_x = self.GetLayer( arch_dict["mu_z_given_x"] ).z_mu
    
    #self.cost = tf.reduce_sum( tf.square(self.mu_z_given_y-self.mu_z_given_x) / self.var_z_given_y) - self.log_p_z
    KL_mat = self.GetLayer( arch_dict["mu_z_given_y"] ).KL_mat(self.GetLayer( arch_dict["mu_z_given_x"] ))
    self.cost = -self.log_p_x_given_z + self.beta*(tf.reduce_sum(KL_mat) - self.log_p_z )
    
    #self.cost = -self.log_p_x_given_z + self.beta*tf.reduce_sum( tf.square(self.mu_z_given_y-self.mu_z_given_x) / self.var_z_given_y) - self.log_p_z

  def CostToMinimize(self):
    return self.cost + self.weight_penalty
    
  def FillFeedDict( self, feed_dict, imputation_dict ):
    # use stuff from imputation_dict to fill feed_dict
    for name, imputed_values in imputation_dict.iteritems():
      if self.HasLayer( name ) and self.HasDropout( name ) is False:
        feed_dict[ self.GetTensor(name) ]  = imputed_values
      elif self.HasDropout( name ):
        feed_dict[ self.GetDropout(name).GetKeepRateTensor() ]  = imputed_values
    if imputation_dict.has_key("beta"):
      #print "filling beta"
      feed_dict[self.beta] = imputation_dict["beta"]
    if imputation_dict.has_key("free_bits"):
      #print "filling beta"
      feed_dict[self.free_bits] = imputation_dict["free_bits"]
      
      
class GeneralizedLinearRegression(NeuralNetwork):
  def __init__( self, arch_dict, data_dict ):
    self.layers   = OrderedDict()
    self.dropouts = OrderedDict()

    self.use_matrix = True
    self.arch = arch_dict

   
    self.generative_names  = self.BuildLayers( arch_dict,  data_dict, arch_dict[VARIABLES] )

    self.weight_penalties = []
    if arch_dict.has_key(REGULARIZERS):
      self.weight_penalties = self.ApplyRegularizers( arch_dict[REGULARIZERS], arch_dict[VARIABLES] )
    if len(self.weight_penalties)>0:
      self.weight_penalty = tf.add_n( self.weight_penalties )
    else:
      self.weight_penalty = 0.0

    
    self.loglikes_data_as_matrix  = self.BuildLoglikelihoods( arch_dict[DATA_LOGLIK],    data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    

    self.loglikes_data  = self.OrderedDictOp( tf.reduce_sum, self.loglikes_data_as_matrix )

    self.log_p_x  = tf.add_n( self.loglikes_data.values(), name = "log_p_x" )

    self.cost = -self.log_p_x + self.weight_penalty

    self.batch_log_tensors = [self.log_p_x,self.weight_penalty]
    self.batch_log_columns = ["Epoch","log p(x)", "log p(w)"]
    

  def CostToMinimize(self):
    return self.cost
    
  def FillFeedDict( self, feed_dict, imputation_dict ):
    # use stuff from imputation_dict to fill feed_dict
    for name, imputed_values in imputation_dict.iteritems():
      if self.HasLayer( name ) and self.HasDropout( name ) is False:
        feed_dict[ self.GetTensor(name) ]  = imputed_values
      elif self.HasDropout( name ):
        feed_dict[ self.GetDropout(name).GetKeepRateTensor() ]  = imputed_values
    # if imputation_dict.has_key("beta"):
    #   #print "filling beta"
    #   feed_dict[self.beta] = imputation_dict["beta"]
    # if imputation_dict.has_key("free_bits"):
    #   #print "filling beta"
    #   feed_dict[self.free_bits] = imputation_dict["free_bits"]
    
class VanillaClassifier(NeuralNetwork):
  def __init__( self, arch_dict, data_dict ):
    self.layers   = OrderedDict()
    self.dropouts = OrderedDict()

    self.use_matrix = True
    self.arch = arch_dict

   
    self.generative_names  = self.BuildLayers( arch_dict,  data_dict, arch_dict[VARIABLES] )

    self.weight_penalties = self.ApplyRegularizers( arch_dict[REGULARIZERS], arch_dict[VARIABLES] )
    if len(self.weight_penalties)>0:
      self.weight_penalty = tf.add_n( self.weight_penalties )
    else:
      self.weight_penalty = 0.0

    
    self.loglikes_data_as_matrix  = self.BuildLoglikelihoods( arch_dict[DATA_LOGLIK],    data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
    

    self.loglikes_data  = self.OrderedDictOp( tf.reduce_sum, self.loglikes_data_as_matrix )

    self.log_p_x  = tf.reduce_mean( tf.add_n( self.loglikes_data.values(), name = "log_p_x" ) )

    

    model_layer     = self.GetLayer( arch_dict[DATA_LOGLIK][0][MODEL] )
    obs_layer       = self.GetLayer( arch_dict[DATA_LOGLIK][0][OBSERVATIONS] )
    
    # correct error
    self.error = tf.reduce_mean( obs_layer.tensor*(1.0-model_layer.expectation)+(1-obs_layer.tensor)*model_layer.expectation )
    
    # self.cost = -self.log_p_x + self.weight_penalty
    self.cost = -self.log_p_x + self.weight_penalty
    
    # BAD ERROR: self.error = tf.reduce_sum( obs_layer.tensor*model_layer.expectation+(1-obs_layer.tensor)*(1-model_layer.expectation), 1 )
    #all_labels_true = tf.reduce_min(tf.cast(correct_prediction), tf.float32), 1)
    #accuracy2 = tf.reduce_mean(all_labels_true)
    
    self.correct_prediction = tf.cast( tf.equal(tf.round(model_layer.expectation), tf.round(obs_layer.tensor)), tf.float32 )
    
    #self.all_labels_true = tf.cast(self.correct_prediction, tf.float32)

    self.accuracy =  tf.reduce_mean(self.correct_prediction)
    
    #pdb.set_trace()
    self.batch_log_tensors = [self.log_p_x,self.weight_penalty,self.accuracy, self.error]
    self.batch_log_columns = ["Epoch","log p(x)", "log p(w)", "accuracy", "error"]
    

  def CostToMinimize(self):
    return self.cost
    
  def FillFeedDict( self, feed_dict, imputation_dict ):
    # use stuff from imputation_dict to fill feed_dict
    for name, imputed_values in imputation_dict.iteritems():
      if self.HasLayer( name ) and self.HasDropout( name ) is False:
        feed_dict[ self.GetTensor(name) ]  = imputed_values
      elif self.HasDropout( name ):
        feed_dict[ self.GetDropout(name).GetKeepRateTensor() ]  = imputed_values
        