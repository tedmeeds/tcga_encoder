
from tcga_encoder.utils.helpers import *
from tcga_encoder.utils.math_funcs import *

import tensorflow as tf
from tcga_encoder.models.layers import *
from tcga_encoder.models.regularizers import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
from tcga_encoder.definitions.tcga import *
import pdb

def MakeBatchShape( shape ):
  if shape is None:
    return None
  batch_shape = [s for s in shape]
  if len(shape)>0:
    if shape[0] is not None:
      batch_shape.insert(0,None)
  return batch_shape

def MatDif( t1, t2 ):
  dims1 = t1.get_shape().dims
  ndims1 = len(dims1)
  
  dims2 = t2.get_shape().dims
  ndims2 = len(dims2)
  
  if ndims1 == ndims2 == 2:
    # eg t1=[None,20], t2 = [20,4] => t1*t2 => [None,4]
    return t1 - t2
    
  if ndims1 == 2 and ndims2 == 3:
    # eg t1=[None,20], t2 = [20,4,10] => [None,20]*[20,40] => t1*t2 => [None,4,10]
    t2_reshaped = tf.reshape(t2, [-1,dims2[1].value*dims2[2].value])
    inter_result = t1 - t2_reshaped #tf.matmul( t1, t2_reshaped )
    return tf.reshape(inter_result, [-1,dims2[1].value,dims2[2].value] )
  
  if ndims1 == 3 and ndims2 == 2:
    # eg t1=[None,10,4], t2 = [10,4] => [None,40]*[40] => t1*t2 => [None,]
    return tf.reshape(t1, [-1,dims1[1].value*dims1[2].value]) -  t2 
    
  if ndims1 == 3 and ndims2 == 3:
    # eg t1=[None,4,10], t2 = [4,10,30] => [None,40]*[40,30] => t1*t2 => [None,30]
    return tf.reshape(t1, [-1,dims1[1].value*dims1[2].value]) - tf.reshape(t2, [dims2[0].value*dims2[1].value,-1])
  
  assert False, "Cannot handle these sizes "
  print dims1, dims2
      
def MatMul( t1, t2, name ):
  dims1 = t1.get_shape().dims
  ndims1 = len(dims1)
  
  dims2 = t2.get_shape().dims
  ndims2 = len(dims2)
  
  if ndims1 == ndims2 == 2:
    # eg t1=[None,20], t2 = [20,4] => t1*t2 => [None,4]
    return tf.matmul( t1, t2, name=name )
    
  if ndims1 == 2 and ndims2 == 3:
    # eg t1=[None,20], t2 = [20,4,10] => [None,20]*[20,40] => t1*t2 => [None,4,10]
    t2_reshaped = tf.reshape(t2, [-1,dims2[1].value*dims2[2].value])
    inter_result = tf.matmul( t1, t2_reshaped )
    return tf.reshape(inter_result, [-1,dims2[1].value,dims2[2].value] )
  
  if ndims1 == 3 and ndims2 == 2:
    # eg t1=[None,10,4], t2 = [10,4] => [None,40]*[40] => t1*t2 => [None,]
    return tf.matmul( tf.reshape(t1, [-1,dims1[1].value*dims1[2].value]), t2 ) 
    
  if ndims1 == 3 and ndims2 == 3:
    # eg t1=[None,4,10], t2 = [4,10,30] => [None,40]*[40,30] => t1*t2 => [None,30]
    return tf.matmul( tf.reshape(t1, [-1,dims1[1].value*dims1[2].value]), tf.reshape(t2, [dims2[0].value*dims2[1].value,-1]) ) 
  
  assert False, "Cannot handle these sizes "
  print dims1, dims2
  
def SparseMatMul( t1, t2, name ):
  dims1 = t1.get_shape().dims
  ndims1 = len(dims1)
  
  dims2 = t2.get_shape().dims
  ndims2 = len(dims2)
  
  if ndims1 == ndims2 == 2:
    # eg t1=[None,20], t2 = [20,4] => t1*t2 => [None,4]
    return tf.matmul( t1, t2, a_is_sparse = True, name=name )
    
  if ndims1 == 2 and ndims2 == 3:
    # eg t1=[None,20], t2 = [20,4,10] => [None,20]*[20,40] => t1*t2 => [None,4,10]
    t2_reshaped = tf.reshape(t2, [-1,dims2[1].value*dims2[2].value])
    inter_result = tf.matmul( t1, t2_reshaped, a_is_sparse = True )
    return tf.reshape(inter_result, [-1,dims2[1].value,dims2[2].value] )
  
  if ndims1 == 3 and ndims2 == 2:
    # eg t1=[None,10,4], t2 = [10,4] => [None,40]*[40] => t1*t2 => [None,]
    return tf.matmul( tf.reshape(t1, [-1,dims1[1].value*dims1[2].value]), t2, a_is_sparse = True ) 
    
  if ndims1 == 3 and ndims2 == 3:
    # eg t1=[None,4,10], t2 = [4,10,30] => [None,40]*[40,30] => t1*t2 => [None,30]
    return tf.matmul( tf.reshape(t1, [-1,dims1[1].value*dims1[2].value]), tf.reshape(t2, [dims2[0].value*dims2[1].value,-1]), a_is_sparse = True ) 
  
  assert False, "Cannot handle these sizes "
  print dims1, dims2
    
def xavier_init(fan_in, fan_out, constant=0.1): 
  
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)
                             

def weight_init_old(in_shape, out_shape, constant=0.1): 
    low = -constant*np.sqrt(6.0/(sum(in_shape) + sum(out_shape))) 
    high = constant*np.sqrt(6.0/(sum(in_shape) + sum(out_shape)))
    
    s = in_shape + out_shape
    return tf.random_uniform(tuple(s), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

def weight_init(weight_shape, constant=0.1): 
    n_weights = sum( weight_shape )
    low = -constant*np.sqrt(6.0/n_weights) 
    high = constant*np.sqrt(6.0/n_weights)
    
    #s = in_shape + out_shape
    return tf.random_uniform( tuple(weight_shape), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

def EstimateWeightShape( input_shape, output_shape ):
  # assume first dimension of input_shape is None or not used
  print "Estimating weights from ", input_shape, output_shape
  n_dims_in = len(input_shape)
  n_dims_out = len(output_shape)
  
  weight_shape = []
  
  if n_dims_in > 0:
    if input_shape[0] is None:
      for idx in range(n_dims_in-1):
        weight_shape.append( input_shape[idx+1] )
    else:
      for idx in range(n_dims_in):
        weight_shape.append( input_shape[idx] )

  if n_dims_out > 0:
    if output_shape[0] is None:
      for idx in range(n_dims_out-1):
        weight_shape.append( output_shape[idx+1] )
    else:
      for idx in range(n_dims_out):
        weight_shape.append( output_shape[idx] )

        
  
  return weight_shape
                                                            
def MakeWeights( input_sources, output_shape, name = "", has_biases=True, constant=None, shared_layers = None, shared_idx = None ):
    weights   = []
    default_constant = 0.1
    input_idx = 0
    for input_source in input_sources:
      
      weight_shape = EstimateWeightShape( input_source.shape, output_shape )
      
      print "MAKE WEIGHTS for %s"%(name)
      print "                 shape: ", weight_shape
      # if constants is None:
      #   const = default_constant
      # else:
      #   const = constants[input_idx]
      is_shared = False
      if shared_layers is not None:
        assert shared_idx is not None, "should specify weight index for borrowed weight"
        for shared in shared_layers:
          borrowed_for = shared[0]
          borrowed_layer = shared[1]
          
          if input_source.name == borrowed_for:
            w = borrowed_layer.weights[shared_idx][0]
            is_shared = True
      if is_shared is False:
        w = tf.Variable( weight_init( weight_shape, constant=default_constant ), name = "w_"+input_source.name+"2"+name )
      else:
        print "USING BORROWED WEIGHT"
      weights.append(w)
    
    biases = None
    if has_biases:
      biases = tf.Variable( tf.zeros(tuple(output_shape), dtype=tf.float32), name = "b_"+name  )
    
    return weights, biases 

def ForwardPropagate( input_layers, weights, biases, transfer_function = None, name = "", observation_layer = None  ):
  input_activations = []
    
  for idx,source, w in zip( range(len(input_layers)),input_layers, weights ):
    
    if hasattr(source,"is_sparse"):
      print "ForwardPropagate with SPARSE input"
      a_input = SparseMatMul( source.tensor, w, name = "act_input_"+source.name+"2h" )
    else:
      a_input = MatMul( source.tensor, w, name = "act_input_"+source.name+"2h" )
    
    if observation_layer is not None:
      source_w = observation_layer.tensor #expand_dims(t, 1)
      input_activations.append( tf.expand_dims(source_w[:,idx],1)*a_input )
      #pdb.set_trace()
    else:
      input_activations.append( a_input )
  
  added_activations = tf.add_n( input_activations )
  
  if biases is not None:
    if transfer_function is not None:  
      activations = transfer_function( tf.add( added_activations, biases ), name = "act_%s"%(name) )
    else:
      activations = tf.add( added_activations, biases, name = "act_%s"%(name) )
  else:
    if transfer_function is not None:  
      activations = transfer_function( added_activations, name = "act_%s"%(name) )
    else:
      activations = added_activations #, biases, name = "act_%s"%(name) )
    
  return activations, input_activations

def GetPenaltiesFromLayers( list_of_layers ):
  penalties = []
  for layer in list_of_layers:
    if hasattr( layer, "penalties" ):
      print "Adding penalties ",layer.penalties
      penalties.append( layer.penalties )
  return tf.add_n( penalties )
    
def Connect( layer_class, input_layers, layer_specs={}, shared_layers = None, name="" ):
  print "making ", layer_class
  if layer_class == HiddenLayer:
    #print "making HiddenLayer class"
    shape             = layer_specs[SHAPE]
    transfer_function = layer_specs[TRANSFER]
    
    has_biases = True
    if layer_specs.has_key("biases"):
      has_biases = layer_specs["biases"]
    
    constant = 0.1
    if layer_specs.has_key("weight_constant"):
          constant = layer_specs["weight_constant"]
    weights, biases    = MakeWeights( input_layers, shape, name, has_biases=has_biases, constant=constant  )  
    # if shared_layer is None:
    #   weights, biases    = MakeWeights( input_layers, shape, name, has_biases=has_biases  )
    #   #total_penalty = tf.add_n( penalties )
    # else:
    if shared_layers is not None:
      
      for idx, input_layer in zip( range(len(input_layers)), input_layers ):
        for shared in shared_layers:
          borrowed_for = shared[0]
          borrowed_layer = shared[1]
          
          pdb.set_trace()
          #weights = shared_weights.weights
          #biases  = shared_weights.biases
      
    activation, activation_input  = ForwardPropagate( input_layers, weights, biases, transfer_function, name )

    model = {ACTIVATION:activation, ACTIVATION_INPUT:activation_input, WEIGHTS:weights,BIASES:biases}
    
    layer = layer_class( shape, model, name=name )

  elif layer_class == WeightedMultiplyLayer:
    assert len(input_layers) == 2, "must provide 2 inputs"
    transfer = None
    if layer_specs.has_key(TRANSFER):
      transfer = layer_specs[TRANSFER]
      
    layer = layer_class( layer_specs[SHAPE], input_layers, name, transfer )
   
  elif layer_class == WeibullModelLayer:
     assert len(input_layers) == 2, "must provide 2 inputs"
     layer = layer_class( input_layers[0], input_layers[1], name )
     
  elif layer_class == ScaledLayer:
    shape = layer_specs[SHAPE]
    input_layer = input_layers[0]
    
    weights_location = tf.Variable( weight_init( shape, constant=0.1 ), name = name+"_location" )
    weights_scale    = tf.Variable( weight_init( shape, constant=0.1 ), name = name+"_location" )
    
    assert len(input_layers) == 1, "must provide 1 inputs"
    transfer = None
    if layer_specs.has_key(TRANSFER):
      transfer = layer_specs[TRANSFER]
      

    layer = layer_class( shape, input_layer, weights_location, weights_scale, name, transfer )

  elif layer_class == BetaScaledLayer:
    assert len(input_layers)==2, "must have 2 only"
    shape = layer_specs[SHAPE]
    input_layer = input_layers[0]
    beta_layer  = input_layers[1]
    
    #weights_location = tf.Variable( weight_init( shape, constant=0.1 ), name = name+"_location" )
    #weights_scale    = tf.Variable( weight_init( shape, constant=0.1 ), name = name+"_location" )
    
    #assert len(input_layers) == 1, "must provide 1 inputs"
    #transfer = None
    #if layer_specs.has_key(TRANSFER):
    #  transfer = layer_specs[TRANSFER]
      

    layer = layer_class( shape, input_layer, beta_layer, name )

        
  elif layer_class == SumLayer:
    layer = layer_class( input_layers, name )
    
  elif layer_class == DropoutLayer:
    assert len(input_layers) == 1, "only allow one layer"
    layer = layer_class( input_layers[0], name )

  elif layer_class == DroppedSourceHiddenLayer:
    assert len(input_layers) >= 2, "requires at least 2 inputs"
    
    source_layers  = input_layers[:-1]
    observation_layer = input_layers[-1]
    
    shape           = layer_specs[SHAPE]
    transfer_function = layer_specs[TRANSFER]
    has_biases = True
    if layer_specs.has_key("biases"):
      has_biases = layer_specs["biases"]
    
    if shared_layers is None:
      weights, biases    = MakeWeights( input_layers, shape, name, has_biases=has_biases  )
      #total_penalty = tf.add_n( penalties )
    else:
      weights = shared_layers.weights
      biases  = shared_layers.biases
      
    activation, activation_input  = ForwardPropagate( source_layers, weights, biases, transfer_function, name, observation_layer=observation_layer )

    model = {ACTIVATION:activation, ACTIVATION_INPUT:activation_input, WEIGHTS:weights,BIASES:biases}
    
    layer = layer_class( shape, model, name=name )
    
    
  elif layer_class == GeneratedDataLayer:
    assert len(input_layers) >= 2, "requires 2 inputs"
    
    #pdb.set_trace()
    model_layer  = input_layers[0]
    random_layers = input_layers[1:]
    
    if layer_specs.has_key("output"):
      output = layer_specs["output"]
      gen_data = model_layer.GenerateX( random_layers, output=output )
    else:
      gen_data = model_layer.GenerateX( random_layers )
      
    layer = layer_class( layer_specs[SHAPE], tensor=gen_data, name = name )

  elif layer_class == GaussianStaticLayer:
    shape           = layer_specs[SHAPE]
    prior           = layer_specs[PRIOR]
    
    layer = layer_class( shape, prior, name=name )

  elif layer_class == GaussianModelLayer:
    shape           = layer_specs[SHAPE]

    has_biases = True
    if layer_specs.has_key("biases"):
      has_biases = layer_specs["biases"]

    
    if shared_layers is None:
      weights_mu,  biases_mu  = MakeWeights( input_layers, shape, name+"_"+MU, has_biases=has_biases  )
      weights_var, biases_var = MakeWeights( input_layers, shape, name+"_"+VAR, has_biases=has_biases  )
    else:
      weights_mu    = shared_weights.weights[0]
      weights_var   = shared_weights.weights[1]
      biases_mu     = shared_weights.biases[0]
      biases_var    = shared_weights.biases[1]
    
    z_mu, z_mu_input    = ForwardPropagate( input_layers, weights_mu, biases_mu, \
                                                     transfer_function=None, name=name+"_"+MU )

    z_var, z_var_input  = ForwardPropagate( input_layers, weights_var, biases_var, \
                                                     transfer_function=tf.exp, name=name+"_"+VAR )
    mu  = {WEIGHTS:weights_mu,  BIASES:biases_mu,  Z:z_mu }
    var = {WEIGHTS:weights_var, BIASES:biases_var, Z:z_var }
    
    layer = layer_class( shape, {MU:mu, VAR:var}, name=name )

  elif layer_class == GaussianLogNormalStaticLayer:
    shape           = layer_specs[SHAPE]
    prior           = layer_specs[PRIOR]
    
    layer = layer_class( shape, prior, name=name )
    
  elif layer_class == LogNormalStudentModelLayer or layer_class == GaussianLogNormalModelLayer:
    shape           = layer_specs[SHAPE]

    has_biases = True
    if layer_specs.has_key("biases"):
      has_biases = layer_specs["biases"]

    
    if shared_layers is None:
      weights_mu,  biases_mu  = MakeWeights( input_layers, shape, name+"_"+MU, has_biases=has_biases  )
      weights_logprec_mu, biases_logprec_mu = MakeWeights( input_layers, shape, name+"_"+LOG_PREC_MU, has_biases=has_biases  )
      weights_logprec_var, biases_logprec_var   = MakeWeights( input_layers, shape, name+"_"+LOG_PREC_VAR, has_biases=has_biases  )
    else:
      weights_mu    = shared_weights.weights[0]
      weights_var   = shared_weights.weights[1]
      weights_nu    = shared_weights.weights[2]
      biases_mu     = shared_weights.biases[0]
      biases_var    = shared_weights.biases[1]
      biases_nu     = shared_weights.biases[2]
    
    z_mu, z_mu_input    = ForwardPropagate( input_layers, weights_mu, biases_mu, \
                                                     transfer_function=None, name=name+"_"+MU )

    z_logprec_mu, z_logprec_mu_input  = ForwardPropagate( input_layers, weights_logprec_mu, biases_logprec_mu, \
                                                     transfer_function=None, name=name+"_"+LOG_PREC_MU )
                                                     
    z_logprec_var, z_logprec_var_input  = ForwardPropagate( input_layers, weights_logprec_var, biases_logprec_var, \
                                                     transfer_function=tf.exp, name=name+"_"+LOG_PREC_VAR )
                                                     
    mu  = {WEIGHTS:weights_mu,  BIASES:biases_mu,  Z:z_mu }
    logprec_mu = {WEIGHTS:weights_logprec_mu, BIASES:biases_logprec_mu, Z:z_logprec_mu }
    logprec_var = {WEIGHTS:weights_logprec_var, BIASES:biases_logprec_var, Z:z_logprec_var }
    
    layer = layer_class( shape, {MU:mu, LOG_PREC_MU:logprec_mu, LOG_PREC_VAR:logprec_var}, name=name )
    
    
  elif layer_class == StudentModelLayer:
    shape           = layer_specs[SHAPE]

    has_biases = True
    if layer_specs.has_key("biases"):
      has_biases = layer_specs["biases"]

    
    if shared_layers is None:
      weights_mu,  biases_mu  = MakeWeights( input_layers, shape, name+"_"+MU, has_biases=has_biases  )
      weights_var, biases_var = MakeWeights( input_layers, shape, name+"_"+VAR, has_biases=has_biases  )
      weights_nu, biases_nu   = MakeWeights( input_layers, shape, name+"_"+NU, has_biases=has_biases  )
    else:
      weights_mu    = shared_weights.weights[0]
      weights_var   = shared_weights.weights[1]
      weights_nu    = shared_weights.weights[2]
      biases_mu     = shared_weights.biases[0]
      biases_var    = shared_weights.biases[1]
      biases_nu     = shared_weights.biases[2]
    
    z_mu, z_mu_input    = ForwardPropagate( input_layers, weights_mu, biases_mu, \
                                                     transfer_function=None, name=name+"_"+MU )

    z_var, z_var_input  = ForwardPropagate( input_layers, weights_var, biases_var, \
                                                     transfer_function=tf.exp, name=name+"_"+VAR )
                                                     
    z_nu, z_nu_input  = ForwardPropagate( input_layers, weights_nu, biases_nu, \
                                                     transfer_function=tf.exp, name=name+"_"+NU )
                                                     
    mu  = {WEIGHTS:weights_mu,  BIASES:biases_mu,  Z:z_mu }
    var = {WEIGHTS:weights_var, BIASES:biases_var, Z:z_var }
    nu = {WEIGHTS:weights_nu, BIASES:biases_nu, Z:z_nu }
    
    layer = layer_class( shape, {MU:mu, VAR:var, NU:var}, name=name )

  elif layer_class == HouseholderModelLayer:
    shape           = layer_specs[SHAPE]

    has_biases = True
    if layer_specs.has_key("biases"):
      has_biases = layer_specs["biases"]

    
    if shared_layers is None:
      weights_mu,  biases_mu  = MakeWeights( input_layers, shape, name+"_"+MU, has_biases=has_biases  )
      weights_var, biases_var = MakeWeights( input_layers, shape, name+"_"+VAR, has_biases=has_biases  )
      weights_v, biases_v   = MakeWeights( input_layers, shape, name+"_"+"V", has_biases=has_biases  )
    else:
      weights_mu    = shared_weights.weights[0]
      weights_var   = shared_weights.weights[1]
      weights_v    = shared_weights.weights[2]
      biases_mu     = shared_weights.biases[0]
      biases_var    = shared_weights.biases[1]
      biases_v     = shared_weights.biases[2]
    
    z_mu, z_mu_input    = ForwardPropagate( input_layers, weights_mu, biases_mu, \
                                                     transfer_function=None, name=name+"_"+MU )

    z_var, z_var_input  = ForwardPropagate( input_layers, weights_var, biases_var, \
                                                     transfer_function=tf.exp, name=name+"_"+VAR )
                                                     
    z_v, z_v_input  = ForwardPropagate( input_layers, weights_v, biases_v, \
                                                     transfer_function=None, name=name+"_"+"V" )
                                                     
    mu  = {WEIGHTS:weights_mu,  BIASES:biases_mu,  Z:z_mu }
    var = {WEIGHTS:weights_var, BIASES:biases_var, Z:z_var }
    v = {WEIGHTS:weights_v, BIASES:biases_v, Z:z_v }
    
    layer = layer_class( shape, {MU:mu, VAR:var, "V":v}, name=name )

  elif layer_class == HouseholderLayer:
    shape           = layer_specs[SHAPE]

    assert len(input_layers) == 2, "must have 2 inputs"
    y_layer = input_layers[0]
    v_layer = input_layers[0]
    
    # mu  = {WEIGHTS:weights_mu,  BIASES:biases_mu,  Z:z_mu }
    # var = {WEIGHTS:weights_var, BIASES:biases_var, Z:z_var }
    # v = {WEIGHTS:weights_v, BIASES:biases_v, Z:z_v }
    
    layer = layer_class( shape, {"V":v_layer, "Y":y_layer}, name=name )
        
  elif layer_class == GaussianProductLayer:
    assert len(input_layers) >= 2, "requires at least 2 inputs"
    
    source_layers  = input_layers[:-1]
    observation_layer = input_layers[-1]
    
    shape           = layer_specs[SHAPE]
      
    precisions = []
    mu_div_var = []
    #pdb.set_trace()
    product_prec = 0; product_mu_div_var=0
    for idx,source in zip( range(len(source_layers)),source_layers ):
       precisions.append( 1.0/source.GetVariance() )
       mu_div_var.append( source.GetMean()/source.GetVariance() )
       
       
       source_w = observation_layer.tensor #expand_dims(t, 1)
       product_prec += tf.expand_dims(source_w[:,idx],1)*precisions[-1]
       product_mu_div_var += tf.expand_dims(source_w[:,idx],1)*mu_div_var[-1]
    
    product_var  = 1.0 / product_prec
    product_mean = product_var*product_mu_div_var
    
      
    mu  = {Z:product_mean }
    var = {Z:product_var }
    
    layer = layer_class( shape, {MU:mu, VAR:var}, name=name )
    
  elif layer_class == BetaModelLayer:
    shape           = layer_specs[SHAPE]
    prior           = layer_specs[PRIOR]
    has_biases = True
    if layer_specs.has_key("biases"):
      has_biases = layer_specs["biases"]
    
    
    weights_log_a, biases_log_a =  MakeWeights( input_layers, shape, name+"_log_a", has_biases=has_biases, shared_layers=shared_layers, shared_idx = 0  )
    
    weights_log_b,  biases_log_b = MakeWeights( input_layers, shape, name+"_log_b", has_biases=has_biases, shared_layers=shared_layers, shared_idx = 1  )

    a, a_input          = ForwardPropagate( input_layers, weights_log_a, biases_log_a, \
                                                     transfer_function=tf.exp, name=name+"_"+A )
                                                     
    b, b_input          = ForwardPropagate( input_layers, weights_log_b, biases_log_b, \
                                                     transfer_function=tf.exp, name=name+"_"+B )

    #pdb.set_trace()
    a_clipped = tf.clip_by_value( a, 0.00001, 1000.0 )
    b_clipped = tf.clip_by_value( b, 0.00001, 1000.0 )
    
    model     = { A: a_clipped,  \
                  B: b_clipped,     \
                  WEIGHTS:[weights_log_a,weights_log_b], \
                  BIASES:[biases_log_a,biases_log_b], 
                  PRIOR:prior }
    
    layer = layer_class( shape, model, name=name )
    
  elif layer_class == BetaGivenModelLayer:
    assert len(input_layers) == 2, "must only have 2 input layers"
    shape           = None #layer_specs[SHAPE]
    prior           = layer_specs[PRIOR]

    
    model     = { A: input_layers[0].tensor,  \
                  B: input_layers[1].tensor,     \
                  PRIOR:prior }
    
    layer = layer_class( shape, model, name=name )
    
  elif layer_class == SigmoidModelLayer:
    shape           = layer_specs[SHAPE]
    has_biases = True
    if layer_specs.has_key("biases"):
      has_biases = layer_specs["biases"]
    
    if shared_layers is None:
      weights, biases  = MakeWeights( input_layers, shape, name, has_biases=has_biases  )
    else:
      weights = shared_weights.weights
      biases = shared_weights.biases
      
    a, a_input          = ForwardPropagate( input_layers, weights, biases, \
                                                     transfer_function=tf.sigmoid, name=name )
                                                     

    model     = dict(prob = a,   \
                     weights = weights, \
                     biases = biases, \
                     shape = shape )
    
    layer = layer_class( shape, model, name=name )

  # elif layer_class == KumaModelLayer:
  #   shape           = layer_specs[SHAPE]
  #   prior           = layer_specs[PRIOR]
  #   has_biases = True
  #   if layer_specs.has_key("biases"):
  #     has_biases = layer_specs["biases"]
  #
  #   weights_log_a, \
  #   biases_log_a  = MakeWeights( input_layers, weight_shape, name+"_log_a", has_biases=has_biases  )
  #
  #   weights_log_b, \
  #   biases_log_b  = MakeWeights( input_layers, weight_shape, name+"_log_b", has_biases=has_biases  )
  #
  #   log_a, a_input          = ForwardPropagate( input_layers, weights_log_a, biases_log_a, \
  #                                                    transfer_function=None, name=name+"_log_a" )
  #
  #   log_b, b_input          = ForwardPropagate( input_layers, weights_log_b, biases_log_b, \
  #                                                    transfer_function=None, name=name+"_log_b" )
  #
  #   model     = dict(log_a=log_a,     log_b=log_b,     \
  #                    weights=[weights_log_a,weights_log_b], \
  #                    biases=[biases_log_a,biases_log_b],
  #                    prior = prior  )
  #
  #   layer = layer_class( shape, model, name=name )
        
  elif layer_class == SoftmaxModelLayer:
    shape           = layer_specs[SHAPE]
    has_biases = True
    if layer_specs.has_key("biases"):
      has_biases = layer_specs["biases"]

    if shared_layers is None:
      weights, biases = MakeWeights( input_layers, shape, name, has_biases=has_biases  )
    else:
      weights = shared_weights.weights
      biases = shared_weights.biases
      
    a, a_input          = ForwardPropagate( input_layers, weights, biases, \
                                                     transfer_function=tf.nn.softmax, name=name )
                                                     

    model     = dict(prob=a,   \
                     weights=weights, \
                     biases=biases, \
                     shape=shape )
    
    layer = layer_class( shape, model, name=name )
             
  else:
    raise NotImplemented, "No implementation for " + str(layer_class)
      
  return layer
    
class MissingModel(object):
  def __init__( self, kind="full", name=None):
    self.type = kind
    self.observed = tf.placeholder( tf.float32, [None,1], name=name+"_observed" )
       
class DataLayer(object):
  def __init__( self, shape, dtype = tf.float32, tensor = None, is_sparse = False, name = "" ):
    
    self.shape       = shape
    self.batch_shape = MakeBatchShape( shape )
    self.name        = name
    self.is_sparse   = is_sparse
    

    if tensor is None:
      self.tensor = tf.placeholder( dtype, self.batch_shape, name=name )
    else:
      self.tensor = tensor
      
  def EvalWeights(self):
    return []
    
  def EvalBiases(self):
    return []

class MaskLayer(object):
  def __init__( self, name = "" ):
    
    self.shape       = []
    self.batch_shape = [None]
    self.name        = name

    self.tensor = tf.placeholder( tf.bool, self.batch_shape, name=name )
  
      
  def EvalWeights(self):
    return []
    
  def EvalBiases(self):
    return []

class WeightedMultiplyLayer(object):
  def __init__( self, shape, input_layers, name = "", transfer = None ):
    print "WARNING: WeightedMultiplyLayer assuming specific shapes"
    t1 = input_layers[0].tensor
    t2 = input_layers[1].tensor
    
    #t2 = tf.expand_dims( t2, 1 )
    
    
    tensor = tf.reduce_sum( t1*tf.expand_dims( t2, 1 ), 2 )
    
    if transfer is None:
      self.tensor = tensor
    else:
      self.tensor = transfer(tensor)
      
    self.shape       = shape
    self.batch_shape = MakeBatchShape(shape)
    self.name        = name

  def EvalWeights(self):
    return []
    
  def EvalBiases(self):
    return []
        
class ScaledLayer(object):
  #shape, input_layer, weights_location, weights_scale, name, transfer
  def __init__( self, shape, input_layer, weights_location, weights_scale, name = "", transfer = None ):
      
    self.weights_location = weights_location
    self.weights_scale = weights_scale
    
    #pdb.set_trace()
    
    if transfer is None:
      tf.expand_dims( input_layer.tensor, -1 )
      self.tensor = ( tf.expand_dims( input_layer.tensor, -1 ) - weights_location)*weights_scale
    else:
      self.tensor = ( tf.expand_dims( input_layer.tensor, -1 ) - weights_location)*transfer(weights_scale)
      
    self.shape       = shape
    self.batch_shape = MakeBatchShape(shape)
    self.name        = name
    self.weights = [self.weights_location,self.weights_scale]

  def EvalWeights(self):
    if self.weights.__class__ == list:
      return [w.eval() for w in self.weights]
    else:
      return self.weights.eval()
    
  def EvalBiases(self):
    return []

class BetaScaledLayer(object):
  #shape, input_layer, weights_location, weights_scale, name, transfer
  def __init__( self, shape, input_layer, beta_layer, name = "" ):
      
    #self.weights_location = weights_location
    #self.weights_scale = weights_scale
    
    #pdb.set_trace()
    self.a = tf.exp( tf.transpose( beta_layer.weights_a[0] ) )
    self.b = tf.exp( tf.transpose( beta_layer.weights_b[0] ) )
    
    res = [1]
    res.extend(shape)
    self.weights = [ tf.transpose( beta_layer.weights_a[0] ), tf.transpose( beta_layer.weights_b[0] )]
    a_plus_b = self.a+self.b
    self.mean = self.a / a_plus_b
    self.std  = tf.sqrt( (self.a*self.b)/( tf.square(a_plus_b)*(a_plus_b+1.0) ) )
    
    self.tensor = ( tf.expand_dims( input_layer.tensor, -1 ) - self.mean )*self.std 
      
    #pdb.set_trace()
    self.shape       = shape
    self.batch_shape = MakeBatchShape(shape)
    self.name        = name
    #self.weights = beta_layer.weights

  def EvalWeights(self):
    if self.weights.__class__ == list:
      return [w.eval() for w in self.weights]
    else:
      return self.weights.eval()
    
  def EvalBiases(self):
    return []
        
class DifLayer(object):
  def __init__( self, input_layers, name = "" ):
      
    self.tensor = input_layers[0].tensor - input_layers[1].tensor
    
    self.shape       = input_layers[0].shape
    self.batch_shape = input_layers[0].batch_shape
    self.name        = name
    
class SumLayer(object):
  def __init__( self, input_layers, name = "" ):
    
    tensors = []
    for input_layer in input_layers:
      tensors.append( input_layer.tensor )
      
    self.tensor = tf.add_n( tensors )
    
    self.shape       = input_layers[0].shape
    self.batch_shape = input_layers[0].batch_shape
    self.name        = name
    
  def EvalWeights(self):
    return []
    
  def EvalBiases(self):
    return []
    
class DropoutLayer(object):
  def __init__( self, input_layer, name = "" ):
    self.shape       = input_layer.shape
    self.batch_shape = input_layer.batch_shape
    self.name        = name

    self.keep_rate     = tf.placeholder_with_default( 1.0, [], name=input_layer.name+KEEP_RATE )
    self.dropout_scale = 1.0 / self.keep_rate
    self.tensor        = tf.nn.dropout(input_layer.tensor, self.keep_rate ) #self.dropout_scale*input_layer.tensor

  def GetKeepRateTensor(self):
    return self.keep_rate
    
  def EvalWeights(self):
    return []
    
  def EvalBiases(self):
    return []
              
class GeneratedDataLayer(DataLayer):
  def __init__( self, shape, tensor, dtype = tf.float32, name = "" ):
    DataLayer.__init__( self, shape, dtype=dtype, tensor=tensor, name = name )

    
# class MaskedLayer(DataLayer):
#   def __init__( self, shape, tensor, dtype = tf.float32, name = "" ):
#     DataLayer.__init__( self, shape, dtype=dtype, tensor=tensor, missing_model = missing_model, name = name )

  
class HiddenLayer(object):
  def __init__( self, shape, model, name="" ):
    #model = {"activation":activation, "activation_input":activation_input, "weights":weights,"biases":biases}
    
    self.model            = model
    self.shape            = shape
    self.batch_shape      = MakeBatchShape(shape)
    self.weights          = model[WEIGHTS]
    self.biases           = model[BIASES] 
    self.activation       = model[ACTIVATION] 
    self.activation_input = model[ACTIVATION_INPUT] 
    self.name             = name
    
    self.tensor           = self.activation
    
  def EvalWeights(self):
    if self.weights.__class__ == list:
      return [w.eval() for w in self.weights]
    else:
      return self.weights.eval()
    
  def EvalBiases(self):
    if self.biases is None:
      return []
    if self.biases.__class__ == list:
      b = []
      for w in self.biases:
        if w is not None:
          b.append( w.eval())
      return b  
      #return [w.eval() for w in self.biases]
    else:
      return self.biases.eval()
      
  def SetWeights( self, sess, weights ):
    if self.weights.__class__ == list:
      assert weights.__class__ == list, "should assign same weights"
      assert len(weights) == len(self.weights), "should assign same weights"
      
      for tf_w, np_w in zip( self.weights, weights ):
        sess.run(tf_w.assign(np_w))
        
      #return [w.eval() for w in self.weights]
    else:
      sess.run(self.weights.assign(weights))
      #return self.weights.eval()
      
  def SetBiases( self, sess, biases ):
    if self.biases.__class__ == list:
      assert biases.__class__ == list, "should assign same biases"
      assert len(biases) == len(self.biases), "should assign same biases"
      
      for tf_w, np_w in zip( self.biases, biases ):
        sess.run(tf_w.assign(np_w))
    else:
      sess.run(self.biases.assign(biases))
  
  def AddRegularizer( self, reg, weight_idx = 0 ):
    if reg.__class__ == list:
      assert len(reg) == len(self.weights), "if reg is a list, must be one per weight"
      applied = []
      for r,w in zip(reg, self.weights):
        applied.append( r.Apply(w) )
      return tf.add_n( applied )
    else:
      return reg.Apply( self.weights[ weight_idx ] )

class DroppedSourceHiddenLayer(HiddenLayer):
  pass
  
  # def __init__( self, shape, model, name="" ):
  #   return __init__( self, shape, model, nam
      

class WeibullModelLayer(object):
  def __init__( self, scale_var, shape_var, name="" ):
    # alpha == scale
    # beta == shape
    self.shape_var = shape_var.tensor
    self.scale_var = scale_var.tensor
    self.a = self.scale_var
    self.b = self.shape_var
    
    self.log_scale = tf.log( self.scale_var + 1e-12)
    self.log_shape = tf.log( self.shape_var + 1e-12)
    self.name = name
    
  def EvalWeights(self):
    return [] #return wa.extend(wb) #[w[0].eval() for w in self.weights]
    
  def EvalBiases(self):
    #wa = [w.eval() for w in self.biases_a]
    #wb = [w.eval() for w in self.biases_b]
    #wa.extend(wb) #[w[0].eval() for w in self.weights]
    
    return []

  def SetWeights( self, sess, weights ):
    return None
        
      
  def SetBiases( self, sess, biases ):
    return None
  
  #  def LogLikelihood( self, E, T, Z ):
  #    # E: events, binary vector indicating "death" (n by 1)
  #    # T: time of event or censor (n by 1)
  #    # Z: matrix of covariates (n by dim)
  #    log_hazard = self.LogHazard( T, Z )
  #    log_survival = self.LogSurvival( T, Z )
  #
  #    return E*log_hazard + log_survival
  
  def LogHazard( self, T ):
    return self.log_shape + self.log_scale + (self.scale_var-1.0)*tf.log( T )
  
  def LogSurvival( self, T ):
    return -self.CumulativeHazard( T)
  
  def LogCumulativeHazard( self, T ):
    return self.log_shape + self.scale_var*tf.log(T)

  def CumulativeHazard( self, T ):
    return tf.exp( self.LogCumulativeHazard(T) )
                
  def LogLikelihood( self, X, as_matrix = False, boolean_mask = None ):
    #pdb.set_trace()
    #Z = X[0]
    T = X[0]
    E = X[1]
    log_hazard = self.LogHazard( T.tensor )
    log_survival = self.LogSurvival( T.tensor )
    #
    #    return E*log_hazard + log_survival
    
    
    self.loglik_matrix = E.tensor*log_hazard + log_survival

    
    if boolean_mask is not None:
      self.loglik_matrix = tf.boolean_mask( self.loglik_matrix, boolean_mask )
      
    self.loglik = tf.reduce_sum(self.loglik_matrix ,name = self.name+"_loglik")
    if as_matrix is True:
      return self.loglik_matrix
    else:
      return self.loglik
  

      
class GaussianModelLayer(HiddenLayer):    
  def __init__( self, shape, model, prior = None, name="" ):
    self.shape           = shape
    self.batch_shape      = MakeBatchShape(shape)
    self.mu               = model[MU]
    self.var              = model[VAR]
    self.activation       = [self.mu[Z], self.var[Z]]
    
    self.prior    = prior
    self.z_mu     = self.mu[Z]
    self.expectation = self.z_mu
    
    self.z_var    = tf.clip_by_value( self.var[Z], 0.1, 1000.0 ) 
    self.z_logvar = tf.log( self.z_var )
    self.z_std    = tf.sqrt( self.z_var )
    
    self.mu_weights       = self.mu[WEIGHTS]
    self.var_weights      = self.var[WEIGHTS]
    #self.penalties        = self.mu["penalties"] + self.var["penalties"]
    
    self.weights = [self.mu_weights,self.var_weights]
    self.biases = [self.mu[BIASES],self.var[BIASES]]
    
    self.name             = name
    self.output_dims      = [self.shape, self.shape]
    self.tensor           = [self.z_mu, self.z_var]

  def GetVariance(self):
    return self.z_var
    
  def GetMean(self):
    return self.z_mu
    
  def EvalWeights(self):
    return [w[0].eval() for w in self.weights]
    
  def EvalBiases(self):
    if self.biases is None:
      return []
    b = []
    for w in self.biases:
      if w is not None:
        b.append( w.eval())
    return b
    
    # if self.biases is None:
    #   return []
    # return [w.eval() for w in self.biases]

  def SetWeights( self, sess, weights ):
    assert weights.__class__ == list, "should assign same weights"
    assert len(weights) == len(self.weights), "should assign same weights"
      
    for tf_w, np_w in zip( self.weights, weights ):
      sess.run(tf_w[0].assign(np_w))
        
      
  def SetBiases( self, sess, biases ):
    assert biases.__class__ == list, "should assign same biases"
    assert len(biases) == len(self.biases), "should assign same biases"
      
    for tf_w, np_w in zip( self.biases, biases ):
      sess.run(tf_w.assign(np_w))
              
  def KL( self, model = None ):
    
    if model is None:
      if self.prior is None:
        self.latent_kl = -0.5*tf.reduce_sum(1 + self.z_logvar - tf.square(self.z_mu) - self.z_var )
      else:
        print "KL using prior ", self.prior
        p_mu, p_var = self.prior
        log_p_var = np.log(p_var)
      
        self.latent_kl = -0.5*tf.reduce_sum(1 + self.z_logvar - log_p_var - tf.square(self.z_mu-p_mu)/p_var - self.z_var/p_var )
    else:
      print "KL using previous layer as prior" 
      self.latent_kl = -0.5*tf.reduce_sum(1 + self.z_logvar - model.z_logvar - tf.square(self.z_mu-model.z_mu)/model.z_var - self.z_var/model.z_var )
      
    
    return self.latent_kl

  def KL_mat( self, model = None ):
    
    if model is None:
      if self.prior is None:
        self.latent_kl_mat = -0.5*(1 + self.z_logvar - tf.square(self.z_mu) - self.z_var )
      else:
        print "KL using prior ", self.prior
        p_mu, p_var = self.prior
        log_p_var = np.log(p_var)
      
        self.latent_kl_mat = -0.5*(1 + self.z_logvar - log_p_var - tf.square(self.z_mu-p_mu)/p_var - self.z_var/p_var )
    else:
      print "KL using previous layer as prior" 
      self.latent_kl_mat = -0.5*(1 + self.z_logvar - model.z_logvar - tf.square(self.z_mu-model.z_mu)/model.z_var - self.z_var/model.z_var )
      
    
    return self.latent_kl_mat
     
  def CustomDistance( self, model = None ):
    
    if model is None:
      self.custom_distance = tf.reduce_sum( tf.square(self.z_mu) + self.z_var )
    else:
      #print "KL using previous layer as prior" 
      self.custom_distance = tf.reduce_sum( tf.square(self.z_mu-model.z_mu) + tf.square( self.z_std - model.z_std ) )
      #pdb.set_trace()
    
    return self.custom_distance

  def GenerateX( self, u_zs, use_expectation = False  ):
    u_z = u_zs[0]
    if use_expectation:
      z = self.expectation
    else:
      # generate z using deterministic transform
      z = self.z_mu + self.z_std*u_z.tensor
      #pdb.set_trace()

    # return generic data layer
    return z
            
  def Generate( self, u_z, shape, name = "", use_expectation = False  ):
    
    if use_expectation:
      z = self.expectation
    else:
      # generate z using deterministic transform
      z = self.z_mu + self.z_std*u_z.tensor

    # return generic data layer
    return GeneratedDataLayer( shape, tensor=z, name=name )

  def LogLikelihood( self, z, as_matrix = False, boolean_mask = None  ):
    self.loglik_matrix = -0.5*np.log(2*np.pi) - 0.5*self.z_logvar - 0.5*tf.square( z.tensor-self.z_mu )/self.z_var
    self.loglik = tf.reduce_sum(self.loglik_matrix, name = self.name+"_loglik")
    
    if as_matrix is True:
      return self.loglik_matrix
    else:
      return self.loglik
class HouseholderModelLayer(GaussianModelLayer):    
  def __init__( self, shape, model, prior = None, name="" ):
    self.shape           = shape
    self.batch_shape      = MakeBatchShape(shape)
    self.mu               = model[MU]
    self.var              = model[VAR]
    self.v                = model["V"]
    self.activation       = [self.mu[Z], self.var[Z], self.v[Z]]
    
    self.prior    = prior
    self.z_mu     = self.mu[Z]
    self.expectation = self.z_mu
    
    self.z_var    = tf.clip_by_value( self.var[Z], 0.1, 1000.0 ) 
    self.z_logvar = tf.log( self.z_var )
    self.z_std    = tf.sqrt( self.z_var )
    
    self.z_v = self.v[Z]
    self.norm_v = tf.reduce_sum( tf.square( self.z_v ), 1 )
    self.normed_v = self.z_v / self.norm_v
    
    
    self.mu_weights       = self.mu[WEIGHTS]
    self.var_weights      = self.var[WEIGHTS]
    self.v_weights      = self.v[WEIGHTS]
    #self.penalties        = self.mu["penalties"] + self.var["penalties"]
    
    self.weights = [self.mu_weights,self.var_weights,self.v_weights]
    self.biases = [self.mu[BIASES],self.var[BIASES],self.v[BIASES]]
    
    self.name             = name
    self.output_dims      = [self.shape, self.shape, self.shape]
    self.tensor           = [self.z_mu, self.z_var, self.z_v]

  def GenerateX( self, u_zs, output, use_expectation = False  ):
    u_z = u_zs[0]
    if use_expectation:
      z = self.expectation
    else:
      # generate z using deterministic transform
      y = self.z_mu + self.z_std*u_z.tensor
      z = y - 2*self.normed_v*tf.reduce_sum( self.z_v*y, 1 )
      #pdb.set_trace()

    # return generic data layer
    if output==0:
      return y
    else:
      return z
            
  def Generate( self, u_z, shape, name = "", use_expectation = False  ):
    
    if use_expectation:
      z = self.expectation
    else:
      # generate z using deterministic transform
      y = self.z_mu + self.z_std*u_z.tensor
      z = y - 2*self.normed_v*tf.dot( self.z_v.T, y )

    # return generic data layer
    return GeneratedDataLayer( shape, tensor=z, name=name )

  def LogLikelihood( self, z, as_matrix = False, boolean_mask = None  ):
    self.loglik_matrix = -0.5*np.log(2*np.pi) - 0.5*self.z_logvar - 0.5*tf.square( z.tensor-self.z_mu )/self.z_var
    self.loglik = tf.reduce_sum(self.loglik_matrix, name = self.name+"_loglik")
    
    if as_matrix is True:
      return self.loglik_matrix
    else:
      return self.loglik
      
class HouseholderLayer(HiddenLayer):    
  def __init__( self, shape, model, prior = None, name="" ):
    self.model            = model
    self.shape            = shape
    self.batch_shape      = MakeBatchShape(shape)
    self.weights          = []#model[WEIGHTS]
    self.biases           = []#model[BIASES] 
    #self.activation       = model[ACTIVATION] 
    #self.activation_input = model[ACTIVATION_INPUT] 
    self.name             = name
    
    self.y_layer               = model["Y"]
    self.v_layer               = model["V"]
    
    self.v = self.v_layer.tensor
    self.y = self.y_layer.tensor
    #tf.expand_dims(source_w[:,idx],1)
    self.norm_v = tf.expand_dims(tf.reduce_sum( tf.square( self.v ), 1 ), 1)
    self.normed_v = self.v / self.norm_v
    
    self.z = self.y - 2*self.normed_v*tf.expand_dims(tf.reduce_sum( self.v*self.y, 1 ),1)
    
    #pdb.set_trace()
    self.activation       = self.z
    
    self.weights = [] #self.mu_weights,self.var_weights,self.v_weights]
    self.biases = [] #self.mu[BIASES],self.var[BIASES],self.v[BIASES]]
    
    self.output_dims      = self.shape
    self.tensor           = self.activation 

      
class StudentModelLayer(GaussianModelLayer):    
  def __init__( self, shape, model, prior = None, name="" ):
    self.shape           = shape
    self.batch_shape      = MakeBatchShape(shape)
    self.mu               = model[MU]
    self.var              = model[VAR]
    self.nu               = model[NU]
    self.activation       = [self.mu[Z], self.var[Z], self.nu[Z]]
    
    self.prior    = prior
    self.z_mu     = self.mu[Z]
    self.expectation = self.z_mu
    
    self.z_var    = tf.clip_by_value( self.var[Z], 0.1, 1000.0 ) 
    self.z_nu    = tf.clip_by_value( self.nu[Z], 0.1, 1000.0 ) 
    self.z_logvar = tf.log( self.z_var )
    self.z_std    = tf.sqrt( self.z_var )
    
    self.mu_weights       = self.mu[WEIGHTS]
    self.var_weights      = self.var[WEIGHTS]
    self.nu_weights      = self.nu[WEIGHTS]
    #self.penalties        = self.mu["penalties"] + self.var["penalties"]
    
    self.weights = [self.mu_weights,self.var_weights,self.nu_weights]
    self.biases = [self.mu[BIASES],self.var[BIASES],self.nu[BIASES]]
    
    self.name             = name
    self.output_dims      = [self.shape, self.shape, self.shape]
    self.tensor           = [self.z_mu, self.z_var, self.z_nu]
    
    self.log_norm_const = tf.lgamma( (self.z_nu+1)/2.0 ) \
                        - tf.lgamma( self.z_nu/2.0 ) \
                        -0.5*tf.log( self.z_nu ) \
                        - 0.5*self.z_logvar - 0.5*np.log(np.pi) 
   
  def GetDof(self):
    return self.z_nu
    
  def GetNu(self):
    return self.z_nu
              
  def KL( self, model = None ):
    assert False, "Not Implemented"
    # if model is None:
    #   if self.prior is None:
    #     self.latent_kl = -0.5*tf.reduce_sum(1 + self.z_logvar - tf.square(self.z_mu) - self.z_var )
    #   else:
    #     print "KL using prior ", self.prior
    #     p_mu, p_var = self.prior
    #     log_p_var = np.log(p_var)
    #
    #     self.latent_kl = -0.5*tf.reduce_sum(1 + self.z_logvar - log_p_var - tf.square(self.z_mu-p_mu)/p_var - self.z_var/p_var )
    # else:
    #   print "KL using previous layer as prior"
    #   self.latent_kl = -0.5*tf.reduce_sum(1 + self.z_logvar - model.z_logvar - tf.square(self.z_mu-model.z_mu)/model.z_var - self.z_var/model.z_var )
    #
    #
    # return self.latent_kl

  def KL_mat( self, model = None ):
    assert False, "Not Implemented"
    # if model is None:
    #   if self.prior is None:
    #     self.latent_kl_mat = -0.5*(1 + self.z_logvar - tf.square(self.z_mu) - self.z_var )
    #   else:
    #     print "KL using prior ", self.prior
    #     p_mu, p_var = self.prior
    #     log_p_var = np.log(p_var)
    #
    #     self.latent_kl_mat = -0.5*(1 + self.z_logvar - log_p_var - tf.square(self.z_mu-p_mu)/p_var - self.z_var/p_var )
    # else:
    #   print "KL using previous layer as prior"
    #   self.latent_kl_mat = -0.5*(1 + self.z_logvar - model.z_logvar - tf.square(self.z_mu-model.z_mu)/model.z_var - self.z_var/model.z_var )
    #
    #
    # return self.latent_kl_mat
     
  def GenerateX( self, u_zs, use_expectation = False  ):
    u_z = u_zs[0]
    if use_expectation:
      z = self.expectation
    else:
      # generate z using deterministic transform
      #z = self.z_mu + self.z_std*u_z.tensor
      #assert False, 
      chi_sqr = tf.random_gamma(self.shape, self.z_nu/2.0, beta=0.5, dtype=tf.float32)
      z = self.z_mu + self.z_std*u_z.tensor*tf.sqrt(self.z_nu/chi_sqr)

    # return generic data layer
    return z
            
  def Generate( self, u_z, shape, name = "", use_expectation = False  ):
    
    if use_expectation:
      z = self.expectation
    else:
      # generate z using deterministic transform
      chi_sqr = tf.random_gamma(shape, self.z_nu/2.0, beta=0.5, dtype=tf.float32)
      z = self.z_mu + self.z_std*u_z.tensor*tf.sqrt(self.z_nu/chi_sqr)

    # return generic data layer
    return GeneratedDataLayer( shape, tensor=z, name=name )

  def LogLikelihood( self, z, as_matrix = False, boolean_mask = None  ):
    
    self.loglik_matrix = self.log_norm_const - ( (self.z_nu+1.0)/2.0)*tf.log( 1.0 + tf.square( (z.tensor-self.z_mu)/self.z_std )/self.z_nu )
    #self.loglik_matrix = -0.5*np.log(2*np.pi) - 0.5*self.z_logvar - 0.5*tf.square( z.tensor-self.z_mu )/self.z_var
    self.loglik = tf.reduce_sum(self.loglik_matrix, name = self.name+"_loglik")
    
    if as_matrix is True:
      return self.loglik_matrix
    else:
      return self.loglik

class GaussianLogNormalModelLayer(HiddenLayer):    
  def __init__( self, shape, model, prior = None, name="" ):
    self.shape           = shape
    self.batch_shape      = MakeBatchShape(shape)
    self.mu               = model[MU]
    self.log_prec_mu      = model[LOG_PREC_MU]
    self.log_prec_var     = model[LOG_PREC_VAR]
    self.activation       = [self.mu[Z], self.log_prec_mu[Z], self.log_prec_var[Z]]
    
    self.prior    = prior
    self.z_mu     = self.mu[Z]
    self.expectation = self.z_mu
    
    #self.z_var    = tf.clip_by_value( self.var[Z], 0.1, 1000.0 ) 
    self.z_log_prec_var    = tf.clip_by_value( self.log_prec_var[Z], 0.1, 1000.0 ) 
    self.z_log_prec_mu = self.log_prec_mu[Z] 
    
    self.z_log_prec_logvar = tf.log( self.z_log_prec_var )
    self.z_log_prec_std    = tf.sqrt( self.z_log_prec_var )
    
    # self.mu_weights       = self.mu[WEIGHTS]
    # self.var_weights      = self.var[WEIGHTS]
    # self.nu_weights      = self.nu[WEIGHTS]
    #self.penalties        = self.mu["penalties"] + self.var["penalties"]
    
    self.weights = [self.mu[WEIGHTS],self.log_prec_mu[WEIGHTS],self.log_prec_var[WEIGHTS]]
    self.biases = [self.mu[BIASES],self.log_prec_mu[BIASES],self.log_prec_var[BIASES]]
    
    self.name             = name
    self.output_dims      = [self.shape, self.shape, self.shape]
    self.tensor           = [self.z_mu, self.z_log_prec_mu, self.z_log_prec_var]
    
    # self.log_norm_const = tf.lgamma( (self.z_nu+1)/2.0 ) \
    #                     - tf.lgamma( self.z_nu/2.0 ) \
    #                     -0.5*tf.log( self.z_nu ) \
    #                     - 0.5*self.z_logvar - 0.5*np.log(np.pi)
        
            
  def Generate( self, u_z, shape, name = "", use_expectation = False  ):
    
    if use_expectation:
      z = self.expectation
    else:
      # generate z using deterministic transform
      log_prec = self.z_log_prec_mu + u_z.tensor*self.z_log_prec_std
      self.z_prec = tf.exp( log_prec )
      self.z_var = 1.0/self.z_prec
      self.z_std = tf.sqrt(self.z_var)
      self.z_logvar = -log_prec
      #chi_sqr = tf.random_gamma(shape, self.z_nu/2.0, beta=0.5, dtype=tf.float32)
      #z = self.z_mu + self.z_std*u_z.tensor*tf.sqrt(self.z_nu/chi_sqr)
      z = self.z_mu + self.z_std*u_z.tensor

    # return generic data layer
    return GeneratedDataLayer( shape, tensor=z, name=name )

  def GenerateX( self, u_zs, output, use_expectation = False  ):
    
    u_z = u_zs[0]
    u_prec = u_zs[1]
    if use_expectation:
      z = self.expectation
    else:
      # generate z using deterministic transform
      # generate z using deterministic transform
      log_prec = self.z_log_prec_mu + u_prec.tensor*self.z_log_prec_std
      self.z_log_prec = log_prec
      self.z_prec = tf.exp( log_prec )
      self.z_var = 1.0/self.z_prec
      self.z_std = tf.sqrt(self.z_var)
      self.z_logvar = -log_prec
      #chi_sqr = tf.random_gamma(shape, self.z_nu/2.0, beta=0.5, dtype=tf.float32)
      #z = self.z_mu + self.z_std*u_z.tensor*tf.sqrt(self.z_nu/chi_sqr)
      z = self.z_mu + self.z_std*u_z.tensor

    # return generic data layer
    if output==0:
      return z
    elif output==1:
      return self.z_prec
    else:
      assert False, 'cant handle output'  
    #return [z, self.z_prec]
    
  def LogLikelihood( self, z, as_matrix = False, boolean_mask = None  ):
    assert z.__class__ == list, "must have list for this layer"
    #pdb.set_trace()
    # Gaussian part
    z_prec = tf.clip_by_value( z[1].tensor , 0.001, 1000.0 ) 
    #z_prec = z[1].tensor
    #tf.clip_by_value( self.log_prec_var[Z], 0.001, 10.0 ) 
    z_var  = 1.0 / z_prec
    z_logvar = tf.log( z_var )
    z_logprec = -z_logvar
     
    self.loglik_z = -0.5*np.log(2*np.pi) - 0.5*z_logvar - 0.5*tf.square( z[0].tensor-self.z_mu )/z_var
    # Lognormal part
    self.log_lik_prec =  -z_logprec - 0.5*np.log(2*np.pi) - 0.5*self.z_log_prec_logvar - 0.5*tf.square( z_logprec-self.z_log_prec_mu )/self.z_log_prec_var
    
    
    self.loglik_matrix =  self.loglik_z #+ self.log_lik_prec
    
    self.loglik = tf.reduce_sum(self.loglik_matrix, name = self.name+"_loglik")
    
    if as_matrix is True:
      return self.loglik_matrix
    else:
      return self.loglik

class GaussianLogNormalStaticLayer(GaussianLogNormalModelLayer):    
  def __init__( self, shape, prior, name="" ):

    self.weights = []
    self.biases = []

    self.shape           = shape
    self.batch_shape      = MakeBatchShape(shape)
    #self.mu               = prior[0]*np.ones( self.shape, dtype=np.float32 )
    #self.var              = prior[1]*np.ones( self.shape, dtype=np.float32 )
    
    
    self.mu               = prior[0]*np.ones( self.shape, dtype=np.float32 )
    self.log_prec_mu      = prior[1]*np.ones( self.shape, dtype=np.float32 )
    self.log_prec_var     = prior[2]*np.ones( self.shape, dtype=np.float32 )
    #self.activation       = [self.mu[Z], self.log_prec_mu[Z], self.log_prec_var[Z]]
    
    self.prior    = prior
    self.z_mu     = self.mu
    self.expectation = self.z_mu
    
    #self.z_var    = tf.clip_by_value( self.var[Z], 0.1, 1000.0 ) 
    self.z_log_prec_var    = tf.clip_by_value( self.log_prec_var, 0.1, 1000.0 ) 
    self.z_log_prec_mu = self.log_prec_mu
    
    self.z_log_prec_logvar = tf.log( self.z_log_prec_var )
    self.z_log_prec_std    = tf.sqrt( self.z_log_prec_var )

    self.name             = name
    self.output_dims      = [self.shape, self.shape, self.shape]
    self.tensor           = [self.z_mu, self.z_log_prec_mu, self.z_log_prec_var]
    
    
  def EvalWeights(self):
    return []
    
  def EvalBiases(self):
    return []

  def SetWeights( self, sess, weights ):
    pass
    
        
      
  def SetBiases( self, sess, biases ):
    pass
                
class GaussianStaticLayer(GaussianModelLayer):    
  def __init__( self, shape, prior, name="" ):
    self.shape           = shape
    self.batch_shape      = MakeBatchShape(shape)
    self.mu               = prior[0]*np.ones( self.shape, dtype=np.float32 )
    self.var              = prior[1]*np.ones( self.shape, dtype=np.float32 )
    
    self.z_mu = self.mu
    self.z_var = self.var
    
    self.prior    = prior
    self.expectation = self.z_mu
    
    self.z_logvar = tf.log( self.z_var )
    self.z_std    = tf.sqrt( self.z_var )
    
    self.weights = []
    self.biases = []
    
    self.name             = name
    self.output_dims      = [self.shape, self.shape]
    self.tensor           = [self.z_mu, self.z_var]
    
  def EvalWeights(self):
    return []
    
  def EvalBiases(self):
    return []

  def SetWeights( self, sess, weights ):
    pass
    
        
      
  def SetBiases( self, sess, biases ):
    pass
    
class LogNormalStudentModelLayer(StudentModelLayer):    
  def __init__( self, shape, model, prior = None, name="" ):
    self.shape           = shape
    self.batch_shape      = MakeBatchShape(shape)
    self.mu               = model[MU]
    self.log_prec_mu      = model[LOG_PREC_MU]
    self.log_prec_var     = model[LOG_PREC_VAR]
    self.activation       = [self.mu[Z], self.log_prec_mu[Z], self.log_prec_var[Z]]
    
    self.prior    = prior
    self.z_mu     = self.mu[Z]
    self.expectation = self.z_mu
    
    #self.z_var    = tf.clip_by_value( self.var[Z], 0.1, 1000.0 ) 
    self.z_log_prec_var    = tf.clip_by_value( self.log_prec_var[Z], 0.1, 1000.0 ) 
    self.z_log_prec_mu = self.log_prec_mu[Z] 
    
    self.z_log_prec_logvar = tf.log( self.z_log_prec_var )
    self.z_log_prec_std    = tf.sqrt( self.z_log_prec_var )
    
    # self.mu_weights       = self.mu[WEIGHTS]
    # self.var_weights      = self.var[WEIGHTS]
    # self.nu_weights      = self.nu[WEIGHTS]
    #self.penalties        = self.mu["penalties"] + self.var["penalties"]
    
    self.weights = [self.mu[WEIGHTS],self.log_prec_mu[WEIGHTS],self.log_prec_var[WEIGHTS]]
    self.biases = [self.mu[BIASES],self.log_prec_mu[BIASES],self.log_prec_var[BIASES]]
    
    self.name             = name
    self.output_dims      = [self.shape, self.shape, self.shape]
    self.tensor           = [self.z_mu, self.z_log_prec_mu, self.z_log_prec_var]
    
    # self.log_norm_const = tf.lgamma( (self.z_nu+1)/2.0 ) \
    #                     - tf.lgamma( self.z_nu/2.0 ) \
    #                     -0.5*tf.log( self.z_nu ) \
    #                     - 0.5*self.z_logvar - 0.5*np.log(np.pi)
        
            
  def Generate( self, u_z, shape, name = "", use_expectation = False  ):
    
    if use_expectation:
      z = self.expectation
    else:
      # generate z using deterministic transform
      log_prec = self.z_log_prec_mu + u_z.tensor*self.z_log_prec_std
      self.z_prec = tf.exp( log_prec )
      self.z_var = 1.0/self.z_prec
      self.z_std = tf.sqrt(self.z_var)
      self.z_logvar = -log_prec
      #chi_sqr = tf.random_gamma(shape, self.z_nu/2.0, beta=0.5, dtype=tf.float32)
      #z = self.z_mu + self.z_std*u_z.tensor*tf.sqrt(self.z_nu/chi_sqr)
      z = self.z_mu + self.z_std*u_z.tensor

    # return generic data layer
    return GeneratedDataLayer( shape, tensor=z, name=name )

  def GenerateX( self, u_zs, use_expectation = False  ):
    
    u_z = u_zs[0]
    u_prec = u_zs[1]
    if use_expectation:
      z = self.expectation
    else:
      # generate z using deterministic transform
      # generate z using deterministic transform
      log_prec = self.z_log_prec_mu + u_prec.tensor*self.z_log_prec_std
      self.z_prec = tf.exp( log_prec )
      self.z_var = 1.0/self.z_prec
      self.z_std = tf.sqrt(self.z_var)
      self.z_logvar = -log_prec
      #chi_sqr = tf.random_gamma(shape, self.z_nu/2.0, beta=0.5, dtype=tf.float32)
      #z = self.z_mu + self.z_std*u_z.tensor*tf.sqrt(self.z_nu/chi_sqr)
      z = self.z_mu + self.z_std*u_z.tensor

    # return generic data layer
    return z
    
  def LogLikelihood( self, z, as_matrix = False, boolean_mask = None  ):
    self.loglik_matrix = -0.5*np.log(2*np.pi) - 0.5*self.z_logvar - 0.5*tf.square( z.tensor-self.z_mu )/self.z_var
    self.loglik = tf.reduce_sum(self.loglik_matrix, name = self.name+"_loglik")
    
    if as_matrix is True:
      return self.loglik_matrix
    else:
      return self.loglik


                
class GaussianStaticLayer(GaussianModelLayer):    
  def __init__( self, shape, prior, name="" ):
    self.shape           = shape
    self.batch_shape      = MakeBatchShape(shape)
    self.mu               = prior[0]*np.ones( self.shape, dtype=np.float32 )
    self.var              = prior[1]*np.ones( self.shape, dtype=np.float32 )
    
    self.z_mu = self.mu
    self.z_var = self.var
    
    self.prior    = prior
    self.expectation = self.z_mu
    
    self.z_logvar = tf.log( self.z_var )
    self.z_std    = tf.sqrt( self.z_var )
    
    self.weights = []
    self.biases = []
    
    self.name             = name
    self.output_dims      = [self.shape, self.shape]
    self.tensor           = [self.z_mu, self.z_var]
    
  def EvalWeights(self):
    return []
    
  def EvalBiases(self):
    return []

  def SetWeights( self, sess, weights ):
    pass
    
        
      
  def SetBiases( self, sess, biases ):
    pass
    
class GaussianProductLayer(GaussianModelLayer):
  def __init__( self, shape, model, prior = None, name="" ):
    self.shape           = shape
    self.batch_shape      = MakeBatchShape(shape)
    self.mu               = model[MU]
    self.var              = model[VAR]
    #self.activation       = [self.mu[Z], self.var[Z]]
    
    self.prior    = prior
    self.z_mu     = self.mu[Z]
    self.expectation = self.z_mu
    
    self.z_var    = tf.clip_by_value( self.var[Z], 0.1, 1000.0 ) 
    self.z_logvar = tf.log( self.z_var )
    self.z_std    = tf.sqrt( self.z_var )
    
    #self.mu_weights       = self.mu[WEIGHTS]
    #self.var_weights      = self.var[WEIGHTS]
    #self.penalties        = self.mu["penalties"] + self.var["penalties"]
    
    #self.weights = [self.mu_weights,self.var_weights]
    #self.biases = [self.mu[BIASES],self.var[BIASES]]
    
    self.name             = name
    self.output_dims      = [self.shape, self.shape]
    self.tensor           = [self.z_mu, self.z_var]

  def GetVariance(self):
    return self.z_var
    
  def GetMean(self):
    return self.z_mu
    
  def EvalWeights(self):
    return []
    
  def EvalBiases(self):
    return []

  def SetWeights( self, sess, weights ):
    assert False, "no weights"
        
      
  def SetBiases( self, sess, biases ):
    assert False, "no biases"
  
class BetaModelLayer(HiddenLayer):
  def __init__( self, shape, model, name = ""):
    self.shape             = shape
    self.batch_shape       = MakeBatchShape(shape)
    self.model             = model
    self.weights           = model[WEIGHTS]
    self.weights_a         = self.weights[0]
    self.weights_b         = self.weights[1]
    
    self.biases             = model[BIASES]
    self.biases_a         = self.biases[0]
    self.biases_b         = self.biases[1]
    self.a           = self.model[A]
    self.b           = self.model[B]
    self.prior_a     = self.model[PRIOR][0]
    self.prior_b     = self.model[PRIOR][1]
    
    #self.expectation = (self.a + self.prior_a -1.0)/(self.b + self.prior_b + self.a + self.prior_a - 2.0)
    self.expectation = (self.a + self.prior_a)/(self.b + self.prior_b + self.a + self.prior_a)
    self.variance = (self.a + self.prior_a)*(self.b + self.prior_b)/( tf.square(self.b + self.prior_b + self.a + self.prior_a)*(self.b + self.prior_b + self.a + self.prior_a+1.0))
    self.std_dev = tf.sqrt( self.variance)
    self.name             = name
    self.tensor           = [self.a, self.b]
    
    #pdb.set_trace()
  
  def EvalWeights(self):
    wa = [w.eval() for w in self.weights_a]
    wb = [w.eval() for w in self.weights_b]
    wa.extend(wb) #[
    return wa #return wa.extend(wb) #[w[0].eval() for w in self.weights]
    
  def EvalBiases(self):
    if self.biases is None:
      return []
    if self.biases.__class__ == list:
      b = []
      for w in self.biases:
        if w is not None:
          b.append( w.eval())
      return b  
      #return [w.eval() for w in self.biases]
    else:
      return self.biases.eval()

  def SetWeights( self, sess, weights ):
    assert weights.__class__ == list, "should assign same weights"
    assert len(weights) == len(self.weights), "should assign same weights"
      
    for tf_w, np_w in zip( self.weights, weights ):
      sess.run(tf_w[0].assign(np_w))
        
      
  def SetBiases( self, sess, biases ):
    assert biases.__class__ == list, "should assign same biases"
    assert len(biases) == len(self.biases), "should assign same biases"
      
    for tf_w, np_w in zip( self.biases, biases ):
      sess.run(tf_w.assign(np_w))
          
  def LogLikelihood( self, X, as_matrix = False, boolean_mask = None ):
    
    self.loglik_matrix =           -tf_log_beta(self.a+self.prior_a, self.b+self.prior_b) \
                                      + (self.a + self.prior_a -1.0 )* tf.log( X.tensor + 1e-12 ) \
                                      + (self.b + self.prior_b -1.0 )* tf.log( 1.0 - X.tensor + 1e-12 )

    
    if boolean_mask is not None:
      self.loglik_matrix = tf.boolean_mask( self.loglik_matrix, boolean_mask )
      
    self.loglik = tf.reduce_sum(self.loglik_matrix ,name = self.name+"_loglik")
    if as_matrix is True:
      return self.loglik_matrix
    else:
      return self.loglik

  def Generate( self, u_z, shape, name = "", use_expectation = False  ):
    
    if use_expectation:
      z = self.expectation
    else:
      # generate z using deterministic transform
      assert False, "No generative model for Beta, use expectation" #z = self.z_mu + self.z_std*u_z.tensor

    # return generic data layer
    return GeneratedDataLayer( shape, tensor=z, name=name )

class BetaGivenModelLayer(BetaModelLayer):
  def __init__( self, shape, model, name = ""):
    self.model             = model

    self.a           = self.model[A]
    self.b           = self.model[B]
    self.prior_a     = self.model[PRIOR][0]
    self.prior_b     = self.model[PRIOR][1]

    self.shape             = shape
    self.batch_shape       = MakeBatchShape(shape)
    
    #self.expectation = (self.a + self.prior_a -1.0)/(self.b + self.prior_b + self.a + self.prior_a - 2.0)
    self.expectation = (self.a + self.prior_a)/(self.b + self.prior_b + self.a + self.prior_a)
    self.name             = name
    self.tensor           = [self.a, self.b]
    
    #pdb.set_trace()
  
  def EvalWeights(self):
    return []
    
  def EvalBiases(self):
    return []

  def SetWeights( self, sess, weights ):
    pass        
      
  def SetBiases( self, sess, biases ):
    pass
    
# class KumaModelLayer(HiddenLayer):
#   def __init__( self, shape, model, name = ""):
#     self.shape             = shape
#     self.batch_shape      = MakeBatchShape(shape)
#     self.model             = model
#     self.penalties         = model["penalties"]
#
#     self.log_a = tf.clip_by_value( self.model["log_a"], np.log(0.0001), np.log(100) )
#     self.log_b = tf.clip_by_value( self.model["log_b"], np.log(0.0001), np.log(100) )
#     self.a =  tf.exp(self.log_a)
#     self.b =  tf.exp(self.log_b)
#
#     self.log_expectation = self.log_b + tf.lgamma(1.0+1.0/self.a) + tf.lgamma(self.b) - tf.lgamma(1.0+1.0/self.a+self.b)
#     self.expectation = tf.exp( self.log_expectation )
#
#
#     self.name             = name
#     self.tensor           = [self.a, self.b]
#     self.output_dims      = [self.a.get_shape().dims[1].value, self.a.get_shape().dims[1].value]
#
#   def LogLikelihood( self, X, missing_model = None ):
#     #shape = layer.get_shape()
#     if missing_model is None:
#       self.loglik = tf.reduce_sum( tf.log(self.a) + tf.log(self.b) \
#                                     + (self.a -1.0 )* tf.log( X.tensor + 1e-12 ) \
#                                     + (self.b -1.0 )* tf.log( 1.0 - tf.pow(X.tensor,self.a)+1e-12), name = self.name+"_loglik" )
#
#     else:
#       assert False, "not implemented"
#       if missing_model.type == "full":
#         binary_observed_vector = missing_model.observed
#         self.loglik_by_case = tf.reduce_sum( -tf_log_beta(self.a+self.prior_a, self.b+self.prior_b) \
#                                       + (self.a + self.prior_a -1.0 )* tf.log( X.tensor + 1e-12 ) \
#                                       + (self.b + self.prior_b -1.0 )* tf.log( 1.0 - X.tensor + 1e-12 ), -1, name = self.name+"_loglik_by_case" )
#         while self.loglik_by_case.get_shape().ndims > 1:
#           self.loglik_by_case = tf.reduce_sum( self.loglik_by_case, -1, name = self.name+"_loglik_by_case" )
#         self.loglik = tf.reduce_sum( tf.mul( binary_observed_vector, self.loglik_by_case ), name = self.name+"_loglik" )
#       else:
#         raise NotImplemented, "no implementation for " + missing_model.type
#
#     return self.loglik
#
#   def LogLikelihoodAsSigmoid( self, X ):
#     return tf.reduce_sum( X.tensor * tf.log(self.expectation+1e-6) + ( 1.0 - X.tensor ) * tf.log(1.0-self.expectation+1e-6) )
#
#
#   def Generate( self, u_z, shape, name = "", use_expectation = False  ):
#
#     if use_expectation:
#       #assert False
#       z = self.expectation
#
#     else:
#       #z= tf.pow( 1.0-tf.pow(1.0-u_z.tensor, 1.0/self.b), 1.0/self.a)
#       #z= tf.pow( 1.0-tf.pow(1.0-u_z.tensor, 1.0/self.b), 1.0/self.a)
#       z = tf.exp( (1.0/self.b)*tf.log( 1.0- tf.exp( tf.log(1.0-u_z.tensor)/self.b))  )
#       #exp( ( 1.0/b )*log(1.0-exp( log(u)/b)))
#       # generate z using deterministic transform
#       #assert False, "No generative model for Beta, use expectation" #z = self.z_mu + self.z_std*u_z.tensor
#
#     # return generic data layer
#     return GeneratedDataLayer( shape, tensor=z, name=name )


class SigmoidModelLayer(HiddenLayer):
  def __init__( self, shape, model, name = ""):
    self.shape             = shape
    self.batch_shape      = MakeBatchShape(shape)
    #self.inputs            = input_layers
    self.model             = model
    #self.penalties         = model["penalties"]
    self.weights           = self.model[WEIGHTS]
    self.biases            = self.model[BIASES]
    
    if self.model.has_key(EPSILON):
      self.gen_epsilon = self.model[EPSILON]
    else:
      self.gen_epsilon = 0.01
    # the output is prob(c=1|x)
    self.p_of_c = self.model["prob"]
    self.p_of_c_not = 1.0 - self.p_of_c
    #self.n_units          = n_units
    #self.shape             = model["shape"]
    #self.dims              = [model["shape"][1:]]
    self.name              = name
    self.tensor            = self.p_of_c
    self.expectation       = self.p_of_c
    
    self.log_p_of_c     = tf.log( self.p_of_c + 1e-12 )
    self.log_p_of_c_not = tf.log( self.p_of_c_not + 1e-12 )
    
  def GetExpectation(self):
    return self.expectation
    
  def LogLikelihood( self, X, as_matrix = False ):
    self.loglik_matrix = X.tensor * self.log_p_of_c + ( 1.0 - X.tensor ) * self.log_p_of_c_not
    
    self.loglik = tf.reduce_sum( self.loglik_matrix, name = self.name+"_loglik" )
    
    if as_matrix is True:
      return self.loglik_matrix
    else:
      return self.loglik

  def LogLikelihoodAsSigmoid(self, X ):  
    return self.LogLikelihood(X)
    
  def Generate( self, u_z, shape, name = "", use_expectation = False  ):
    
    if use_expectation:
      z = self.expectation
    else:
      # e.g. p = 0.2, u = 0.9 -> 0, ceil( 0.2-0.9 ) => ceil( -0.7 ) => 0
      #z = tf.maximum( 0, tf.minimum( 1.0, self.tensor - u_z.tensor ) )
      z = tf.sigmoid( (self.tensor - u_z.tensor)/self.gen_epsilon )
    # return generic data layer
    return GeneratedDataLayer( shape, tensor=z, name=name )

  def KL( self ):
    #self.latent_kl_by_case = tf.reduce_sum( tf.square( self.z_mu) + tf.square(1.0-self.z_var), -1)
    #self.latent_kl_by_case = -0.5 * tf.reduce_sum(1 + self.z_logvar - tf.square(self.z_mu) - self.z_var, -1)
    self.prior = 0.5
    
    #self.log_p_of_c 
    self.latent_kl = tf.reduce_sum( self.p_of_c * self.log_p_of_c +  self.p_of_c_not * self.log_p_of_c_not + np.log(2.0) )
    return self.latent_kl

class SoftmaxModelLayer(HiddenLayer):
  def __init__( self, shape, model, name = ""):
    self.shape             = shape
    self.batch_shape      = MakeBatchShape(shape)
    self.model             = model
    self.weights           = self.model[WEIGHTS]
    self.biases            = self.model[BIASES]
    
    if self.model.has_key(EPSILON):
      self.gen_epsilon = self.model[EPSILON]
    else:
      self.gen_epsilon = 0.01
    # the output is prob(c=1|x)
    self.p_of_c = self.model["prob"]
    self.p_of_c_not = 1.0 - self.p_of_c
    self.name              = name
    self.tensor            = self.p_of_c
    self.expectation       = self.p_of_c
    
    self.log_p_of_c     = tf.log( self.p_of_c + 1e-12 )
    self.log_p_of_c_not = tf.log( self.p_of_c_not + 1e-12 )
    
  # def LogLikelihood( self, X ):
  #   self.loglik = tf.reduce_sum( X.tensor * self.log_p_of_c, name = self.name+"_loglik" )
  #   return self.loglik
  #
  def LogLikelihood( self, X, as_matrix = False, boolean_mask = None ):
    if boolean_mask is None:
      self.loglik_matrix = X.tensor * self.log_p_of_c
    else:
      self.loglik_matrix = tf.boolean_mask( X.tensor, boolean_mask ) * tf.boolean_mask( self.log_p_of_c, boolean_mask ) 

    self.loglik = tf.reduce_sum( self.loglik_matrix, name = self.name+"_loglik" )

    if as_matrix is True:
     return self.loglik_matrix
    else:
     return self.loglik   

      