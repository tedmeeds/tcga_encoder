import tensorflow as tf
from tcga_encoder.utils.helpers import *
from tcga_encoder.data.data import *
from tcga_encoder.definitions.tcga import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.definitions.locations import *
#from tcga_encoder.algorithms import *
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("talk")

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def add_variables( var_dict, data_dict ):
  # add very specific numbers:
  var_dict["x_dim"]    = data_dict['n_dim']
  var_dict["tissue_dim"]    = 33 #data_dict['tissue_dim']
  var_dict["t_dim"]   = 1
  var_dict["e_dim"]    = 1
  
def load_architecture( arch_dict, data_dict ):
  add_variables( arch_dict[VARIABLES], data_dict )
  return arch_dict[NETWORK]( arch_dict, data_dict)

def extract_data_dict( train ):
  Z_train = train[0]
  T_train = train[1]
  E_train = train[2]
  
  data_dict = {}
  data_dict[ N_TRAIN ] = len( Z_train )
  data_dict[ "n_dim" ] = Z_train.shape[1]

  return data_dict

def train( sess, network, spec, data_dict, train_data, test_data ):
  algo_dict = spec[ALGORITHM]
  n_epochs            = int(algo_dict[EPOCHS])

  # -------------------------------------------------- #
  # SET-UP NETWORK'S PARAMS                            #
  # -------------------------------------------------- #
  cb_info = OrderedDict()

  p                   = algo_dict
  batch_size          = int(p[BATCH_SIZE])
  learning_rate       = float(p[LEARNING_RATE])
  #learning_rate_decay = float(p[LEARNING_RATE_DECAY])
  #min_learning_rate   = float(p[MIN_LEARNING_RATE])
  optimizer           = p[OPTIMIZER]
  #batch_maker         = p[BATCH_CALLBACK]
  #batcher = p[BATCHER]
  logging_frequency = p["logging_frequency"]
  current_learning_rate = learning_rate

  data_cost = network.CostToMinimize()
  learning_rate_placeholder = tf.placeholder_with_default( learning_rate, [], name=LEARNING_RATE)

  gradient_noise_scale = None
  if p.has_key("gradient_noise_scale"):
    # gradient_noise_scale = p["gradient_noise_scale"]
    # optimizer = optimizer( learning_rate=learning_rate_placeholder )
    # #.minimize(data_cost)
    # gradients = optimizer.compute_gradients(data_cost)
    #
    # pdb.set_trace() #noisy_gradients = gradients +
    # #train_op = optimizer.apply_gradients(gradients)

    train_op =tf.contrib.layers.optimize_loss( \
                             loss=data_cost, \
                             global_step=None,\
                             learning_rate=learning_rate_placeholder, \
                             optimizer='Adam', \
                             gradient_noise_scale=gradient_noise_scale )
    # , gradient_multipliers=None, clip_gradients=None, learning_rate_decay_fn=None, update_ops=None, variables=None, name=None, summaries=None)

  elif p.has_key("gradient_clipping"):
    print "USING GRADIENT CLIPPING!!!"
    optimizer = optimizer( learning_rate=learning_rate_placeholder )
    gvs = optimizer.compute_gradients(data_cost)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)
    
  else:
    train_op  = optimizer( learning_rate=learning_rate_placeholder ).minimize(data_cost)

  #call_backs, call_back_rates = PrepareCallbacks( p[CALL_BACKS] )
  n_train             = int(data_dict[N_TRAIN])

  batch_id_generator  = batch_ids_maker( batch_size, n_train )
  #batcher.batch_id_generator = batch_id_generator
  test_feed_imputation = network.TestBatch( test_data )
  test_feed_dict = {}
  network.FillFeedDict( test_feed_dict, test_feed_imputation )

  #val_feed_imputation = batcher.ValBatch()
  #val_feed_dict = {}
  #network.FillFeedDict( val_feed_dict, val_feed_imputation )

  #test_feed_dicts[name]                = test_feed_dict
  #cb_info[TEST_FEED_DICT]       = test_feed_dict
  #cb_info[TEST_FEED_IMPUTATION] = test_feed_imputation
  #cb_info[VAL_FEED_DICT]       = val_feed_dict
  #cb_info[VAL_FEED_IMPUTATION] = val_feed_imputation

  # -------------------------------------------------- #
  # TRAIN                                              #
  # -------------------------------------------------- #
  print "Running : init = tf.global_variables_initializer()"
  init = tf.global_variables_initializer()
  #init = tf.initialize_all_variables()
  print "Running : sess.run(init)"
  sess.run(init)
  #batcher.InitializeAnythingYouWant( sess, network )
  
  print "Running : for epoch in range(n_epochs):"
  for epoch in range(n_epochs):
    
    # -------------------------------------------------- #
    # BATCH SET-UP                                       #
    # -------------------------------------------------- #
    batch_feed_dict = {}
    batch_ids             = batch_id_generator.next()
    batch_feed_imputation = network.NextBatch(batch_ids, train_data )

    network.FillFeedDict( batch_feed_dict, batch_feed_imputation )
    batch_feed_dict[learning_rate_placeholder] = current_learning_rate

    # -------------------------------------------------- #
    # TRAIN STEP                                         #
    # -------------------------------------------------- #
    train_op_eval = sess.run( train_op, feed_dict = batch_feed_dict )

    if np.mod(epoch+1,logging_frequency)==0:
      train_cost = network.ComputeCost( sess, train_data )
      test_cost  = network.ComputeCost( sess, test_data )
      print "Epoch %d  train cost = %0.3f    test_cost = %0.3f"%(epoch+1, train_cost, test_cost)
      network.PrintModel()
      
    # -------------------------------------------------- #
    # CALLBACKS                                          #
    # -------------------------------------------------- #
    #cb_info[EPOCH]                 = epoch+1
    #cb_info[BATCH_FEED_DICT]       = batch_feed_dict
    #cb_info[BATCH_FEED_IMPUTATION] = batch_feed_imputation
    #cb_info[BATCH_IDS]             = batch_ids #batch_feed_imputation["batch_ids"]
    #for cb_idx in pp.find( np.mod(epoch+1,call_back_rates)==0 ):
    #  if call_backs[cb_idx] == LEARNING_DECAY:
    #    print "** Decreasing learning rate"
    #    current_learning_rate *= algo_dict[LEARNING_RATE_DECAY]
    #    current_learning_rate = min(current_learning_rate,algo_dict[MIN_LEARNING_RATE])
    #  else:
    #    batcher.CallBack( call_backs[cb_idx], sess, cb_info )
          
  
def main( train_data, val_data, spec ):
  y = spec #  = load_yaml( yaml_file)
  
  logging_dict = {}
  arch_dict = spec[ARCHITECTURE]
  data_dict = extract_data_dict( train_data )
  network = load_architecture( arch_dict, data_dict )
  network_name = arch_dict[NAME]

  sess = tf.InteractiveSession()
  
  
  train( sess, network, spec, data_dict, train_data, val_data )
  
  return network, sess

######################################################################################################
if __name__ == "__main__":
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_file = sys.argv[1]
  print "Running: ",yaml_file
  
  
    
  main( train, val,  load_yaml( yaml_file) )

  
  
  