from tcga_encoder.utils.helpers import *

import tensorflow as tf
from tcga_encoder.models.layers import *
from tcga_encoder.models.regularizers import *
from tcga_encoder.definitions.nn import *
#from utils.logging import *
#import pdb

def PrepareCallbacks( callback_list ):
  call_backs      = []
  call_back_rates = []
  for cb in callback_list:
    call_backs.append( cb[NAME])
    call_back_rates.append( cb[RATE] )
  return call_backs, call_back_rates


def train( sess, network, algo_dict, data_dict, logging_dict, results_dict ):

  n_epochs            = int(algo_dict[EPOCHS])

  # -------------------------------------------------- #
  # set-up network cost functions and learning rates   #
  # -------------------------------------------------- #
  # train_ops                  = OrderedDict()
  # learning_rate_placeholders = OrderedDict()
  # call_backs                 = OrderedDict()
  # call_back_rates            = OrderedDict()
  # batch_callbacks            = OrderedDict()
  # batch_id_generators        = OrderedDict()
  # cb_infos                   = OrderedDict()
  # test_feed_dicts            = OrderedDict()

  #results_dict[ "CALL_BACKS" ] = cb_infos

  # -------------------------------------------------- #
  # SET-UP NETWORK'S PARAMS                            #
  # -------------------------------------------------- #
  cb_info = OrderedDict()

  p                   = algo_dict
  batch_size          = int(p[BATCH_SIZE])
  learning_rate       = float(p[LEARNING_RATE])
  learning_rate_decay = float(p[LEARNING_RATE_DECAY])
  min_learning_rate   = float(p[MIN_LEARNING_RATE])
  optimizer           = p[OPTIMIZER]
  #batch_maker         = p[BATCH_CALLBACK]
  batcher = p[BATCHER]

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

  call_backs, call_back_rates = PrepareCallbacks( p[CALL_BACKS] )
  n_train             = int(data_dict[N_TRAIN])

  batch_id_generator  = batch_ids_maker( batch_size, n_train )
  batcher.batch_id_generator = batch_id_generator
  test_feed_imputation = batcher.TestBatch()
  test_feed_dict = {}
  network.FillFeedDict( test_feed_dict, test_feed_imputation )

  val_feed_imputation = batcher.ValBatch()
  val_feed_dict = {}
  network.FillFeedDict( val_feed_dict, val_feed_imputation )

  #test_feed_dicts[name]                = test_feed_dict
  cb_info[TEST_FEED_DICT]       = test_feed_dict
  cb_info[TEST_FEED_IMPUTATION] = test_feed_imputation
  cb_info[VAL_FEED_DICT]       = val_feed_dict
  cb_info[VAL_FEED_IMPUTATION] = val_feed_imputation

  # -------------------------------------------------- #
  # TRAIN                                              #
  # -------------------------------------------------- #
  print "Running : init = tf.global_variables_initializer()"
  init = tf.global_variables_initializer()
  #init = tf.initialize_all_variables()
  print "Running : sess.run(init)"
  sess.run(init)
  batcher.InitializeAnythingYouWant( sess, network )
  
  print "Running : for epoch in range(n_epochs):"
  for epoch in range(n_epochs):
    
    
    # -------------------------------------------------- #
    # BATCH SET-UP                                       #
    # -------------------------------------------------- #
    batch_feed_dict = {}
    batch_ids             = batch_id_generator.next()
    batch_feed_imputation = batcher.NextBatch(batch_ids)


    cb_info[EPOCH]                 = epoch+1
    cb_info[BATCH_FEED_DICT]       = batch_feed_dict
    cb_info[BATCH_FEED_IMPUTATION] = batch_feed_imputation
    cb_info[BATCH_IDS]             = batch_ids #batch_feed_imputation["batch_ids"]
    
    network.FillFeedDict( batch_feed_dict, batch_feed_imputation )
    batch_feed_dict[learning_rate_placeholder] = current_learning_rate
    batcher.DoWhatYouWantAtEpoch( sess, epoch, network,cb_info )
    # -------------------------------------------------- #
    # TRAIN STEP                                         #
    # -------------------------------------------------- #
    train_op_eval = sess.run( train_op, feed_dict = batch_feed_dict )

    # -------------------------------------------------- #
    # CALLBACKS                                          #
    # -------------------------------------------------- #
    for cb_idx in pp.find( np.mod(epoch+1,call_back_rates)==0 ):
      if call_backs[cb_idx] == LEARNING_DECAY:
        print "** Decreasing learning rate"
        current_learning_rate *= algo_dict[LEARNING_RATE_DECAY]
        current_learning_rate = min(current_learning_rate,algo_dict[MIN_LEARNING_RATE])
      else:
        batcher.CallBack( call_backs[cb_idx], sess, cb_info )
