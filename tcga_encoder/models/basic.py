from __future__ import print_function
import pdb
import tensorflow as tf
# Parameters
learning_rate = 0.1
training_epochs = 60
batch_size = 100
display_step = 10


def logistic_regression( x_train, y_train, x_test, y_test, l1 = 0.0, l2 = 0.0 ):
  n,d = x_train.shape
  
  # tf Graph Input
  x = tf.placeholder(tf.float32, [None, d], name="x") # mnist data image of shape 28*28=784
  y = tf.placeholder(tf.float32, [None, ], name = "y") # 0-9 digits recognition => 10 classes

  # Set model weights
  W = tf.Variable(tf.zeros([d, 1]), name="W")
  b = tf.Variable(tf.zeros([1]), name="b")

  # Construct model
  pred = tf.nn.sigmoid(tf.matmul(x, W) + b) # Softmax

  weight_penalty = 0
  if l1 > 0:
    weight_penalty += l1*tf.reduce_sum( tf.abs( W ))
  if l2 > 0:
    weight_penalty += l2*tf.reduce_sum( tf.square( W ))
    
  # Minimize error using cross entropy
  cost = -tf.reduce_sum( y*tf.log(pred+1e-12) ) -  tf.reduce_mean( (1-y)*tf.log(1.0-pred+1e-12) ) 
  cost += weight_penalty
  # Gradient Descent
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

  # Test model
  correct_prediction = tf.equal( tf.cast(pred, tf.int32), tf.cast(y, tf.int32) )
  
  #pdb.set_trace()
  # Calculate accuracy
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  
  # Initializing the variables
  init = tf.global_variables_initializer()

  # Launch the graph
  with tf.Session() as sess:
      sess.run(init)

      # Training cycle
      for epoch in range(training_epochs):
          avg_cost = 0.
          total_batch = int(n/batch_size)
          _, c = sess.run([optimizer, cost], feed_dict={x: x_train,
                                                        y: y_train})
          avg_cost += c / n
          # Display logs per epoch step
          if (epoch+1) % display_step == 0:
              print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

      print("Optimization Finished!")
      #print(y_train)
      #print(sess.run( [y], feed_dict={x: x_train}))
      #print(pred.eval( feed_dict={x: x_train}))
      acc = accuracy.eval({x: x_train, y: y_train})
      test_acc = accuracy.eval({x: x_test, y: y_test})
      test_pred = pred.eval({x: x_test, y: y_test})
      print("Train Accuracy:", acc )
      print("Test Accuracy:", test_acc )
      W = W.eval()
      b = b.eval()
  return test_acc, W, b
  
# if __name__ == "__main__":
#
#   disease = "sarc"
#   data_file = "pan_tiny_multi_set"
#   experiment_name = "tiny_leave_blca_out"
#
#   if len(sys.argv) == 4:
#     disease   = sys.argv[1]
#     data_file = sys.argv[2]
#     experiment_name = sys.argv[3]
#
#   data_location = os.path.join( HOME_DIR, "data/broad_processed_post_recomb/20160128/%s/data.h5"%(data_file) )
#   #survival_location = os.path.join( HOME_DIR, "results/tcga_vae_post_recomb/%s/full_vae_survival.h5"%(experiment_name) )
#   survival_location = os.path.join( HOME_DIR, "Dropbox/SYNC/brca_large_full_vae_survival.h5" )
#   #s = pd.HDFStore( survival_file, "r" )
#   #S1 = s["/%s/split1"%(disease)]
#   f1 = pp.figure()
#   ax0 = f1.add_subplot(111)
#   f = pp.figure()
#   ax1 = f.add_subplot(211)
#   ax2 = f.add_subplot(212)
#   K=10
#   penalties = ["l2","l1"]
#   Css = [[ 1.0,0.9,0.75,0.5,0.1,0.01, 0.001, 0.0001, 0.00001],[5.0,2.0,1.0,0.1,0.01]]
#   best_values = OrderedDict()
#   mn_models = OrderedDict()
#   axs = [ax1,ax2]
#   for penalty_idx, penalty,Cs in zip( range(2), penalties,Css ):
#     best_values[ penalty ] = []
#     for C  in Cs:
#
#       test_accuracy, models, y, test_predictions, test_prob, test_log_prob, test_auc, feature_names  = compress_survival_prediction( disease, data_location, survival_location, K, penalty, C )
#       print "%s %d-fold auc = %0.3f accuracy = %0.3f, log prob = %0.3f (C = %f, reg = %s)"%( disease, K, test_auc, test_accuracy, test_log_prob, C, penalty )
#       best_values[ penalty ].append([test_accuracy,test_log_prob,test_auc])
#
#       I = np.argsort(y)
#       ax0.plot( test_prob[I], 'o-')
#     best_values[ penalty ] = np.array(best_values[ penalty ], dtype=float )
#
#     best_idx = np.argmin( best_values[ penalty ][:,0] )
#
#     best_C = Cs[ best_idx ]
#
#     test_accuracy, models, y, test_predictions, test_prob, test_log_prob, test_auc, feature_names  = compress_survival_prediction( disease, data_location, survival_location, K, penalty, best_C )
#
#     mn_models[ penalty ] = np.zeros(models[0].coef_.shape[1])
#     for m in models:
#       axs[penalty_idx].plot( m.coef_.T, 'o-' )
#       mn_models[ penalty ] += np.squeeze(m.coef_.T)
#   pp.show()
  