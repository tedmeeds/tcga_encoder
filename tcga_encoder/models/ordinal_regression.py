from __future__ import print_function
import pdb
import tensorflow as tf
# Parameters
learning_rate = 0.1
training_epochs = 500
batch_size = 100
display_step = 100
import pylab as pp
import numpy as np

def ordinal_regression( x_train, e_train, t_train, l1 = 0.0, l2 = 0.0 ):
  # e_train: binary vector indicating left-censored event
  # t_train: time stamps for event or censor
  # 
  n,d = x_train.shape
  
  # tf Graph Input
  x = tf.placeholder(tf.float32, [None, d], name="x") # mnist data image of shape 28*28=784
  e = tf.placeholder(tf.float32, [None, ], name = "event") # 0-9 digits recognition => 10 classes
  t = tf.placeholder(tf.float32, [None, ], name = "time") # 0-9 digits recognition => 10 classes

  # Set model weights
  W = tf.Variable(tf.zeros([d, 1]), name="W")
  b = tf.Variable(tf.zeros([1]), name="b")

  # Construct model
  log_pred = tf.matmul(x, W) + b
  #pred = e*tf.nn.sigmoid( t - tf.matmul(x, W) - b ) + ()

  weight_penalty = 0
  if l1 > 0:
    weight_penalty += l1*tf.reduce_mean( tf.abs( W ))
  if l2 > 0:
    weight_penalty += l2*tf.reduce_mean( tf.square( W ))
    
  # Minimize error using cross entropy
  event_pred = tf.nn.sigmoid( t - log_pred )
  censor_pred = tf.nn.sigmoid( log_pred - t )
  I_censored = pp.find( e_train == 0)
  max_t = max(t_train)
  
  max_dif = max_t - t_train
  #pdb.set_trace()
  #cost = -tf.reduce_mean( e*tf.log(event_pred+1e-12) +  (1-e)*tf.log(censor_pred+1e-12) ) 
  cost = -tf.reduce_mean( tf.log(event_pred+1e-12) ) # + (1-e)*tf.log( 1.0-event_pred+1e-12) ) 
  cost += weight_penalty
  # Gradient Descent
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

  # Test model
  #correct_prediction = tf.equal( tf.cast(pred, tf.int32), tf.cast(y, tf.int32) )
  
  #pdb.set_trace()
  # Calculate accuracy
  #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  
  # Initializing the variables
  init = tf.global_variables_initializer()

  # Launch the graph
  with tf.Session() as sess:
      sess.run(init)

      # Training cycle
      for epoch in range(training_epochs):
          avg_cost = 0.
          total_batch = int(n/batch_size)
          
          predicted = np.squeeze( log_pred.eval( feed_dict={x: x_train[I_censored]} ) )
          
          # set death to censored between their censor and max time
          t_batch = t_train #.copy()
          r_vals = np.random.rand(len(I_censored))
          
          I_more = pp.find( predicted > t_train[I_censored] )
          #t_batch[ I_censored[I_more] ] += r_vals[I_more]*(predicted[I_more]-t_train[I_censored[I_more]] )
          t_batch[ I_censored ] += 0.5*r_vals*max_dif[I_censored]
          #pdb.set_trace()
          _, c = sess.run([optimizer, cost], feed_dict={x: x_train,
                                                        e: e_train,
                                                        t: t_batch})
          avg_cost += c / n
          # Display logs per epoch step
          if (epoch+1) % display_step == 0:
              print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

      print("Optimization Finished!")
      #print(y_train)
      #print(sess.run( [y], feed_dict={x: x_train}))
      #print(pred.eval( feed_dict={x: x_train}))
      #acc = accuracy.eval({x: x_train, y: y_train})
      #test_acc = accuracy.eval({x: x_test, y: y_test})
      #test_pred = log_pred.eval({x: x_test, e: e_test, t: t_test })
      #test_error = np.mean( np.square( test_pred - t_test))
      #print("Train Accuracy:", acc )
      #print("Test Error:", test_error )
      W = W.eval()
      b = b.eval()
  return W, b

  