from __future__ import print_function
import numpy as np
from tcga_encoder.utils.helpers import *

import tensorflow as tf
from tcga_encoder.models.layers import *
from tcga_encoder.models.regularizers import *
from tcga_encoder.definitions.nn import *
from tcga_encoder.models.networks import *
import pylab as pp

import pdb

def make_data( E_val, T_val, Z_val, bootstrap = False ):
  if bootstrap is True:
    ids = np.squeeze( make_bootstraps( np.arange(len(E_val)),1) )
    
  
    Z = Variable( torch.FloatTensor( Z_val[ids,:] ) )
    E = Variable( torch.FloatTensor( E_val[ids] ) )
    T = Variable( torch.FloatTensor( T_val[ids] ) )
  else:
    Z = Variable( torch.FloatTensor( Z_val ) )
    E = Variable( torch.FloatTensor( E_val ) )
    T = Variable( torch.FloatTensor( T_val) )
  return E,T,Z
  

class WeibullSurvivalModel(NeuralNetwork):
    def __init__(self, arch_dict, data_dict ):
      self.layers   = OrderedDict()
      self.dropouts = OrderedDict()
    
      self.use_matrix = True
    
      self.arch = arch_dict
      
      self.data_dict = data_dict
        
      self.dim       = self.data_dict["n_dim"]
      self.layer_names      = self.BuildLayers( arch_dict, data_dict, arch_dict[VARIABLES] )
      self.weight_penalties = self.ApplyRegularizers( arch_dict[REGULARIZERS], arch_dict[VARIABLES] )
      if len(self.weight_penalties)>0:
        self.weight_penalty = tf.add_n( self.weight_penalties )
      else:
        self.weight_penalty = 0.0
      
      
      self.loglikes_data_as_matrix  = self.BuildLoglikelihoods( arch_dict[DATA_LOGLIK],    data_dict, arch_dict[VARIABLES], as_matrix = self.use_matrix )
      self.loglikes_data  = self.OrderedDictOp( tf.reduce_sum, self.loglikes_data_as_matrix )
      self.log_p_x  = tf.add_n( self.loglikes_data.values(), name = "log_p_x" )
      #self.lower_bound = self.log_p_x_given_z - self.beta*tf.maximum(self.free_bits, self.log_q_z - self.log_p_z )
      self.batch_log_tensors = [self.log_p_x]
      self.batch_log_columns = ["Epoch","log p(x)"]
    
    def CostToMinimize(self):
      return -self.log_p_x + self.weight_penalty

    def ImputeAndFill( self, E, T, X, tissue ):
      feed_dict = {}
      self.FillFeedDict( feed_dict, {"X_input":X, "Times":T.reshape( (len(T),1)), "Events":E.reshape( (len(E),1)), "tissue_input":tissue} )
      return feed_dict


    def TestBatch( self, data ):
      X = data[0]
      T = data[1].reshape( (len(data[1]),1))
      E = data[2].reshape( (len(data[2]),1))
      tissue=data[3]
      imputation_dict = {"X_input":X, "Times":T, "Events":E, "tissue_input":tissue}
      return imputation_dict
      
    def NextBatch( self, batch_ids, data ):
      X = data[0][ batch_ids, :]
      T = data[1][ batch_ids].reshape( (len(batch_ids),1))
      E = data[2][ batch_ids].reshape( (len(batch_ids),1))
      tissue=data[3][batch_ids,:]
      imputation_dict = {"X_input":X, "Times":T, "Events":E, "tissue_input":tissue}
      return imputation_dict
      
    def ComputeCost( self, sess, data ):
      n = len(data[0])
      impute = self.TestBatch( data )
      feed_dict = {}
      self.FillFeedDict( feed_dict, impute )
      
      cost_eval = sess.run( self.log_p_x, feed_dict=feed_dict )
      return -cost_eval/n
      
    def GetWeights(self, sess ):
      layer_scale = self.GetLayer( "scale" )
      layer_shape = self.GetLayer( "shape" ) 
      
      w_scale = layer_scale.EvalWeights(  )
      w_shape = layer_shape.EvalWeights(  )
      
      return np.hstack( (np.squeeze(w_scale),np.squeeze(w_shape)))
     
    def LogTime( self, sess, X, tissue, at_time = 0.5 ):
      log_at_time = np.log(at_time)
      
      model_layer = self.GetLayer( "survival_model" )

      layer_scale = self.GetLayer( "scale" )
      layer_shape = self.GetLayer( "shape" ) 
      
      impute = {"X_input":X, "tissue_input":tissue} #self.TestBatch( data )
      feed_dict = {}
      self.FillFeedDict( feed_dict, impute )
      
      [scale_eval, shape_eval] = self.sess.run( [layer_scale.tensor, layer_shape.tensor ], feed_dict = feed_dict )
      
      #pdb.set_trace()
      log_t = np.log( -log_at_time / shape_eval ) / scale_eval
      
      return log_t

    def Survival( self, T, X ):
      return np.exp( - self.CumulativeHazard( T, X ) )
    

    def CumulativeHazard( self, T, X ):
      return np.exp( self.LogCumulativeHazard( T, X ) )
      
    def LogCumulativeHazard( self, T, X ):
      layer_scale = self.GetLayer( "scale" )
      layer_shape = self.GetLayer( "shape" ) 
      
      feed_dict = self.ImputeAndFill( np.array([]), T, X )

      [scale_eval, shape_eval] = self.sess.run( [layer_scale.tensor, layer_shape.tensor ], feed_dict = feed_dict )
      
      #pdb.set_trace()
      return np.log( shape_eval + 1e-12 ) + scale_eval*np.log(T.reshape((len(T),1)))
      #return self.LogShape(X) + self.Scale(X)*torch.log(T)

    
    def PrintModel( self ):
      layer_scale = self.GetLayer( "scale" )
      layer_shape = self.GetLayer( "shape" ) 
      
      w_scale = np.squeeze( layer_scale.EvalWeights() )
      b_scale = np.squeeze( layer_scale.EvalBiases() )
      
      w_shape = np.squeeze( layer_shape.EvalWeights() )
      b_shape = np.squeeze( layer_shape.EvalBiases() )
      
      a_str = ""
      b_str = ""
      for i in range( min(10,len(w_scale))):
        a_str += "%0.3f "%(w_scale[i])
        b_str += "%0.3f "%(w_shape[i])
        
      print( "    Alpha0 %0.3f log alpha: %s"%(np.exp( b_scale ),a_str))
      print( "    Beta0 %0.3f log beta: %s"%(np.exp( b_shape),b_str))
      #pdb.set_trace()
      
    def PlotSurvival( self, E, T, X, tissue, ax = None, color = "k" ):
      print( "Running PlotSurvival()")
      if ax is None:
        f = pp.figure()
        ax = f.add_subplot(111)    
      times = np.linspace( 1.0, max(T), 100 )

      s = self.Survival( T, X )   
   
      for xi, si, ti in zip( X, s, T ):
        #pdb.set_trace()
        s_series = self.Survival( times, np.tile(xi,( len(times),1) ) )
        
        #pdb.set_trace()
        ax.plot( times, s_series, color+'-', lw=1, alpha = 0.5 )
  
      base_s_series = self.Survival( times, np.tile(0*xi,( len(times),1) ) )
      ax.plot( times, base_s_series, 'm-', lw=4, alpha = 0.75 )

      events = pp.find( E )
      censors = pp.find(1-E)  
      ax.plot(T[events], s[events], 'ro')
      ax.plot(T[censors], s[censors], 'cs')
      print( "Done PlotSurvival()")
      return ax


if __name__ == "__main__":
  from tcga_encoder.utils.helpers import *
  from tcga_encoder.data.data import *
  from tcga_encoder.definitions.tcga import *
  from tcga_encoder.definitions.nn import *
  from tcga_encoder.definitions.locations import *
  from tcga_encoder.models.survival_analysis import *
  #from tcga_encoder.algorithms import *
  import seaborn as sns
  import pdb
  from sklearn.metrics import accuracy_score
  from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
  from lifelines.statistics import logrank_test
  sns.set_style("whitegrid")
  sns.set_context("talk")
  from tcga_encoder.models.pytorch.bootstrap_linear_regression import BootstrapLinearRegression, BootstrapLassoRegression

  pd.set_option('display.max_columns', 500)
  pd.set_option('display.width', 1000)
  
  assert len(sys.argv) >= 2, "Must pass yaml file."
  yaml_file = sys.argv[1]
  print( "Running: " + yaml_file )
  y = load_yaml( yaml_file)
  load_data_from_dict( y[DATA] )
  data_dict      = y[DATA] #{N_TRAIN:4000}
  survival_dict  = y["survival"]
  logging_dict   = y[LOGGING]
  
  logging_dict[SAVEDIR] = os.path.join( HOME_DIR, os.path.join( logging_dict[LOCATION], logging_dict[EXPERIMENT] ) )
  
  fill_location = os.path.join( logging_dict[SAVEDIR], "full_vae_fill.h5" )
  survival_location = os.path.join( logging_dict[SAVEDIR], "full_vae_survival.h5" )
 
  print( "FILL: "  + fill_location )
  print( "SURV: " + survival_location )
  s=pd.HDFStore( survival_location, "r" )
  d=data_dict['store'] #pd.HDFStore( data_location, "r" )
  f=pd.HDFStore( fill_location, "r" ) 
  
  #pdb.set_trace()
  
  
  for survival_spec in survival_dict:
    name = survival_spec["name"]
    print( "running run_pytorch_survival_folds ," + str(data_dict['validation_tissues']) )
    
    #folds = survival_spec["folds"]
    bootstraps = survival_spec["bootstraps"]
    epsilon =  survival_spec["epsilon"]
    if survival_spec.has_key("l1_survival"):
      l1_survival = survival_spec["l1_survival"]
    else:
      l1_survival = 0.0
    if survival_spec.has_key("n_epochs"):
      n_epochs = survival_spec["n_epochs"]
    else:
      n_epochs = 1000
    

    if survival_spec.has_key("l1_regression"):
      l1_regression = survival_spec["l1_regression"]
    else:
      l1_regression = 0.0
    
    folds_survival =  survival_spec["folds_survival"]
    folds_regression =  survival_spec["folds_regression"]
    
      
    
    save_weights_template = os.path.join( logging_dict[SAVEDIR], "survival_weights_" ) 
    projections, probabilties, weights, averages, X, y, E_train, T_train = run_pytorch_survival_folds( data_dict['validation_tissues'], \
                                                                               f, d, k_fold = folds_survival, \
                                                                               n_bootstraps = bootstraps, \
                                                                               l1= l1_survival, n_epochs = n_epochs, normalize=True )  
    disease = data_dict['validation_tissues'][0]
    
    
    avg_proj = averages[0]
    avg_prob = averages[1]

    fig = pp.figure()
    mn_proj = projections[0]
    std_proj = np.sqrt(projections[1])
    mn_prob = probabilties[0]
    std_prob = np.sqrt(probabilties[1])
    mn_w = weights[0]
    std_w = np.sqrt(weights[1])

    ax1 = fig.add_subplot(111)
    I = pp.find( np.isnan(mn_prob))
    mn_prob[I] = 0
    I = pp.find( np.isinf(mn_prob))
    mn_prob[I] = 1
    
    I = pp.find( np.isnan(mn_proj))
    mn_proj[I] = 0
    I = pp.find( np.isinf(mn_proj))
    mn_proj[I] = 1
    
    I = np.argsort(-mn_proj)
    #I = np.argsort(-mn_prob)
    third = int(len(I)/3.0)
    half = int(len(I)/2.0)
    # I0 = I[:third]
    # I1 = I[third:2*third]
    # I2 = I[2*third:]
    I0 = I[:half]
    I1 = [] #I[third:2*third]
    I2 = I[half:]
    kmf = KaplanMeierFitter()
    if len(I2) > 0:
      kmf.fit(T_train[I2], event_observed=E_train[I2], label =  "lda_1 E=%d C=%d"%(E_train[I2].sum(),len(I2)-E_train[I2].sum()))
      ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='red')
    if len(I1) > 0:
      kmf.fit(T_train[I1], event_observed=E_train[I1], label =  "lda_1 E=%d C=%d"%(E_train[I1].sum(),len(I1)-E_train[I1].sum()))
      ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='green')
    if len(I0) > 0:
      kmf.fit(T_train[I0], event_observed=E_train[I0], label = "lda_0 E=%d C=%d"%(E_train[I0].sum(),len(I0)-E_train[I0].sum()))
      ax1=kmf.plot(ax=ax1,at_risk_counts=False,show_censors=True, color='blue')
    results = logrank_test(T_train[I0], T_train[I2], event_observed_A=E_train[I0], event_observed_B=E_train[I2])
    pp.title("%s Log-rank Test: %0.1f"%(disease, results.test_statistic))
    save_location = os.path.join( logging_dict[SAVEDIR], "survival_pytorch_xval.png" )  
    pp.savefig(save_location, dpi=300, format='png')
    print( "ROC mn_prob " + str(roc_auc_score(y,mn_prob) ) )
    print( "ROC mn_proj " + str(roc_auc_score(y,mn_proj) ) )
    
    
    print( "LOG RANK TEST: " +  str(results.test_statistic) )
  
  
  