from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import torch.optim as optim
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
  
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)
        
class WeibullSurvivalModelNeuralNetwork(nn.Module):
    def __init__(self, dim, K  ):
        # this is the place where you instantiate all your modules
        # you can later access them using the same names you've given them in here
        super(WeibullSurvivalModelNeuralNetwork, self).__init__()
        self.dim       = dim
        self.K         = K
        
        self.H = torch.nn.Linear(self.dim, self.K, bias=True)
        self.beta  = torch.nn.Linear(self.K, 1, bias=True)
        self.alpha  = torch.nn.Linear(self.K, 1, bias=True)
        
        self.P = []
        for p in self.H.parameters():
          self.P.append(p)

        self.Pb = []
        for p in self.beta.parameters():
          self.Pb.append(p)

        self.Pa = []
        for p in self.alpha.parameters():
          self.Pa.append(p)



        self.w_b = self.Pb[0]  
        self.w_a = self.Pa[0]  
        self.w = self.P[0]
        # linear parameters for scale
        #self.alpha0 = Parameter( torch.zeros([1]), requires_grad = True )
        #self.alpha  = Parameter( torch.zeros(self.dim), requires_grad = True )

        # linear parameters for shape
        #self.beta0 = Parameter( torch.zeros([1]), requires_grad = True )
        #self.beta  = Parameter( torch.zeros(self.dim), requires_grad = True )

    def Hidden( self, Z ):
      return F.relu( self.H(Z) )
      
    def forward( self, x ):
      E = x[0]
      T = x[1]
      Z = x[2]
      
      hidden = self.Hidden( Z ) 
 
      return E*self.LogHazard( T, hidden ), self.LogSurvival( T, hidden )
      
    def LogScale( self, hidden ):
      return - self.alpha(hidden)
      #return - self.alpha0.expand(Z.size()[0]) # - torch.mv(Z,self.alpha)
      #return -self.alpha0 - torch.mv(Z,self.alpha) #torch.dot( Z, self.alpha )
      #return -self.alpha0 - Z*self.alpha

    def Scale( self, hidden ):
      return torch.exp( self.LogScale(hidden) )

    def LogShape( self, hidden ):
      return -self.beta(hidden)
      #return - self.beta0.expand(Z.size()[0]) - torch.mv(Z,self.beta)
      #return -self.beta0 - Z*self.beta
    
    def Shape( self, hidden ):
      return torch.exp( self.LogShape(hidden) )
      
    def LogHazard( self, T, hidden ):
      log_shape = self.LogShape( hidden )
      log_scale = self.LogScale( hidden )

      scale = torch.exp( log_scale )

      return log_shape + log_scale + (scale-1.0)*torch.log( T )

    # def LogFrailty( self, Z, T ):
    #   #return self.LogHazard( T, Z )
    #   log_shape = self.LogShape( Z )
    #   log_scale = self.LogScale( Z )
    #
    #   #f = log_scale+log_shape #
    #   f = - torch.mv(Z,self.alpha)  - torch.mv(Z,self.beta)
    #   return f
    
    def LogTime( self, Z, at_time = 0.5 ):
      at_time = torch.FloatTensor([1]); at_time[0] = 0.5; at_time = Variable(at_time)
      log_at_time = torch.log(at_time)
      
      hidden = self.Hidden( Z ) 
      #log_at_time = torch.log(at_time)
      scale = self.Scale( hidden ) # alpha
      shape = self.Shape( hidden ) # lambda
      
      #pdb.set_trace()
      log_t = torch.log( -log_at_time.expand(shape.size()) / shape ) / scale
      
      #pdb.set_trace()
      return log_t
        
    def Hazard( self, T, Z ):
      return torch.exp( self.log_hazard( T, Z ) )

    def LogSurvival( self, T, Z ):
      return -self.CumulativeHazard( T, Z )

    def Survival( self, T, Z ):
      return torch.exp( - self.CumulativeHazard( T, Z ) )

    def LogCumulativeHazard( self, T, Z ):
      if Z.size()[0] == 1 and T.size()[0]>1:
        return self.LogShape(Z).expand(T.size()) + self.Scale(Z).expand(T.size())*torch.log(T)
      else:
        return self.LogShape(Z) + self.Scale(Z)*torch.log(T)

    def CumulativeHazard( self, T, Z ):
      return torch.exp( self.LogCumulativeHazard(T,Z) )

    # ie log f(t|z) = log h(t|z) + log S(T|z)
    def LogPdf( self, T, Z ):
      return self.LogHazard(T,Z) + self.LogSurvival(Z,T)

    def Pdf(self, T, Z ):
      return self.LogPdf( T, Z )


    def LogLikelihood( self, E, T, Z ):
      # E: events, binary vector indicating "death" (n by 1)
      # T: time of event or censor (n by 1)
      # Z: matrix of covariates (n by dim)
      hidden = self.H(Z)
      log_hazard = self.LogHazard( T, hidden )
      log_survival = self.LogSurvival( T, hidden )

      return torch.sum( E*log_hazard ) + torch.sum( log_survival )

    def Loss( self, E, T, Z ):
      return - self.LogLikelihood( E, T, Z )


      
    def PlotSurvival( self, E, T, Z, ax = None, color = "k" ):
      if ax is None:
        f = pp.figure()
        ax = f.add_subplot(111)    
      times = np.linspace( 1.0, max(T.data.numpy()), 100 )
      var_times = Variable( torch.FloatTensor( times ) )

      s = self.Survival( T, Z )   
      #log_lambda = model.LogShape( var_Z )
      #log_scale  = model.LogScale( var_Z )

      #f = pp.figure()
      #ax = f.add_subplot(111)     
      for zi, si, ti in zip( Z, s, T ):
        s_series = self.Survival( var_times, zi.resize(1,Z.size()[1]) )
        ax.plot( times, s_series.data.numpy(), color+'-', lw=1, alpha = 0.5 )
  
      base_s_series = self.Survival( var_times, 0*zi.resize(1,Z.size()[1]) )
      ax.plot( times, base_s_series.data.numpy(), 'm-', lw=4, alpha = 0.75 )

      events = pp.find( E.data.numpy() )
      censors = pp.find(1-E.data.numpy())  
      ax.plot(T.data.numpy()[events], s.data.numpy()[events], 'ro')
      ax.plot(T.data.numpy()[censors], s.data.numpy()[censors], 'cs')
      
      return ax
      
    def fit( self, E, T, Z, n_epochs = 10000, lr = 2*1e-2, logging_frequency = 500, l1 = 0.0, normalize=False ):
      
      #model = WeibullSurvivalModel( dim )
      data  = [E, T, Z]
      n = len(Z)
      #def loss_function( log_hazard, log_survival):
      #  return -torch.sum( log_hazard + log_survival )  

      optimizer = optim.Adam(self.parameters(), lr=lr)
      
      for epoch in xrange(1, n_epochs):
        self.train()
        train_loss = 0
        optimizer.zero_grad()
        log_hazard, log_survival = self(data)
        loss = -torch.mean( log_hazard + log_survival  ) #loss_function(log_hazard, log_survival)
        #pdb.set_trace()
        if l1 > 0:
         loss += l1*torch.sum( torch.pow( self.w, 2) )
         loss += l1*torch.sum( torch.pow( self.w_a,2) )
         loss += l1*torch.sum( torch.pow( self.w_b,2) )
        #  #loss += l1*torch.sum( torch.abs( self.alpha) )
        loss.backward()
        optimizer.step()

        if epoch%logging_frequency == 0:
          print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss.data[0] ))
          #print('                alpha0: {:.3f} '.format( self.alpha0.data[0]))
          #b_str = ""
          #for b in self.beta.data.numpy():
          #  b_str += "%0.3f "%(b)
          #print('                beta0: {:.3f} beta: {:s}'.format( self.beta0.data[0], b_str ) ) 
      if n_epochs%logging_frequency == 0:
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss.data[0] ))
        #print('                alpha0: {:.3f} '.format( self.alpha0.data[0]))
        #b_str = ""
        #for b in self.beta.data.numpy():
        #  b_str += "%0.3f "%(b)
        #print('                beta0: {:.3f} beta: {:s}'.format( self.beta0.data[0], b_str ) ) 
        

class WeibullSurvivalModel(nn.Module):
    def __init__(self, dim ):
        # this is the place where you instantiate all your modules
        # you can later access them using the same names you've given them in here
        super(WeibullSurvivalModel, self).__init__()
        self.dim       = dim

        # linear parameters for scale
        self.alpha0 = Parameter( torch.zeros([1]), requires_grad = True )
        self.alpha  = Parameter( torch.zeros(self.dim), requires_grad = True )

        # linear parameters for shape
        self.beta0 = Parameter( torch.zeros([1]), requires_grad = True )
        self.beta  = Parameter( torch.zeros(self.dim), requires_grad = True )
        
        # self.P=[]
        # for p in self.beta.parameters():
        #   self.P.append(p)
        self.w = self.beta
        
        self.E_test = None
        self.T_test = None
        self.Z_test = None

    def add_test(self,E_test,T_test,Z_test):
      self.E_test = E_test
      self.T_test = T_test
      self.Z_test = Z_test 
      self.test_data = [self.E_test,self.T_test,self.Z_test]
      
    def LogScale( self, Z ):
      return - self.alpha0.expand(Z.size()[0])  - torch.mv(Z,self.alpha)
      #return -self.alpha0 - torch.mv(Z,self.alpha) #torch.dot( Z, self.alpha )
      #return -self.alpha0 - Z*self.alpha

    def Scale( self, Z ):
      return torch.exp( self.LogScale(Z) )

    def LogShape( self, Z ):
      return - self.beta0.expand(Z.size()[0]) - torch.mv(Z,self.beta)
      #return -self.beta0 - Z*self.beta
    
    def Shape( self, Z ):
      return torch.exp( self.LogShape(Z) )
      
    def LogHazard( self, T, Z ):
      log_shape = self.LogShape( Z )
      log_scale = self.LogScale( Z )

      scale = torch.exp( log_scale )

      return log_shape + log_scale + (scale-1.0)*torch.log( T )

    def LogFrailty( self, Z, T ):
      #return self.LogHazard( T, Z )
      log_shape = self.LogShape( Z )
      log_scale = self.LogScale( Z )
      
      #f = log_scale+log_shape #
      f = - torch.mv(Z,self.alpha)  - torch.mv(Z,self.beta)
      return f
    
    def LogTime( self, Z, at_time = 0.5 ):
      at_time = torch.FloatTensor([at_time]); #at_time[0] = 0.5; 
      at_time = Variable(at_time)
      log_at_time = torch.log(at_time)
      
      #log_at_time = torch.log(at_time)
      scale = self.Scale( Z ) # alpha
      shape = self.Shape( Z ) # lambda
      
      #shape = torch.exp( - torch.mv(Z,self.beta) )
      #scale = torch.exp( - torch.mv(Z,self.alpha) )
      
      #pdb.set_trace()
      log_t = torch.log( -log_at_time.expand(shape.size()) / shape ) / scale
      
      return log_t
        
    def Hazard( self, T, Z ):
      return torch.exp( self.log_hazard( T, Z ) )

    def LogSurvival( self, T, Z ):
      return -self.CumulativeHazard( T, Z )

    def Survival( self, T, Z ):
      return torch.exp( - self.CumulativeHazard( T, Z ) )

    def LogCumulativeHazard( self, T, Z ):
      if Z.size()[0] == 1 and T.size()[0]>1:
        return self.LogShape(Z).expand(T.size()) + self.Scale(Z).expand(T.size())*torch.log(T)
      else:
        return self.LogShape(Z) + self.Scale(Z)*torch.log(T)

    def CumulativeHazard( self, T, Z ):
      return torch.exp( self.LogCumulativeHazard(T,Z) )

    # ie log f(t|z) = log h(t|z) + log S(T|z)
    def LogPdf( self, T, Z ):
      return self.LogHazard(T,Z) + self.LogSurvival(Z,T)

    def Pdf(self, T, Z ):
      return self.LogPdf( T, Z )


    def LogLikelihood( self, E, T, Z ):
      # E: events, binary vector indicating "death" (n by 1)
      # T: time of event or censor (n by 1)
      # Z: matrix of covariates (n by dim)
      log_hazard = self.LogHazard( T, Z )
      log_survival = self.LogSurvival( T, Z )

      return E*log_hazard + log_survival

    def Loss( self, E, T, Z ):
      return - self.LogLikelihood( E, T, Z )

    def forward( self, x ):
      E = x[0]
      T = x[1]
      Z = x[2]
      return E*self.LogHazard( T, Z ), self.LogSurvival( T, Z )
      
    def PlotSurvival( self, E, T, Z, ax = None, color = "k" ):
      if ax is None:
        f = pp.figure()
        ax = f.add_subplot(111)    
      times = np.linspace( 1.0, max(T.data.numpy()), 100 )
      var_times = Variable( torch.FloatTensor( times ) )

      s = self.Survival( T, Z )   
      #log_lambda = model.LogShape( var_Z )
      #log_scale  = model.LogScale( var_Z )

      #f = pp.figure()
      #ax = f.add_subplot(111)     
      for zi, si, ti in zip( Z, s, T ):
        s_series = self.Survival( var_times, zi.resize(1,Z.size()[1]) )
        ax.plot( times, s_series.data.numpy(), color+'-', lw=1, alpha = 0.5 )
  
      base_s_series = self.Survival( var_times, 0*zi.resize(1,Z.size()[1]) )
      ax.plot( times, base_s_series.data.numpy(), 'm-', lw=4, alpha = 0.75 )

      events = pp.find( E.data.numpy() )
      censors = pp.find(1-E.data.numpy())  
      ax.plot(T.data.numpy()[events], s.data.numpy()[events], 'ro')
      ax.plot(T.data.numpy()[censors], s.data.numpy()[censors], 'cs')
      
      return ax
    
    def test(self,epoch,logging_frequency):
      if self.test_data is None:
        return
      #self.eval()
      #test_loss = 0
      log_hazard, log_survival = self(self.test_data)
      data_loss = -torch.mean( log_hazard + log_survival ).data[0]

      #test_loss /= len(test_loader.dataset)
      if epoch%logging_frequency == 0:
        print('====> Test set loss: {:.4f}'.format(data_loss))
      
      self.stop = False
      if data_loss < self.test_cost:
        self.test_cost = data_loss
        #print('====> NEW Test set loss: {:.4f}'.format(self.test_cost))
      else:
        print('====> STOPPING ')
        self.stop = True
        #pdb.set_trace()
      #self.train()
      
      
    def fit( self, E_val, T_val, Z_val, n_epochs = 10000, lr = 2*1e-2, logging_frequency = 500, l1 = 0.0, normalize=False, testing_frequency = 100):
      verbose = False
      self.stop = False
      self.test_cost = np.inf
      optimizer = optim.RMSprop(self.parameters(), lr=lr)
      
      for epoch in xrange(1, n_epochs):
        if verbose is True:
          print( "epoch = " + str(epoch))
        
        train_loss = 0
        optimizer.zero_grad()
        if verbose is True:
          print(" a")
        E,T,Z = make_data( E_val, T_val, Z_val, bootstrap = False )
        data  = [E, T, Z]
        if verbose is True:
          print(" b")
        log_hazard, log_survival = self(data)
        data_loss = -torch.mean( log_hazard + log_survival ) #loss_function(log_hazard, log_survival)
        if verbose is True:
          print(" c")
        weight_loss = 0.0
        if l1 > 0:
          weight_loss += l1*torch.sum( torch.pow( self.beta,2) )
          weight_loss += l1*torch.sum( torch.pow( self.alpha,2) )
          #weight_loss += l1*torch.sum( torch.abs( self.beta) )
          #weight_loss += l1*torch.sum( torch.abs( self.alpha) )
        loss = data_loss + weight_loss
        if verbose is True:
          print(" d")
        loss.backward()
        if verbose is True:
          print(" e")
        optimizer.step()
        if verbose is True:
          print(" f")

        if epoch%logging_frequency == 0:
          print(" XX")
          print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, data_loss.data[0] ))
          print('                alpha0: {:.3f} '.format( self.alpha0.data[0]))
          b_str = ""
          for b in self.beta.data.numpy():
            b_str += "%0.3f "%(b)
          print('                beta0: {:.3f} beta: {:s}'.format( self.beta0.data[0], b_str ) ) 
        if epoch%testing_frequency == 0:
          print(" YY")
          self.test(epoch, logging_frequency)
          if self.stop is True:
            print("!!!!!!!!! early stopping" )
            return
      if n_epochs%logging_frequency == 0:
        print(" ZZ")
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, data_loss.data[0] ))
        print('                alpha0: {:.3f} '.format( self.alpha0.data[0]))
        b_str = ""
        for b in self.beta.data.numpy():
          b_str += "%0.3f "%(b)
        print('                beta0: {:.3f} beta: {:s}'.format( self.beta0.data[0], b_str ) ) 
        self.test(epoch,logging_frequency)
      

# if __name__ == '__main__':
#   from lifelines import datasets
#   data=datasets.load_regression_dataset()
#   #
#   E = torch.FloatTensor( data["E"].values.astype(float) )
#   T = torch.FloatTensor( data["T"].values.astype(float) )
#   Z = torch.FloatTensor( data[["var1","var2","var3"]].values.astype(float) )
#
#   var_E, var_T, var_Z = Variable(E), Variable(T), Variable(Z)
#   n   = Z.size()[0]
#   dim = Z.size()[1]
#
#   print( data[:10] )
#
#   model = WeibullSurvivalModel( dim )
#   #data  = [var_E, var_T, var_Z]
#
#   #def loss_function( log_hazard, log_survival):
#   #  return -torch.sum( log_hazard + log_survival )
#
#   #optimizer = optim.Adam(model.parameters(), lr=0.01)
#
#   model.fit( var_E, var_T, var_Z )
#
#   log_lambda = model.LogShape( var_Z )
#   log_scale  = model.LogScale( var_Z )
#
#   f = model.PlotSurvival( var_E, var_T, var_Z )
#
#   pp.show()

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
  
  
  