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


class WeibullSurvivalModel(nn.Module):
    def __init__(self, dim ):
        # this is the place where you instantiate all your modules
        # you can later access them using the same names you've given them in here
        super(WeibullSurvivalModel, self).__init__()
        self.dim       = dim

        # linear parameters for scale
        self.alpha0 = Parameter( torch.zeros([1]), requires_grad = True )
        #self.alpha  = Parameter( torch.zeros(self.dim), requires_grad = True )

        # linear parameters for shape
        self.beta0 = Parameter( torch.zeros([1]), requires_grad = True )
        self.beta  = Parameter( torch.zeros(self.dim), requires_grad = True )

    def LogScale( self, Z ):
      return - self.alpha0.expand(Z.size()[0]) # - torch.mv(Z,self.alpha)
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
      at_time = torch.FloatTensor([1]); at_time[0] = 0.5; at_time = Variable(at_time)
      log_at_time = torch.log(at_time)
      
      #log_at_time = torch.log(at_time)
      scale = self.Scale( Z ) # alpha
      shape = self.Shape( Z ) # lambda
      
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

      return torch.sum( E*log_hazard ) + torch.sum( log_survival )

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
      
    def fit( self, E, T, Z, n_epochs = 10000, lr = 2*1e-2, logging_frequency = 500, l1 = 0.0, normalize=False ):
      
      #model = WeibullSurvivalModel( dim )
      data  = [E, T, Z]

      #def loss_function( log_hazard, log_survival):
      #  return -torch.sum( log_hazard + log_survival )  

      optimizer = optim.Adam(self.parameters(), lr=lr)
      
      for epoch in xrange(1, n_epochs):
        self.train()
        train_loss = 0
        optimizer.zero_grad()
        log_hazard, log_survival = self(data)
        loss = -torch.mean( log_hazard + log_survival ) #loss_function(log_hazard, log_survival)
        
        if l1 > 0:
          loss += l1*torch.sum( torch.abs( self.beta) )
          #loss += l1*torch.sum( torch.abs( self.alpha) )
        loss.backward()
        optimizer.step()

        if epoch%logging_frequency == 0:
          print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss.data[0] ))
          print('                alpha0: {:.3f} '.format( self.alpha0.data[0]))
          #print('                alpha0: {:.3f} alpha: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format( self.alpha0.data[0], self.alpha.data[0], self.alpha.data[1], self.alpha.data[2], self.alpha.data[3], self.alpha.data[4], self.alpha.data[5]))
          b_str = ""
          for b in self.beta.data.numpy():
            b_str += "%0.3f "%(b)
          #print(b_str )
          print('                beta0: {:.3f} beta: {:s}'.format( self.beta0.data[0], b_str ) ) #self.beta.data[0], self.beta.data[1], self.beta.data[2], self.beta.data[3], self.beta.data[4], self.beta.data[5]))
      if n_epochs%logging_frequency == 0:
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss.data[0] ))
        print('                alpha0: {:.3f} '.format( self.alpha0.data[0]))
        #print('                alpha0: {:.3f} alpha: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format( self.alpha0.data[0], self.alpha.data[0], self.alpha.data[1], self.alpha.data[2], self.alpha.data[3], self.alpha.data[4], self.alpha.data[5]))
        #print('                beta0: {:.3f} beta: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format( self.beta0.data[0], self.beta.data[0], self.beta.data[1], self.beta.data[2], self.beta.data[3], self.beta.data[4], self.beta.data[5]))
        b_str = ""
        for b in self.beta.data.numpy():
          b_str += "%0.3f "%(b)
        #print(b_str )
        print('                beta0: {:.3f} beta: {:s}'.format( self.beta0.data[0], b_str ) ) #self.beta.data[0], self.beta.data[1], self.beta.data[2],       #self.train_frailty = self.LogFrailty( Z, T )
    #
    # for epoch in range(1, 15000):
    #     train(epoch)
      

if __name__ == '__main__':
  from lifelines import datasets
  data=datasets.load_regression_dataset()
  #
  E = torch.FloatTensor( data["E"].values.astype(float) )
  T = torch.FloatTensor( data["T"].values.astype(float) )
  Z = torch.FloatTensor( data[["var1","var2","var3"]].values.astype(float) )

  var_E, var_T, var_Z = Variable(E), Variable(T), Variable(Z)
  n   = Z.size()[0]
  dim = Z.size()[1]

  print( data[:10] )

  model = WeibullSurvivalModel( dim )
  #data  = [var_E, var_T, var_Z]

  #def loss_function( log_hazard, log_survival):
  #  return -torch.sum( log_hazard + log_survival )  

  #optimizer = optim.Adam(model.parameters(), lr=0.01)

  model.fit( var_E, var_T, var_Z )

  log_lambda = model.LogShape( var_Z )
  log_scale  = model.LogScale( var_Z )

  f = model.PlotSurvival( var_E, var_T, var_Z )

  pp.show()       

  