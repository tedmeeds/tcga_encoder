from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import torch.optim as optim
import pylab as pp




class PytorchLasso(nn.Module):
    def __init__(self, dim, l1 = 0.0 ):
        # this is the place where you instantiate all your modules
        # you can later access them using the same names you've given them in here
        super(PytorchLasso, self).__init__()
        self.dim       = dim
        self.l1        = l1

        # linear parameters for scale
        self.bias    = Parameter( 0*torch.randn([1]), requires_grad = True )
        self.w       = Parameter( 0.0*torch.randn(self.dim), requires_grad = True )

    # def LogScale( self, Z ):
    #   return - self.alpha0.expand(Z.size()[0]) - torch.mv(Z,self.alpha)
    #   #return -self.alpha0 - torch.mv(Z,self.alpha) #torch.dot( Z, self.alpha )
    #   #return -self.alpha0 - Z*self.alpha
    #
    # def Scale( self, Z ):
    #   return torch.exp( self.LogScale(Z) )
    #
    # def LogShape( self, Z ):
    #   return - self.beta0.expand(Z.size()[0]) - torch.mv(Z,self.beta)
    #   #return -self.beta0 - Z*self.beta
    #
    # def LogHazard( self, T, Z ):
    #   log_shape = self.LogShape( Z )
    #   log_scale = self.LogScale( Z )
    #
    #   scale = torch.exp( log_scale )
    #
    #   return log_shape + log_scale + (scale-1.0)*torch.log( T )
    #
    # def LogFrailty( self, Z, T ):
    #   #return self.LogHazard( T, Z )
    #   log_shape = self.LogShape( Z )
    #   log_scale = self.LogScale( Z )
    #
    #   #f = log_scale+log_shape #
    #   f = - torch.mv(Z,self.alpha)  - torch.mv(Z,self.beta)
    #   return f
    #
    # def Hazard( self, T, Z ):
    #   return torch.exp( self.log_hazard( T, Z ) )
    #
    # def LogSurvival( self, T, Z ):
    #   return -self.CumulativeHazard( T, Z )
    #
    # def Survival( self, T, Z ):
    #   return torch.exp( - self.CumulativeHazard( T, Z ) )
    #
    # def LogCumulativeHazard( self, T, Z ):
    #   if Z.size()[0] == 1 and T.size()[0]>1:
    #     return self.LogShape(Z).expand(T.size()) + self.Scale(Z).expand(T.size())*torch.log(T)
    #   else:
    #     return self.LogShape(Z) + self.Scale(Z)*torch.log(T)
    #
    # def CumulativeHazard( self, T, Z ):
    #   return torch.exp( self.LogCumulativeHazard(T,Z) )
    #
    # # ie log f(t|z) = log h(t|z) + log S(T|z)
    # def LogPdf( self, T, Z ):
    #   return self.LogHazard(T,Z) + self.LogSurvival(Z,T)
    #
    # def Pdf(self, T, Z ):
    #   return self.LogPdf( T, Z )

    #
    # def LogLikelihood( self, E, T, Z ):
    #   # E: events, binary vector indicating "death" (n by 1)
    #   # T: time of event or censor (n by 1)
    #   # Z: matrix of covariates (n by dim)
    #   log_hazard = self.LogHazard( T, Z )
    #   log_survival = self.LogSurvival( T, Z )
    #
    #   return torch.sum( E*log_hazard ) + torch.sum( log_survival )
    #
    # def Loss( self, E, T, Z ):
    #   return - self.LogLikelihood( E, T, Z )

    def forward( self, X ):
      return torch.mv(X,self.w) + self.bias.expand(X.size()[0])
      
    # def PlotSurvival( self, E, T, Z ):
    #   f = pp.figure()
    #   times = np.linspace( min(T.data.numpy()), max(T.data.numpy()), 100 )
    #   var_times = Variable( torch.FloatTensor( times ) )
    #
    #   s = self.Survival( T, Z )
    #   #log_lambda = model.LogShape( var_Z )
    #   #log_scale  = model.LogScale( var_Z )
    #
    #   #f = pp.figure()
    #   ax = f.add_subplot(111)
    #   for zi, si, ti in zip( Z, s, T ):
    #     s_series = self.Survival( var_times, zi.resize(1,Z.size()[1]) )
    #     ax.plot( times, s_series.data.numpy(), 'k-', lw=1, alpha = 0.5 )
    #
    #   base_s_series = self.Survival( var_times, 0*zi.resize(1,Z.size()[1]) )
    #   ax.plot( times, base_s_series.data.numpy(), 'm-', lw=4, alpha = 0.75 )
    #
    #   events = pp.find( E.data.numpy() )
    #   censors = pp.find(1-E.data.numpy())
    #   ax.plot(var_T.data.numpy()[events], s.data.numpy()[events], 'ro')
    #   ax.plot(var_T.data.numpy()[censors], s.data.numpy()[censors], 'cs')
    #
    #   return f
    #
    
    def predict( self, X ):
      return self.forward(X)
      
    def fit( self, X, y, n_epochs = 10, lr = 2*1e-2, logging_frequency = 1, normalize=False ):
      
      #model = WeibullSurvivalModel( dim )
      data  = [X, y]

      #def loss_function( log_hazard, log_survival):
      #  return -torch.sum( log_hazard + log_survival )  

      optimizer = optim.Adam(self.parameters(), lr=lr)
      
      for epoch in xrange(1, n_epochs):
        self.train()
        train_loss = 0
        optimizer.zero_grad()
        y_pred = self(X)
        loss_func = torch.nn.MSELoss() #0.5*torch.mean( torch.square() ) #loss_function(log_hazard, log_survival)
        loss = loss_func( y_pred, y )
        if self.l1 > 0:
          loss += self.l1*torch.sum( torch.abs( self.w ) )
          #loss += l1*torch.sum( torch.abs( self.alpha) )
        loss.backward()
        optimizer.step()

        if epoch%logging_frequency == 0:
          print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss.data[0] ))
          print('                bias: {:.3f} w: {:.3f}, {:.3f}, {:.3f}'.format( self.bias.data[0], self.w.data[0], self.w.data[1], self.w.data[2]))
          #print('                beta0: {:.3f} beta: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format( self.beta0.data[0], self.beta.data[0], self.beta.data[1], self.beta.data[2], self.beta.data[3], self.beta.data[4], self.beta.data[5]))
        if epoch%logging_frequency == 0:
          print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss.data[0] ))
          print('                bias: {:.3f} w: {:.3f}, {:.3f}, {:.3f}'.format( self.bias.data[0], self.w.data[0], self.w.data[1], self.w.data[2]))
      #self.train_frailty = self.LogFrailty( Z, T )
    #
    # for epoch in range(1, 15000):
    #     train(epoch)
      

if __name__ == '__main__':
  l1 = 0.1
  n = 500
  dim = 3
  
  w = np.random.randn(3)
  w[1] = 0
  b = 0.5
  noise = 0.25
  X = np.random.randn( n,dim )
  y = np.dot( X, w ) + b + noise*np.random.randn(n)
  
  from lifelines import datasets
  data=datasets.load_regression_dataset()
  #
  X = Variable( torch.FloatTensor( X ) )
  y = Variable( torch.FloatTensor( y ) )

  model = PytorchLasso( dim, l1 )
  #data  = [var_E, var_T, var_Z]

  #def loss_function( log_hazard, log_survival):
  #  return -torch.sum( log_hazard + log_survival )  

  #optimizer = optim.Adam(model.parameters(), lr=0.01)

  model.fit( X,y )
  y_pred = model.predict( X )

  pp.figure()
  pp.subplot(1,2,1)
  pp.plot( w, 'bo-', alpha=0.5)
  pp.plot( model.w.data.numpy(), 'ro-', alpha=0.5)
  pp.subplot(1,2,2)
  pp.plot( y.data.numpy(), y_pred.data.numpy(), 'bo', alpha=0.5)
  pp.axis('equal')
  pp.show()       

  