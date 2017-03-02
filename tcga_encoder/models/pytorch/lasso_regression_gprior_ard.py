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
        self.log_z   = Parameter( torch.ones(self.dim), requires_grad = True )
        #self.z       = torch.exp(self.log_z)
        

    def forward( self, X ):
      return torch.mv(X,self.w) + self.bias.expand(X.size()[0]), torch.exp(self.log_z)
    
    def predict( self, X ):
      return torch.mv(X,self.w) + self.bias.expand(X.size()[0])
      
    def fit( self, X, y, n_epochs = 10, lr = 2*1e-2, logging_frequency = 1, normalize=False ):
      
      #model = WeibullSurvivalModel( dim )
      data  = [X, y]

      #def loss_function( log_hazard, log_survival):
      #  return -torch.sum( log_hazard + log_survival )  

      optimizer = optim.Adam(self.parameters(), lr=lr)
      sumXX = torch.mean( X*X, dim=0 )
      for epoch in xrange(1, n_epochs):
        self.train()
        train_loss = 0
        optimizer.zero_grad()
        y_pred, z = self(X)
        loss_func = torch.nn.MSELoss() #0.5*torch.mean( torch.square() ) #loss_function(log_hazard, log_survival)
        loss = loss_func( y_pred, y )
        if self.l1 > 0:
          loss += torch.sum( z*torch.abs( self.w ) )
          loss += self.l1*torch.sum( sumXX / z )
          #pdb.set_trace()
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
  
  #from lifelines import datasets
  #data=datasets.load_regression_dataset()
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

  