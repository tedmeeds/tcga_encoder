from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import torch.optim as optim
import pylab as pp

from lifelines import datasets
data=datasets.load_regression_dataset()
#
E = torch.FloatTensor( data["E"].values.astype(float) )
T = torch.FloatTensor( data["T"].values.astype(float) )
Z = torch.FloatTensor( data[["var1","var2","var3"]].values.astype(float) )

var_E, var_T, var_Z = Variable(E), Variable(T), Variable(Z)


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

    def LogScale( self, Z ):
      return - self.alpha0.expand(Z.size()[0]) - torch.mv(Z,self.alpha)
      #return -self.alpha0 - torch.mv(Z,self.alpha) #torch.dot( Z, self.alpha )
      #return -self.alpha0 - Z*self.alpha

    def Scale( self, Z ):
      return torch.exp( self.LogScale(Z) )

    def LogShape( self, Z ):
      return - self.beta0.expand(Z.size()[0]) - torch.mv(Z,self.beta)
      #return -self.beta0 - Z*self.beta

    def LogHazard( self, T, Z ):
      log_shape = self.LogShape( Z )
      log_scale = self.LogScale( Z )

      scale = torch.exp( log_scale )

      return log_shape + log_scale + (scale-1.0)*torch.log( T )

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
      
    

n   = Z.size()[0]
dim = Z.size()[1]

print( data[:10] )

model = WeibullSurvivalModel( dim )
data  = [var_E, var_T, var_Z]

def loss_function( log_hazard, log_survival):
  return -torch.sum( log_hazard + log_survival )  

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epoch):

  model.train()
  train_loss = 0
  #for batch_idx in range(10):
  #data = Variable(data)
  # if args.cuda:
  #     data = data.cuda()
  optimizer.zero_grad()
  log_hazard, log_survival = model(data)
  loss = loss_function(log_hazard, log_survival)
  loss.backward()
  optimizer.step()

  if epoch%100 == 0:
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss.data[0] ))
    print('                alpha0: {:.3f} alpha: {:.3f}, {:.3f}, {:.3f}'.format( model.alpha0.data[0], model.alpha.data[0], model.alpha.data[1], model.alpha.data[2]))
    print('                beta0: {:.3f} beta: {:.3f}, {:.3f}, {:.3f}'.format( model.beta0.data[0], model.beta.data[0], model.beta.data[1], model.beta.data[2]))

for epoch in range(1, 5000):
    train(epoch)
    #test(epoch)

times = np.linspace( min(T), max(T), 100 )
var_times = Variable( torch.FloatTensor( times ) )

s = model.Survival( var_T, var_Z )   

f = pp.figure()
ax = f.add_subplot(111)     
for zi, si, ti in zip( var_Z, s, var_T ):
  s_series = model.Survival( var_times, zi.resize(1,3) )
  
  
  ax.plot( times, s_series.data.numpy(), 'k-', lw=1, alpha = 0.5 )
  
base_s_series = model.Survival( var_times, 0*zi.resize(1,3) )
ax.plot( times, base_s_series.data.numpy(), 'm-', lw=4, alpha = 0.75 )

events = pp.find( var_E.data.numpy() )
censors = pp.find(1-var_E.data.numpy())  
ax.plot(var_T.data.numpy()[events], s.data.numpy()[events], 'ro')
ax.plot(var_T.data.numpy()[censors], s.data.numpy()[censors], 'cs')

pp.show()       
# if __name__ == "__main__":
#   #var_E = torch.IntTensor( E )
#   #var_T = torch.FloatTensor( T )
#   #var_Z = torch.FloatTensor( Z )
#   #print data, E, T, Z
#
#
#
#   print "log-likelihood", M.LogLikelihood( var_E, var_T, var_Z )
#
#   loss = M.Loss( var_E, var_T, var_Z )
#
#
#
#   for batch_idx in range(10):
#     log_hazard, log_survival = model([var_E, var_T, var_Z])
#     loss = loss_function(recon_batch, data, mu, logvar)
#     loss.backward()
#     train_loss += loss.data[0]
#     optimizer.step()
#     if batch_idx % args.log_interval == 0:
#         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#             epoch, batch_idx * len(data), len(train_loader.dataset),
#             100. * batch_idx / len(train_loader),
#             loss.data[0] / len(data)))
#
#
#     print "log-likelihood", M.LogLikelihood( var_E, var_T, var_Z )
  