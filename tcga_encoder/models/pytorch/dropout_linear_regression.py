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
import time
def make_data( X_val, y_val, bootstrap = False, use_cuda=False ):
  if bootstrap is True:
    ids = np.squeeze( make_bootstraps( np.arange(len(X_val)),1) )
    
  
    X = Variable( torch.FloatTensor( X_val[ids,:] ) )
    y = Variable( torch.FloatTensor( y_val[ids] ) )
  else:
    X = Variable( torch.FloatTensor( X_val ) )
    y = Variable( torch.FloatTensor( y_val ) )
    
  if use_cuda:
    X = X.cuda()
    y = y.cuda()
  return X,y
  
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.fc2(x))
#         return F.log_softmax(x)
        
class DropoutLinearRegression(nn.Module):
    def __init__(self, dim, use_cuda = False  ):
        # this is the place where you instantiate all your modules
        # you can later access them using the same names you've given them in here
        super(DropoutLinearRegression, self).__init__()
        self.dim       = dim
        self.K         = 1
        self.use_cuda = use_cuda
        
        if self.use_cuda is True:
          print("!!!! DropoutLinearRegression: USING CUDA !!!!!!")
          self.H = torch.nn.Linear(self.dim, 1, bias=True).cuda()
          
        else:
          self.H = torch.nn.Linear(self.dim, 1, bias=True)
        
        self.P = []
        for p in self.H.parameters():
          self.P.append(p)

        self.w = self.P[0]
        self.b = self.P[1]
   
    def get_w(self):
      if self.use_cuda is True:
        return self.w.cpu().data.numpy()
      else:
        return self.w.data.numpy()

    def get_b(self):
      if self.use_cuda is True:
        return self.b.cpu().data.numpy()
      else:
        return self.b.data.numpy()
            
    def add_test(self, X_test, y_test ):
      self.X_test = X_test
      self.y_test = y_test

    def Hidden( self, X ):
      return F.tanh( self.H(Z) )
      
    def predict( self, x ):
      x = self.H(x)
      if self.use_cuda is True:
        return x.cpu().data.numpy()
      else:
        return x.data.numpy()
      
    def forward( self, x ):
      if self.use_cuda is True:
        x = F.dropout(x, training=self.training).cuda()
      else:
        x = F.dropout(x, training=self.training)
      x = self.H(x)
      return x

    def test(self,epoch,logging_frequency):
      self.training = False
      if self.X_test is None:
        return
      y_est = self(self.X_test)
      data_loss = torch.mean( torch.pow( self.y_test-y_est, 2) ).data[0]
      #data_loss = torch.mean( torch.abs( self.y_test-y_est) ).data[0]

      #if epoch%logging_frequency == 0:
      #  print('====> Test set loss: {:.4f}'.format(data_loss))
      
      self.stop = False
      if data_loss < self.test_cost:
        self.test_cost = data_loss
      else:
        if epoch>=self.min_epochs:
          print('====> STOPPING @ ' + str(epoch) )
          self.stop = True
      self.training = True
      return data_loss
        
    def fit( self, X_train_val, y_train_val, n_epochs = 10000, lr = 2*1e-2, logging_frequency = 200, l1 = 0.0, l2=0.0, testing_frequency = 10,min_epochs=4000):
      print("L1 = " + str(l1))
      print("L2 = " + str(l2))
      self.min_epochs = min_epochs
      self.stop = False
      self.test_cost = np.inf
      optimizer = optim.Adam(self.parameters(), lr=lr)
      self.training = True
      
      std_x = Variable( torch.FloatTensor( X_train_val.std(0) ) )+1e-4
      if self.use_cuda is True:        
        std_x = std_x.cuda()

      start_time=time.time()
      for epoch in xrange(1, n_epochs):
          
        train_loss = 0
        optimizer.zero_grad()
        
        X, y = make_data( X_train_val, y_train_val, bootstrap = False, use_cuda = self.use_cuda )
        y_est = self(X)
        
        #pdb.set_trace()
        #data_loss = torch.mean( torch.abs( y-y_est) ) 
        data_loss = torch.mean( torch.pow( y-y_est, 2) ) 
        weight_loss = 0.0
        if l1 > 0:
         weight_loss += l1*torch.sum( torch.abs( self.w )/std_x )
        if l2 > 0:
         weight_loss += l2*torch.sum( torch.pow( self.w/std_x, 2) )
         
        loss = data_loss + weight_loss
        loss.backward()
        optimizer.step()

        stop_time = time.time()
        delta_time = start_time-stop_time
        if epoch%testing_frequency == 0 and epoch%logging_frequency == 0:
          data_loss_test = self.test(epoch, logging_frequency)

          print('====> Epoch: {} Average loss: {:.4f}  TEST: {:.4f} time: {:.1f}'.format(epoch, data_loss.data[0], data_loss_test, delta_time ))
          start_time=stop_time
          if self.stop is True:
            print("!!!!!!!!! early stopping" )
            return
        elif epoch%logging_frequency == 0:
          print('====> Epoch: {} Average loss: {:.4f}                time: {:1f}'.format(epoch, data_loss.data[0], delta_time ))
          start_time=stop_time
        # if epoch%testing_frequency == 0:
        #   data_loss = self.test(epoch, logging_frequency)
        #   if self.stop is True:
        #     print("!!!!!!!!! early stopping" )
        #     return
      # if n_epochs%logging_frequency == 0:
      #   print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, data_loss.data[0] ))
      data_loss_test = self.test(epoch, logging_frequency)
      print('====> Epoch: {} Average loss: {:.4f}  TEST: {:.4f}'.format(epoch, data_loss.data[0], data_loss_test ))
    