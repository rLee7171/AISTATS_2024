import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
from torch.nn.functional import normalize
from layers import *
from torch.nn import Module




class conicMLP_model_0(torch.nn.Module):  # no radial projection required

    def __init__(self, dim, nclasses, nhidden, nchannels, conicType = 'conic'):
      super().__init__()
      self.dim = dim
      self.nclasses = nclasses
      self.nchannels = nchannels
      self.nhidden = nhidden
      self.conicType = conicType
      self.relu = torch.nn.ReLU()
      self.model_name = f"ConicR_mc + dim={self.dim} + "\
                        f"nclasses={self.nclasses} + nchannels={self.nchannels} + "\
                        f"nhidden={self.nhidden} + conicType={self.conicType}"

      self.reduce = torch.nn.Parameter(torch.Tensor(nchannels,dim,nhidden))
      self.W = torch.nn.Parameter(torch.Tensor(nclasses, nchannels*nhidden))
      self.rho = torch.nn.Parameter(torch.Tensor(nclasses))  # theta:  nlassses*1
      self.reset_parameters()

    def reset_parameters(self):
      torch.nn.init.normal_(self.W)
      torch.nn.init.uniform_(self.rho, a=1, b=1.5)
      #orthogonal initialization of reduce
      for j in range(self.nchannels):
         torch.nn.init.orthogonal_(self.reduce[j])


    def forward(self, x):
      y=x
      y = y @ self.reduce                                          # (nchannels, n, nhidden)
      y = F.normalize(input=y, p=2.0, dim=2, eps=1e-12, out=None)  # radial projection

      y = torch.transpose(y, 0, 1)
      y = torch.flatten(y,start_dim = 1)
      y = self.relu(y)                                             # (n, nhidden*nchannels)


      w_nrm = torch.linalg.norm(self.W, dim=1)  # norms of columns  w_nrm:  nlassses*1

      if self.conicType == 'conic':
        z = torch.matmul(y, self.W.t()) - w_nrm * torch.cos(self.rho)
      if self.conicType == 'spheric':
        y_nrm = torch.linalg.norm(y)
        z = torch.pow(self.rho,2) - (torch.pow(y_nrm,2)+torch.pow(w_nrm,2)-2*torch.matmul(y, self.W.t()))
      if self.conicType == 'linear':
        z = torch.matmul(y, self.W.t()) + self.rho

      return F.log_softmax(z, dim=-1)
      #return z


class conicMLP(torch.nn.Module):  # no radial projection required

    def __init__(self, dim, nclasses, nhidden, nchannels, conicType = 'conic'):
      super().__init__()
      self.dim = dim
      self.nclasses = nclasses
      self.nchannels = nchannels
      self.nhidden = nhidden
      self.conicType = conicType
      self.relu = torch.nn.ReLU()
      self.model_name = f"ConicR_mc + dim={self.dim} + "\
                        f"nclasses={self.nclasses} + nchannels={self.nchannels} + "\
                        f"nhidden={self.nhidden} + conicType={self.conicType}"
      self.reduce = torch.nn.Parameter(torch.Tensor(nchannels,dim,nhidden))
      self.W = torch.nn.Parameter(torch.Tensor(nclasses, nchannels*nhidden))
      self.rho = torch.nn.Parameter(torch.Tensor(nclasses))  # theta:  nlassses*1
      self.reset_parameters()

    def reset_parameters(self):
      torch.nn.init.normal_(self.W)
      torch.nn.init.uniform_(self.rho, a=1, b=1.5)
      #orthogonal initialization of reduce
      for j in range(self.nchannels):
         torch.nn.init.orthogonal_(self.reduce[j])


    def forward(self, x):
      y = F.normalize(input=x, p=2.0, dim=1, eps=1e-12, out=None)  # (n,dim)
      y = y @ self.reduce                                          # (nchannels, n, nhidden)
      y = F.normalize(input=y, p=2.0, dim=2, eps=1e-12, out=None)  # radial projection

      y = torch.transpose(y, 0, 1)
      y = torch.flatten(y,start_dim = 1)
      y = self.relu(y)                                             # (n, nhidden*nchannels)


      w_nrm = torch.linalg.norm(self.W, dim=1)  # norms of columns  w_nrm:  nlassses*1

      if self.conicType == 'conic':
        z = torch.matmul(y, self.W.t()) - w_nrm * torch.cos(self.rho)
      if self.conicType == 'spheric':
        y_nrm = torch.linalg.norm(y)
        z = torch.pow(self.rho,2) - (torch.pow(y_nrm,2)+torch.pow(w_nrm,2)-2*torch.matmul(y, self.W.t()))
      if self.conicType == 'linear':
        z = torch.matmul(y, self.W.t()) + self.rho

      return F.log_softmax(z, dim=-1)
    
class conicMLPAblation(torch.nn.Module):  # no radial projection required

    def __init__(self, dim, nclasses, nhidden, nchannels, conicType = 'conic'):
      super().__init__()
      self.dim = dim
      self.nclasses = nclasses
      self.nchannels = nchannels
      self.nhidden = nhidden
      self.conicType = conicType
      self.relu = torch.nn.ReLU()
      self.model_name = f"ConicR_mc + dim={self.dim} + "\
                        f"nclasses={self.nclasses} + nchannels={self.nchannels} + "\
                        f"nhidden={self.nhidden} + conicType={self.conicType}"
      self.reduce = torch.nn.Parameter(torch.Tensor(nchannels,dim,nhidden))
      self.W = torch.nn.Parameter(torch.Tensor(nclasses, nchannels*nhidden))
      self.rho = torch.nn.Parameter(torch.Tensor(nclasses))  # theta:  nlassses*1
      self.reset_parameters()

    def reset_parameters(self):
      torch.nn.init.normal_(self.W)
      torch.nn.init.uniform_(self.rho, a=1, b=1.5)
      #orthogonal initialization of reduce
      for j in range(self.nchannels):
         torch.nn.init.orthogonal_(self.reduce[j])


    def forward(self, x):
      print(self.dim,self.nclasses,self.nchannels,self.nhidden)
      y = F.normalize(input=x, p=2.0, dim=1, eps=1e-12, out=None)  # (n,dim)
      y = y @ self.reduce
      y = torch.transpose(y, 0, 1)
      y = torch.flatten(y,start_dim = 1)
      w_nrm = torch.linalg.norm(self.W, dim=1)  # norms of columns  w_nrm:  nlassses*1
      print("Shape: ",y.shape)
      if self.conicType == 'conic':
        z = torch.matmul(y, self.W.t()) - w_nrm * torch.cos(self.rho)
      if self.conicType == 'spheric':
        y_nrm = torch.linalg.norm(y)
        z = torch.pow(self.rho,2) - (torch.pow(y_nrm,2)+torch.pow(w_nrm,2)-2*torch.matmul(y, self.W.t()))
      if self.conicType == 'linear':
        z = torch.matmul(y, self.W.t()) + self.rho

      return F.log_softmax(z, dim=-1)