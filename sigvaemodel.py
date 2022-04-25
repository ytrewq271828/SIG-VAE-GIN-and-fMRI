import dgl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import scipy as sp


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.dropout=torch.nn.Dropout(p=dropout)
        self.layer=nn.Linear(input_dim, output_dim, bias=False)
        self.layer.reset_parameters()
        
    def forward(self, X, A):
        X=self.dropout(X)
        self.A=A+torch.eye(A.size(0))
        Dhat=torch.diag(torch.diagonal(self.A,0)) #(N,N)
        temp=self.layer(torch.mm(self.A,X)) #(N,M)
        DhatPowerHalf=sp.linalg.fractional_matrix_power(Dhat, (-1/2)) #(N,N)
        #return shape : (N,M)
        return nn.ReLU(torch.mm(torch.mm(torch.mm(DhatPowerHalf, self.A), DhatPowerHalf), temp))

        
#GNN template in SIGVAE
#1 stochastic layer of size 32 + 1 GCN layer of size 16 models mu
#Stochastic layer : epsilon related GCN layer (noise injection layer)
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0):
        super().__init__()
        #self.adjMatrix=A
        #self.featMatrix=X #shape : (N,M)
        #self.N=X.size(0)
        
        self.GCNLayers=nn.ModuleList()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.dropoutRate=dropout
        #self.epsilon=epsilon
        self.GCNLayers.append(GCNLayer(input_dim,hidden_dim,self.dropoutRate))
        self.GCNLayers.append(GCNLayer(hidden_dim,output_dim,self.dropoutRate))    
        
    def forward(self, A, X, epsilon_dim, h):
        self.adjMatrix=A
        self.featMatrix=X
        self.N=X.size(0)
        self.noise_dim=epsilon_dim
        self.h=h #shape : (N,D)
        
        Bernoulli=torch.distributions.Bernoulli(torch.tensor[0.5])
        epsilon=Bernoulli.rsample(torch.Size([self.N,self.noise_dim]))
        
        temp=torch.cat((self.featMatrix,epsilon,h),1) #shape : (N, M+64+D)
        #fcn=nn.Linear(self.output_dim, self.fc_dim)
        for layer in self.GCNLayers:
            temp=layer.forward(temp, A)
        #temp=fcn(temp)
        
        #Return shape : (N,D)
        return temp

#input_dim : M in paper
#input_dim_noise : epsilon's dimension
#output_dim : D in paper
class SIGVAE(nn.Module):
    def __init__(self, L, input_dim, numNodes,input_dim_noise=64, hidden_dim=32, output_dim=16, dropout=0) :
        super().__init__()
        self.GCNuNetworks=nn.ModuleList()
        for i in range(L):
            self.GCNuNetworks.append(GCN(input_dim_noise+input_dim+output_dim, hidden_dim, output_dim, dropout))
        #self.GCNu=GCN(input_dim_noise, hidden_dim, output_dim, dropout)
        self.GCNmu=GCN(input_dim, hidden_dim, output_dim, dropout)
        self.GCNsigma=GCN(input_dim, hidden_dim, output_dim, dropout)
        
        
    def forward(self, A, X):
        h=0
        for GCNu in self.GCNuNetworks:
            h=GCNu.forward(A, X, 64 ,h)
        self.GCNmu.forward(A, X, 0, h)
        self.GCNsigma.forward(A, X, 0, h)