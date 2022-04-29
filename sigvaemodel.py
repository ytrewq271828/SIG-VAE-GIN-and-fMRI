import dgl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import scipy as sp
import networkx as nx
'''
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
'''
def graph_generator(self, A, X):
        tempGraph=nx.from_numpy_array(A)
        
        for index in A.size(0):
            tempGraph.nodes[index]=X[index]
        tempGraph=tempGraph.to_directed()
        
        finalGraph=dgl.from_networkx(tempGraph, node_attrs=['node_feature'], edge_attrs=['weight'])
        return finalGraph
    
#output_dims=[output dim of 1st layer, output dim of 2nd layer, ..., D]
class GIN(nn.Module):
    def __init__(self, L, output_dims, activation=torch.nn.ReLU,dropout=0):
        super(GIN, self).__init__()
        self.numLayer=L
        self.output_dims=output_dims
        self.activation=activation
        self.GINLayers=nn.ModuleList()
        self.dropoutRate=dropout
        
        if self.output_dims.size(0)!=L:
            raise Exception("number of layer not matched to output dimensions")
        
        self.GINLayers.append(dgl.nn.GINConv(apply_func=nn.Linear(L,output_dims[0]), aggregator_type="sum", init_eps=0, learn_eps=True,activation=self.activation))
        for i in range(L-1):
            self.GINLayers.append(dgl.nn.GINConv(apply_func=nn.Linear(output_dims[i], output_dims[i+1]),aggregator_type="sum", init_eps=0, learn_eps=True,activation=self.activation))
    
    #Generating dgl.graph from adjacency matrix & feature matrix
    #node_features : feature matrix
    
    def forward(self, A, X, epsilon_dim, h):
        self.adjMatrix=A
        self.featMatrix=X
        self.N=X.size(0)
        self.noise_dim=epsilon_dim
        self.h=h
        
        #epsilon generation
        Bernoulli=torch.distributions.Bernoulli(torch.tensor[0.5])
        epsilon=Bernoulli.rsample(torch.Size([self.N,self.noise_dim]))
        
        temp=torch.cat((self.featMatrix,epsilon,h),1) #shape : (N, M+64+D)
        
        self.adjMatrixNumpy=torch.clone(self.adjMatrix).numpy()
        self.tempNumpy=torch.clone(self.temp).numpy()
        self.inputGraphGIN=graph_generator(self.adjMatrixNumpy, self.tempNumpy) 
       
        for layer in self.GINLayers:
            temp=layer.forward(self.inputGraphGIN, temp, edge_weight=None)
        
        #Return shape : (N,D)
        return temp
    
class InnerProductDecoder(nn.Module):
    def __init__(self, distribution=torch.distributions.RelaxedBernoulli,dropout=0, activation=nn.Sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.activation=activation
        self.dropout=nn.Dropout(dropout)
        self.distribution=distribution
    def runDecoder(self, Z):
        Z=self.dropout(Z)
        temp=torch.transpose(torch.clone(Z), 0, 1)
        result=torch.mm(Z,temp)
        A=self.distribution(temp=1,logits=self.activation(result))
        return A
    
class BPDecoder(nn.Module):
    def __init__(self, distribution=torch.distributions.RelaxedBernoulli,dropout=0):
        super(BPDecoder, self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.distribution=distribution
        
    def runDecoder(self, Z, R):
        self.R=R
        Z=self.dropout(Z)
        temp=torch.transpose(torch.clone(Z), 0, 1)
        sigmaInput=torch.diag(R)*torch.mm(Z, temp)
        lambdaterm=torch.exp(torch.sum(sigmaInput))
        A=self.distribution(temp=1, logits=1-torch.exp(-1*lambdaterm))
        return A
#Lu : number of layers of each GIN in GINuNetworks (same for every GIN)
#Lmu : number of layers of GINmu
#Lsigma : number of layers of GINsigma
#output_dim_matrix_u : matrix made by concatenating output_dim vector of each GIN in GINuNetworks (axis=1)
#output_dim_mu : output_dim vector of GINmu
#output_dim_sigma : output_dim vector of GINsigma
class Encoder(nn.Module):
    def __init__(self, Lu, Lmu, Lsigma, output_dim_matrix_u, output_dim_mu, output_dim_sigma, activation=nn.ReLU, dropout=0) :
        super(Encoder, self).__init__()
        self.GINuNetworks=nn.ModuleList()
        for i in range(Lu):
            self.GINuNetworks.append(GIN(Lu, output_dims=output_dim_matrix_u[i], activation=nn.ReLU, dropout=0))
        self.GINmu=GIN(Lmu, output_dims=output_dim_mu, activation=lambda x : x, dropout=0)
        self.GINsigma=GIN(Lsigma, output_dims=output_dim_sigma, activation=lambda x : x, dropout=0)
        
        
    def encode(self, A, X):
        h=0
        for GINu in self.GINuNetworks:
            h=GINu.forward(A, X, 64, h)
        hL=torch.clone(h)
        self.mu=self.GINmu.forward(A, X, 0, hL)
        self.sigma=torch.exp(self.GINsigma.forward(A, X, 0, hL))
        #self.q=1
        #for index in range(X.size(0)):
        #    qzi=torch.distributions.Normal(loc=self.mu[index], covariance_matrix=torch.diag(self.sigma[index]))
        #    zi=qzi.rsample()
        
        #sample_n in the original tensorflow code
        param=torch.normal(mean=0, std=1)
        Z=self.mu+param*self.sigma
        return Z         
    
class SIGVAE_GIN(nn.Module):
    def __init__(self, Lu, Lmu, Lsigma, output_dim_matrix_u, output_dim_mu, output_dim_sigma, decoder_type, Rmatrix, activation=nn.ReLU, dropout=0):
        super(SIGVAE_GIN, self)._init__()
        self.decoder_type=decoder_type
        self.Rmatrix=Rmatrix
        self.encoder=Encoder(Lu, Lmu, Lsigma, output_dim_matrix_u, output_dim_mu, output_dim_sigma, activation=activation, dropout=dropout)
        if self.decoder_type=="inner":
            self.decoder=InnerProductDecoder(dropout=dropout, distribution=torch.distributions.RelaxedBernoulli,activation=nn.Sigmoid)
        if self.decoder_type=="bp":
            self.decoder=BPDecoder(dropout=dropout, distribution=torch.distributions.RelaxedBernoulli)
    
    def forward(self, adj_matrix, feat_matrix):
        self.latent_representation=self.encoder.encode(adj_matrix, feat_matrix)
        if self.decoder_type=="inner":
            self.generated_result=self.decoder.runDecoder(self.latent_representation)
        if self.decoder_type=="bp":
            self.generated_result=self.decoder.runDecoder(self.latent_representation, self.Rmatrix)
            
        return self.latent_representation, self.generated_result
    
    #VAE loss : https://github.com/kampta/pytorch-distributions/blob/master/binconcrete_vae.py
    def loss(self, input, reconstructed_input, mu, prior, eps=1e-10):
        temp=torch.distributions.Bernoulli(logits=reconstructed_input)
        BCE=-1*temp.log_prob(input.view(-1,input.size(0))).sum()
        temp1=mu*((mu+eps)/prior).log()
        temp2=(1-mu)*((1-mu+eps)/(1-prior)).log()
        KLD=torch.sum(temp1+temp2,dim=-1).sum()
        
        return BCE + KLD