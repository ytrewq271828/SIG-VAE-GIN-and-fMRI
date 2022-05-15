from inspect import Parameter
import dgl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import scipy as sp
import networkx as nx

#Generating dgl.graph from adjacency matrix & feature matrix
#node_features : feature matrix
#subX : dimension 1,2 of input matrix of GIN - do not consider sample size
def graph_generator(A, subX):
    tempGraph=nx.from_numpy_array(A)
    
    for index in A.size(0):
        tempGraph.nodes[index]=subX[index]
    tempGraph=tempGraph.to_directed()
        
    finalGraph=dgl.from_networkx(tempGraph, node_attrs=['node_feature'], edge_attrs=['weight'])
    return finalGraph

#Shape of X : [sample_size, N, M]
#output_dims=[output dim of 1st layer, output dim of 2nd layer, ..., D]
#GINConv : Readout not implemented - we should apply readout to the result
#Shape of final result : [sample_size, N, D]
class GIN(nn.Module):
    def __init__(self, L, input_dim, output_dims, noise_dim, sample_size, activation, dropout=0):
        super(GIN, self).__init__()
        self.numLayer=L
        self.input_dim=input_dim
        self.output_dims=output_dims
        self.noise_dim=noise_dim
        self.activation=activation
        self.GINLayers=nn.ModuleList()
        self.dropoutRate=dropout
        self.sample_size=sample_size
        
        if self.output_dims.size(0)!=L:
            raise Exception("number of layer not matched to output dimensions")
        
        self.GINLayers.append(dgl.nn.GINConv(apply_func=nn.Linear(self.input_dim+self.noise_dim,self.output_dims[0]), aggregator_type="sum", init_eps=0, learn_eps=True,activation=self.activation))
        for i in range(L-1):
            self.GINLayers.append(dgl.nn.GINConv(apply_func=nn.Linear(self.output_dims[i], self.output_dims[i+1]),aggregator_type="sum", init_eps=0, learn_eps=True,activation=self.activation))
    
    def forward(self, A, X, h):
        self.adjMatrix=A
        self.featMatrix=X
        self.N=X.size(1)
        self.h=h
        #self.X1=torch.tile(torch.unsqueeze(X, axis=1), [1,K,1]) #(N,K,M)
        
        if X.size(0)!=self.sample_size:
            raise Exception("sample size not matched to the input.shape[0]")
        
        #epsilon generation
        Bernoulli=torch.distributions.Bernoulli(torch.tensor[0.5])
        if self.noise_dim>0:
            epsilon=Bernoulli.rsample(torch.Size(self.sample_size, self.N, self.noise_dim))
            #epsilon=Bernoulli.rsample(torch.Size([self.N,self.K,self.noise_dim]))
            temp=torch.cat((self.featMatrix, epsilon, h), 1)
            #temp=torch.cat((epsilon,self.X1),axis=2) #shape : (N, K, M+noise_dim)
        else:
            epsilon=torch.zeros(self.sample_size, self.N, self.noise_dim)
            temp=torch.cat((self.featMatrix, epsilon, h), 1)
        
        if self.sample_size>=1:
            outputTensor=torch.zeros([self.sample_size])
            self.adjMatrixNumpy=torch.clone(self.adjMatrix).numpy()
            for i in range(self.sample_size):
                self.tempNumpy=torch.clone(temp[i,:,:]).numpy()
                self.inputGraphGIN=graph_generator(self.adjMatrixNumpy, self.tempNumpy) 
                for layer in self.GINLayers:       
                    temp=layer.forward(self.inputGraphGIN, temp, edge_weight=None)
                    temp=torch.mean(temp, dim=0) #Mean READOUT
                outputTensor[i]=temp
        
        #for GINmu and GINsigma
        if self.sample_size==0:
            self.adjMatrixNumpy=torch.clone(self.adjMatrix).numpy()
            self.inputGraphGIN=graph_generator(self.adjMatrixNumpy, self.tempNumpy) 
            for layer in self.GINLayers:       
                temp=layer.forward(self.inputGraphGIN, temp, edge_weight=None)
                temp=torch.mean(temp, dim=0) #Mean READOUT
            outputTensor=temp
        #Return shape : (sample_size, N,D) for GINu / (N,D) for GINmu and GINsigma
        return self.activation(outputTensor), epsilon

#output : reconstructed adjacency amtrix
class InnerProductDecoder(nn.Module):
    def __init__(self, dropout=0):
        super(InnerProductDecoder, self).__init__()
        #self.activation=activation
        self.dropout=nn.Dropout(dropout)
        #self.distribution=distribution
        
    #return logit
    def runDecoder(self, Z):
        Z=self.dropout(Z)
        temp=torch.transpose(torch.clone(Z), 1, 2)
        result=torch.mm(Z,temp)
        #A=self.distribution(temp=1,logits=self.activation(result))
        return torch.nn.Sigmoid(result), Z
    
class BPDecoder(nn.Module):
    def __init__(self, z_dim, dropout=0):
        super(BPDecoder, self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.z_dim=z_dim
        #self.distribution=distribution
        self.rk_logit=Parameter(torch.FloatTensor(torch.size([z_dim, z_dim])))
    #return logit
    def runDecoder(self, Z):
        self.rk=torch.nn.Sigmoid(self.rk_logit)
        Z=self.dropout(Z)
        temp=torch.transpose(torch.clone(Z), 1, 2)
        sigmaInput=self.rk.view(1, self.z_dim, self.z_dim)*torch.mm(Z, temp)
        lambdaterm=torch.exp(torch.sum(sigmaInput))
        #A=self.distribution(temp=1, logits=1-torch.exp(-1*lambdaterm))
        return 1-torch.exp(-1*lambdaterm), Z
    
#Lu : number of layers of GINu
#Lmu : number of layers of GINmu
#Lsigma : number of layers of GINsigma
#output_dim_matrix_u : matrix made by concatenating output_dim vector of each GIN in GINuNetworks (axis=1)
#output_dim_mu : output_dim vector of GINmu
#output_dim_sigma : output_dim vector of GINsigma
class Encoder(nn.Module):
    def __init__(self, Lu, Lmu, Lsigma, input_dim, output_dim_u, output_dim_mu, output_dim_sigma, K, J, noise_dim=64, activation=nn.ReLU, dropout=0) :
        super(Encoder, self).__init__()
        self.K=K
        self.J=J
        self.noise_dim=noise_dim
        self.GINu=GIN(Lu, input_dim=input_dim, output_dims=output_dim_u, noise_dim=noise_dim, sample_size=(self.K+self.J), activation=activation, dropout=0)
        self.GINmu=GIN(Lmu, input_dim=output_dim_u, output_dims=output_dim_mu, noise_dim=0, sample_size=0, activation=lambda x : x, dropout=0)
        self.GINsigma=GIN(Lsigma, input_dim=output_dim_u, output_dims=output_dim_sigma, noise_dim=0, sample_size=0, activation=lambda x : x, dropout=0)
        
        
    def encode(self, A, X):
        h=0
        h, epsilon=self.GINu.forward(A, X, h)
        hL=torch.clone(h)
        #hL's shape : (K+J, N, output_dim_u)
        self.mu, zero_eps=self.GINmu.forward(A, X, hL)
        self.sigma, zero_eps=torch.exp(self.GINsigma.forward(A, X, hL)/2.0)
        
        embedding_mu=self.mu[self.K:,:]
        embedding_sigma=self.sigma[self.K:,:]
        
        if len(embedding_mu.shape)!=len(embedding_sigma.shape):
            raise Exception("mu and sigma have different dimensions")
        
        #sample_n in the original tensorflow code
        param=torch.normal(mean=0, std=1)
        
        #Z is equal to emb in the original code
        Z=embedding_mu+param*embedding_sigma
        
        return Z, self.mu, self.sigma, epsilon
    
class SIGVAE_GIN(nn.Module):
    def __init__(self, Lu, Lmu, Lsigma, input_dim, output_dim_u, output_dim_mu, output_dim_sigma, K, J, noise_dim=64, decoder_type="inner", activation=nn.ReLU, dropout=0):
        super(SIGVAE_GIN, self).__init__()
        self.decoder_type=decoder_type
        #self.Rmatrix=Rmatrix
        self.encoder=Encoder(Lu, Lmu, Lsigma, input_dim, output_dim_u, output_dim_mu, output_dim_sigma, K, J, noise_dim=64, activation=activation, dropout=dropout)
        self.K=K
        self.J=J
        self.noise_dim=noise_dim #ndim in sigvae-torch
        #to make hiddenx + hiddene equivalent to x||e
        self.reweight=((self.noise_dim+output_dim_u) / (input_dim+output_dim_u))**(0.5)    
        
        if self.decoder_type=="inner":
            self.decoder=InnerProductDecoder(dropout=dropout)
        if self.decoder_type=="bp":
            self.decoder=BPDecoder(z_dim=output_dim_mu, dropout=dropout)

    def forward(self, adj_matrix, feat_matrix):
        self.adj_matrix=adj_matrix
        self.feat_matrix=feat_matrix
        self.latent_representation, self.mu, self.sigma, self.epsilon=self.encoder.encode(adj_matrix, feat_matrix)
        if self.decoder_type=="inner":
            self.generated_prob, self.Z=self.decoder.runDecoder(self.latent_representation)
        if self.decoder_type=="bp":
            self.generated_prob, self.Z=self.decoder.runDecoder(self.latent_representation, self.Rmatrix)
        return self.generated_prob, self.mu, self.sigma, self.latent_representation, self.Z, self.epsilon
    
    
        
    
   