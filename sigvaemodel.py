from inspect import Parameter
import dgl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import networkx as nx

#Generating dgl.graph from adjacency matrix & feature matrix
#node_features : feature matrix
#subX : dictionary - key : 'feature', value : dimension 1,2 of input matrix of GIN - do not consider sample size
def graph_generator(inputGraph, X):
    #tempGraph=nx.from_numpy_array(A, create_using=nx.DiGraph)
    #for node in tempGraph.nodes():
    #  for feat, featVal in subX.items():
    #    tempGraph.nodes[node][feat]=featVal[node]
    #node_attr=list(subX.keys())
    #finalGraph=dgl.from_networkx(tempGraph, node_attrs=node_attr, edge_attrs=['weight'])
    
    #src, dst=np.nonzero(A)
    #finalGraph=dgl.graph((src, dst)).to('cuda')
    inputGraph.ndata['feature']=X
    return inputGraph

#Shape of input X : [1, N, M]
#output_dims=[output dim of 1st layer, output dim of 2nd layer, ..., D]
#Shape of final result : [sample_size, N, D]
class GINWithNoise(nn.Module):
    def __init__(self, L, input_dim, output_dims, noise_dim, sample_size, activation, dropout=0):
        super(GINWithNoise, self).__init__()
        self.numLayer=L
        self.input_dim=input_dim
        self.output_dims=output_dims
        self.noise_dim=noise_dim
        self.activation=activation
        self.GINLayers=nn.ModuleList()
        self.dropoutRate=dropout
        self.sample_size=sample_size
        
        if len(self.output_dims)!=L:
            raise Exception("number of layer not matched to output dimensions")
        
        self.GINLayers.append(dgl.nn.GINConv(apply_func=nn.Linear(self.input_dim+self.noise_dim,self.output_dims[0]), aggregator_type="sum", init_eps=0, learn_eps=True, activation=self.activation).to('cuda'))
        for i in range(L-1):
            self.GINLayers.append(dgl.nn.GINConv(apply_func=nn.Linear(self.output_dims[i], self.output_dims[i+1]),aggregator_type="sum", init_eps=0, learn_eps=True, activation=self.activation).to('cuda'))
    
    #return shape : (sample_size, N,D)
    def forward(self, A, X, input_graph):
        self.adjMatrix=A
        self.featMatrix=X.expand(self.sample_size, -1, -1).to('cuda')
        self.N=X.shape[1]
        #self.X1=torch.tile(torch.unsqueeze(X, axis=1), [1,K,1]) #(N,K,M)
        
        if len(X.shape)!=3:
            raise Exception("dimension of input feature matrix is not 2")
        #if X.shape[0]!=self.sample_size:
        #    raise Exception("sample size not matched to the input.shape[0]")
       
        #epsilon generation
        bernoulli=torch.distributions.Bernoulli(torch.tensor([0.5]))
        epsilon=bernoulli.sample(torch.Size([self.sample_size, self.N, self.noise_dim])).view(self.sample_size, self.N, self.noise_dim).to('cuda')
        temp=torch.cat((self.featMatrix, epsilon), axis=2).to('cuda') #shape : (sample_size, N, X.shape[2]+64)

        #for GINu
        if self.sample_size>=1:
            outputTensor=torch.zeros([self.sample_size, self.N, self.output_dims[-1]]).to('cuda')
            #self.adjMatrixNumpy=torch.clone(self.adjMatrix)
            for i in range(self.sample_size):
                tempdim12=temp[i,:,:]
                inputGraphGIN=graph_generator(input_graph, tempdim12) 
                #print(self.inputGraphGIN.type())
                self.inputGraphGIN=inputGraphGIN.to('cuda')
                for layer in self.GINLayers:    
                  tempdim12=layer.forward(self.inputGraphGIN, tempdim12, edge_weight=None).to('cuda')
                  #tempdim12=torch.mean(tempdim12, dim=0) #Mean READOUT
                outputTensor[i]=tempdim12
        return self.activation(outputTensor), epsilon

class GINWithoutNoise(nn.Module):
    def __init__(self, L, input_dim, output_dims, sample_size, activation=lambda x : x, dropout=0):
        super(GINWithoutNoise, self).__init__()
        self.numLayer=L
        self.input_dim=input_dim
        self.output_dims=output_dims
        self.activation=activation
        self.GINLayers=nn.ModuleList()
        self.dropoutRate=dropout
        self.sample_size=sample_size
        
        if len(self.output_dims)!=L:
            raise Exception("number of layer not matched to output dimensions")
        
        self.GINLayers.append(dgl.nn.GINConv(apply_func=nn.Linear(self.input_dim, self.output_dims[0]), aggregator_type="sum", init_eps=0, learn_eps=True, activation=self.activation).to('cuda'))
        for i in range(L-1):
            self.GINLayers.append(dgl.nn.GINConv(apply_func=nn.Linear(self.output_dims[i], self.output_dims[i+1]),aggregator_type="sum", init_eps=0, learn_eps=True, activation=self.activation).to('cuda'))
    
    #return shape : (sample_size, N,D)
    def forward(self, A, X, input_graph, h):
        self.adjMatrix=A
        self.featMatrix=X.expand(self.sample_size, -1, -1).to('cuda')
        self.N=X.shape[1]
        self.h=h
        
        if len(X.shape)!=3:
            raise Exception("dimension of input feature matrix is not 2")
       
        temp=torch.cat((self.featMatrix, self.h), axis=2).to('cuda') #shape : (sample_size, N, X.shape[2]+32)
        
        #for GINmu and GINsigma
        outputTensor=torch.zeros([self.sample_size, self.N, self.output_dims[-1]]).to('cuda')
      
        for i in range(self.sample_size):
            tempdim12=temp[i,:,:]
            inputGraphGIN=graph_generator(input_graph, tempdim12) 
            #print(self.inputGraphGIN.type())
            self.inputGraphGIN=inputGraphGIN.to('cuda')
            for layer in self.GINLayers:    
              tempdim12=layer.forward(self.inputGraphGIN, tempdim12, edge_weight=None).to('cuda')
            outputTensor[i]=tempdim12
        return self.activation(outputTensor)

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
        result=torch.matmul(Z,temp)
        #A=self.distribution(temp=1,logits=self.activation(result))
        return torch.sigmoid(result), Z
    
class BPDecoder(nn.Module):
    def __init__(self, z_dim, J, dropout=0):
        super(BPDecoder, self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.z_dim=z_dim
        self.J=J
        #self.distribution=distribution
        self.rk_logit=torch.nn.Parameter(torch.cuda.FloatTensor(torch.Size([self.z_dim])))
    #return logit
    def runDecoder(self, Z):
        torch.nn.init.uniform_(self.rk_logit, a=-6., b=0.)
        rk=torch.sigmoid(self.rk_logit)
        Z=self.dropout(Z)
        #adj_recon=torch.zeros(self.J, Z.shape[1], Z.shape[1])
        rk_expand=torch.tile(torch.diag(rk), (self.J,1,1))
        X=torch.transpose(Z, 1, 2)
        X=torch.bmm(rk_expand, X)
        X=torch.bmm(Z, X)
        X=torch.clamp(X, min=float('-inf'), max=10)
        adj_recon=1-torch.exp(-torch.exp(X))
        '''
        for i, z in enumerate(Z):
          x = torch.transpose(z, 0, 1)
          x = torch.matmul(torch.diag(rk), x)
          x = torch.matmul(z, x)
          x=torch.clamp(x, min=float('-inf'), max=10)
          outputs = 1 - torch.exp(- torch.exp(x))
          adj_recon[i]=outputs
        '''
        #ZClone=torch.clone(Z)
        #diagonal=torch.zeros(self.J, self.z_dim, self.z_dim).to('cuda')
        #for i, z in enumerate(rk):
        #  diagonal[i]=torch.diag(z)
        
        #temp=torch.bmm(diagonal, torch.transpose(Z, 1, 2)).to('cuda')
        #sigmaInput=torch.bmm(ZClone, temp)
        #sigmaInput=torch.clamp(sigmaInput, min=float('-inf'), max=10)
        #lambdaterm=torch.exp(sigmaInput)
        #print("^^^^^^^^^^^^")
        #print(lambdaterm.shape)
        #lambdaterm=torch.clamp(lambdaterm, min=0, max=1000 )
        #A=self.distribution(temp=1, logits=1-torch.exp(-1*lambdaterm))
        #adj_recon=1-torch.exp(-1*lambdaterm)
        
        #if not self.training:
        #  adj_recon=torch.mean(adj_recon, dim=0, keepdim=True)
        
        return adj_recon, Z, rk
    
#Lu : number of layers of GINu
#Lmu : number of layers of GINmu
#Lsigma : number of layers of GINsigma
#output_dim_matrix_u : matrix made by concatenating output_dim vector of each GIN in GINuNetworks (axis=1)
#output_dim_mu : output_dim vector of GINmu
#output_dim_sigma : output_dim vector of GINsigma
class Encoder(nn.Module):
    def __init__(self, Lu, Lmu, Lsigma, input_dim, output_dim_u, output_dim_mu, output_dim_sigma, K, J, noise_dim=64, activation=nn.functional.relu, dropout=0) :
        super(Encoder, self).__init__()
        self.K=K
        self.J=J
        self.noise_dim=noise_dim
        self.GINu=GINWithNoise(Lu, input_dim=input_dim, output_dims=output_dim_u, noise_dim=noise_dim, sample_size=(self.K+self.J),  activation=activation, dropout=0)
        self.GINmu=GINWithoutNoise(Lmu, input_dim=input_dim+output_dim_u[-1], output_dims=output_dim_mu, sample_size=(self.K+self.J), activation=lambda x : x, dropout=0)
        self.GINsigma=GINWithoutNoise(Lsigma, input_dim=input_dim+output_dim_u[-1], output_dims=output_dim_sigma, sample_size=(self.K+self.J), activation=lambda x : x, dropout=0)
        
    #input X's shape : (1, N, M)
    def encode(self, A, X, input_graph):
        h, epsilon=self.GINu.forward(A, X, input_graph)
        hL=h
        #hL's shape : (K+J, N, output_dim_u)
        self.mu=self.GINmu.forward(A, X, input_graph, hL)
        self.sigma=self.GINsigma.forward(A, X, input_graph, hL)
        self.sigma=torch.exp(self.sigma * 0.5)
        
        #print("$$$$$$$$$$$$$$$$$$$$$$$$$")
        #print(self.mu.shape)
        #print(self.sigma.shape)
        embedding_mu=self.mu[self.K:,:, :]
        embedding_sigma=self.sigma[self.K:,:, :]
        #print(embedding_mu.shape)
        #print(embedding_sigma.shape)
        if len(embedding_mu.shape)!=len(embedding_sigma.shape):
            raise Exception("mu and sigma have different dimensions")
        
        #sample_n in the original tensorflow code
        #stdTensor=torch.ones(embedding_sigma.shape).to('cuda')
        #param=torch.normal(mean=0.0, std=stdTensor)
        
        param=torch.randn_like(embedding_sigma)
        #Z is equal to emb in the original code
        Z=(param*embedding_sigma+embedding_mu).to('cuda')
        epsilon=param
        #print("#################################################")
        #print(Z)
        return Z, self.mu, self.sigma, epsilon
    
class SIGVAE_GIN(nn.Module):
    def __init__(self, Lu, Lmu, Lsigma, input_dim, output_dim_u, output_dim_mu, output_dim_sigma, K, J, device, noise_dim, decoder_type="bp", activation=nn.functional.relu, dropout=0):
        super(SIGVAE_GIN, self).__init__()
        self.decoder_type=decoder_type
        #self.Rmatrix=Rmatrix
        self.encoder=Encoder(Lu, Lmu, Lsigma, input_dim, output_dim_u, output_dim_mu, output_dim_sigma, K, J, noise_dim, activation=activation, dropout=dropout)
        self.K=K
        self.J=J
        self.noise_dim=noise_dim #ndim in sigvae-torch
        self.device=device
        #to make hiddenx + hiddene equivalent to x||e
        #self.reweight=((self.noise_dim+output_dim_u) / (input_dim+output_dim_u))**(0.5)    
        
        if self.decoder_type=="inner":
            self.decoder=InnerProductDecoder(dropout=dropout)
        if self.decoder_type  =="bp":
            self.decoder=BPDecoder(z_dim=output_dim_mu[-1], J=self.J, dropout=dropout)

    def forward(self, adj_matrix, feat_matrix, input_graph):
        self.adj_matrix=adj_matrix
        self.feat_matrix=feat_matrix
        self.latent_representation, self.mu, self.sigma, self.epsilon=self.encoder.encode(adj_matrix, feat_matrix, input_graph)
        if self.decoder_type=="inner":
            self.generated_prob, self.Z, self.rk=self.decoder.runDecoder(self.latent_representation)
        if self.decoder_type=="bp":
            self.generated_prob, self.Z, self.rk=self.decoder.runDecoder(self.latent_representation)
        return self.generated_prob.to(self.device), self.mu, self.sigma, self.latent_representation, self.Z, self.epsilon, self.rk