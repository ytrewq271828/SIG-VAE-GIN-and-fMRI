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
        
        if len(self.output_dims)!=L:
            raise Exception("number of layer not matched to output dimensions")
        
        self.GINLayers.append(dgl.nn.GINConv(apply_func=nn.Linear(self.input_dim+self.noise_dim+32,self.output_dims[0]), aggregator_type="sum", init_eps=0, learn_eps=True, activation=self.activation).to('cuda'))
        for i in range(L-1):
            self.GINLayers.append(dgl.nn.GINConv(apply_func=nn.Linear(self.output_dims[i], self.output_dims[i+1]),aggregator_type="sum", init_eps=0, learn_eps=True, activation=self.activation).to('cuda'))
    
    #return shape : (sample_size, N,D)
    def forward(self, A, X, input_graph, h):
        self.adjMatrix=A
        self.featMatrix=X.expand(self.sample_size, -1, -1).to('cuda')
        self.N=X.shape[1]
        self.h=h.expand(self.sample_size, self.N, 32).to('cuda')
        #self.X1=torch.tile(torch.unsqueeze(X, axis=1), [1,K,1]) #(N,K,M)
        
        if len(X.shape)!=3:
            raise Exception("dimension of input feature matrix is not 2")
        #if X.shape[0]!=self.sample_size:
        #    raise Exception("sample size not matched to the input.shape[0]")
       
        #epsilon generation
        bernoulli=torch.distributions.Bernoulli(torch.tensor([0.5]))
        if self.noise_dim>0:
            epsilon=bernoulli.sample(torch.Size([self.sample_size, self.N, self.noise_dim])).view(self.sample_size, self.N, self.noise_dim).to('cuda')
            temp=torch.cat((self.featMatrix, epsilon, self.h), axis=2).to('cuda')
        else:
            epsilon=torch.zeros(self.sample_size, self.N, self.noise_dim).to('cuda')
            temp=torch.cat((self.featMatrix, epsilon, self.h), axis=2).to('cuda')
        
        #for all three GINs : GINu and GINmu and GINsigma
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
                  tempdim12=torch.mean(tempdim12, dim=0) #Mean READOUT
                  
                outputTensor[i]=tempdim12
        '''
        if self.sample_size==0:
            #self.adjMatrixNumpy=torch.clone(self.adjMatrix).numpy()
            tempdim12=temp.squeeze(0)
            self.inputGraphGIN=graph_generator(self.adjMatrix, {'feature':self.tempdim12}) 
            #tempNumpy=temp.squeeze(0).numpy()
            for layer in self.GINLayers:       
                tempdim12=layer.forward(self.inputGraphGIN, tempdim12, edge_weight=None)
                tempdim12=torch.mean(tempdim12, dim=0) #Mean READOUT
            outputTensor=tempdim12
        #Return shape : (sample_size, N,D) for GINu / (N,D) for GINmu and GINsigma
        '''
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
        result=torch.matmul(Z,temp)
        #A=self.distribution(temp=1,logits=self.activation(result))
        return torch.sigmoid(result), Z
    
class BPDecoder(nn.Module):
    def __init__(self, z_dim, J, dropout=0):
        super(BPDecoder, self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.z_dim=z_dim[-1]
        #self.distribution=distribution
        self.rk_logit=torch.nn.Parameter(torch.cuda.FloatTensor(torch.Size([1, self.z_dim])))
    #return logit
    def runDecoder(self, Z):
        rk=torch.sigmoid(self.rk_logit).pow(0.5)
        Z=self.dropout(Z)
        Z=Z.mul(rk.view(1, 1, self.z_dim))
        temp=torch.transpose(torch.clone(Z), 1, 2)
        sigmaInput=torch.bmm(Z, temp)
        #print(torch.max(sigmaInput))
        sigmaInput=torch.clamp(sigmaInput, min=float('-inf'), max=10)
        lambdaterm=torch.exp(sigmaInput)
        #lambdaterm=torch.clamp(lambdaterm, min=0, max=1000 )
        #A=self.distribution(temp=1, logits=1-torch.exp(-1*lambdaterm))
        return 1-torch.exp(-1*lambdaterm), Z
    
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
        self.GINu=GIN(Lu, input_dim=input_dim, output_dims=output_dim_u, noise_dim=noise_dim, sample_size=(self.K+self.J),  activation=activation, dropout=0)
        self.GINmu=GIN(Lmu, input_dim=input_dim, output_dims=output_dim_mu, noise_dim=0, sample_size=(self.K+self.J), activation=lambda x : x, dropout=0)
        self.GINsigma=GIN(Lsigma, input_dim=input_dim, output_dims=output_dim_sigma, noise_dim=0, sample_size=(self.K+self.J), activation=lambda x : x, dropout=0)
        
    #input X's shape : (1, N, M)
    def encode(self, A, X, input_graph):
        h=torch.zeros(1, 1, 1).to('cuda')
        
        h, epsilon=self.GINu.forward(A, X, input_graph, h)
        hL=torch.clone(h)
        #hL's shape : (K+J, N, output_dim_u)
        self.mu, zero_eps=self.GINmu.forward(A, X, input_graph, hL)
        self.sigma, zero_eps=self.GINsigma.forward(A, X, input_graph, hL)
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
        Z=param.mul(embedding_sigma).add(embedding_mu).to('cuda')
        epsilon=param
        #print("#################################################")
        #print(Z)
        return Z, self.mu, self.sigma, epsilon
    
class SIGVAE_GIN(nn.Module):
    def __init__(self, Lu, Lmu, Lsigma, input_dim, output_dim_u, output_dim_mu, output_dim_sigma, K, J, device, noise_dim=64, decoder_type="bp", activation=nn.functional.relu, dropout=0):
        super(SIGVAE_GIN, self).__init__()
        self.decoder_type=decoder_type
        #self.Rmatrix=Rmatrix
        self.encoder=Encoder(Lu, Lmu, Lsigma, input_dim, output_dim_u, output_dim_mu, output_dim_sigma, K, J, noise_dim=64, activation=activation, dropout=dropout)
        self.K=K
        self.J=J
        self.noise_dim=noise_dim #ndim in sigvae-torch
        self.device=device
        #to make hiddenx + hiddene equivalent to x||e
        #self.reweight=((self.noise_dim+output_dim_u) / (input_dim+output_dim_u))**(0.5)    
        
        if self.decoder_type=="inner":
            self.decoder=InnerProductDecoder(dropout=dropout)
        if self.decoder_type=="bp":
            self.decoder=BPDecoder(z_dim=output_dim_mu, J=self.J, dropout=dropout)

    def forward(self, adj_matrix, feat_matrix, input_graph):
        self.adj_matrix=adj_matrix
        self.feat_matrix=feat_matrix
        self.latent_representation, self.mu, self.sigma, self.epsilon=self.encoder.encode(adj_matrix, feat_matrix, input_graph)
        if self.decoder_type=="inner":
            self.generated_prob, self.Z=self.decoder.runDecoder(self.latent_representation)
        if self.decoder_type=="bp":
            self.generated_prob, self.Z=self.decoder.runDecoder(self.latent_representation)
        return self.generated_prob.to(self.device), self.mu, self.sigma, self.latent_representation, self.Z, self.epsilon