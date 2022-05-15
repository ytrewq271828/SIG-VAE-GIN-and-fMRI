import torch
import numpy as np


def get_rec(norm_constant, weight, adj_matrix, generated_prob):
        log_likelihood=torch.nn.functional.binary_cross_entropy_with_logits(generated_prob, adj_matrix, wieght=weight)
        rec=norm_constant * log_likelihood
        #log_likelihood=norm_constant*(weight*adj_matrix*torch.log(generated_prob)+(1-adj_matrix)*torch.log(1-generated_prob))
        return -rec.mean()

#SIG-VAE loss : https://github.com/YH-UtMSB/sigvae-torch/blob/master/optimizer.py
def loss(adj_matrix, mu, sigma, Z, latent_representation):
        #mean(logp(zj))
    N=adj_matrix.size(0)
    J, N, z_dim=Z.shape
    K=mu.size(0)-J
    mu_mix=mu[:K, :]
    mu_emb=mu[K:, :]
    sigma_mix=sigma[:K, :]
    sigma_emb=sigma[K:, :]

    weight=torch.tensor([float(N * N - adj_matrix.sum()) / adj_matrix.sum()])
    norm=N*N/float((N*N-adj_matrix.sum())*2)
    rec_costs=torch.stack([get_rec(norm, weight, adj_matrix, lr) for lr in torch.unbind(latent_representation, dim=0)])
    rec_cost=rec_costs.mean()
    log_prior_kernel=torch.sum(Z.pow(2)/2.0, dim=[1,2]).mean()
    
    Zcopy=torch.clone(Z)
    Zcopy.view(J, 1, N, z_dim)
    mu_mix=mu_mix.view(1, K, N, z_dim)
    sigma_mix=sigma_mix.view(1, K, N, z_dim)
    
    log_post_kernel_JK=-torch.sum(
        (((Zcopy-mu_mix)/(sigma_mix+1e-6))**2)/2, dim=[-2, -1]
    )
    
    log_post_kernel_JK += -torch.sum(
        (sigma_mix+1e-6).log(), dim=[-2, -1]
    )
    
    log_post_kernel_J = - torch.sum(
    Zcopy.pow(2)/2, dim=[-2,-1]
    )
    
    log_post_kernel_J += - torch.sum(
        (sigma_emb + 1e-6).log(), dim = [-2,-1]
    )
    
    log_post_kernel_J = log_post_kernel_J.view(-1,1)


    # bind up log_post_ker_JK and log_post_ker_J into log_post_ker, the shape of result tensor is [J, K+1].
    log_post_kernel = torch.cat([log_post_kernel_JK, log_post_kernel_J], dim=-1)

    # apply "log-mean-exp" to the above tensor
    log_post_kernel -= np.log(K + 1.) / J
            
    # average over J items.
    log_posterior_kernel = torch.logsumexp(log_post_kernel, dim=-1).mean()

            
            
    return rec_cost, log_prior_kernel, log_posterior_kernel
    '''
    #VAE loss : https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
    def loss(self, input, reconstructed_input, qz):
        K=2000
        J=150
        eps=1e-10
        train_xs_shape=784
        N=self.adj_matrix.size(0)
        adj_train=self.adj_matrix
        
        #Drawing K samples from q(psi|X,A) - final shape : (N, K, J+1, 16=output_dim_mu=output_dim_sigma)
        #J+1 copy of samples of shape (N, K, 16)
        #z_logv : output of GIN network
        #sigma_iw1 : sigma of qphi - since z_logv=log(sigma^2) 
        #psi_iw : output of GIN network - mu of qphi
        #Not explicitly evaluating qphi - implicit approach 
        z_logv=self.sigma #(N, output_dim_sigma[-1])
        z_logv_iw=torch.tile(torch.unsqueeze(z_logv, 1), [1,K,1]) #(N, K, output_dim_mu[-1])
        sigma_iw1=torch.exp(z_logv_iw.type(torch.FloatTensor)/2) 
        sigma_iw2=torch.tile(torch.unsqueeze(sigma_iw1, axis=2), [1, 1, J+1, 1]) #(N, K, J+1, output_dim_mu[-1])
        psi_iw=self.mu #(N, K, output_dim_mu[-1]==output_dim_sigma[-1]) - original paper shape
        psi_iw_vec=torch.mean(psi_iw, axis=1)
        param=torch.normal(mean=0, std=1)
        #sampling Z from q(Z|psi) - z_sample_iw is equal to the Z in the paper
        z_sample_iw=self.mu+param*sigma_iw1 #(N, K, output_dim_mu[-1])
        z_sample_iw2=torch.tile(torch.unsqueeze(z_sample_iw, axis=2), [1, 1, J+1, 1]) #(N, K, J+1, output_dim_mu[-1])

        #reuse=True => sample from same distribution => another sample of mu of qphi
        psi_iw_star=self.encoder.GINmu.forward(self.adj_matrix, self.feat_matrix, 0, self.encoder.GINu.forward(self.adj_matrix, self.feat_matrix, J, 0)) #(N, J, output_dim_mu[-1])  
        psi_iw_star1=torch.tile(torch.unsqueeze(psi_iw_star,axis=1), [1,K,1,1]) #(N, K, J, output_dim_mu[-1])
        psi_iw_star2=torch.cat([psi_iw_star1, torch.unsqueeze(psi_iw, axis=2)],2) #(N, K, J+1, output_dim_mu[-1])
        
        #Bayesian kernel of normal distribution
        #https://namyoungkim.github.io/statistics/2017/09/18/probability/
        ker=torch.exp(-0.5 * torch.sum(torch.square(z_sample_iw2-psi_iw_star2)/torch.square(sigma_iw2+eps),3)) #(N, K, J+1)
        
        
        log_H_iw_vec=torch.log(torch.mean(ker, axis=2)+eps)-0.5*torch.sum(z_logv_iw, 2) #(N,K)
        log_H_iw=torch.mean(log_H_iw_vec, axis=0) #(K)
        
        log_prior_iw_vec=-0.5*torch.sum(torch.square(z_sample_iw),2) #(N,K)
        log_prior_iw=torch.mean(log_prior_iw_vec, axis=0)  #(K)
        
        x=torch.randn(self.adj_matrix.size(0), train_xs_shape)
        x_iw=torch.tile(torch.unsqueeze(x, axis=1),[1,K,1]) #(N, K, 784)
        
        for i in range(K):
            input_=torch.squeeze(z_sample_iw[:,i,:])
            logits_x=torch.nn.Sigmoid(self.decoder.runDecoder(input_))
            if i==0:
                outputs=torch.unsqueeze(logits_x, axis=2)
            else:
                outputs=torch.cat([outputs, torch.unsqueeze(logits_x, axis=2)], axis=2)
        
        logits_x_iw=outputs #(N, N, K) 
        reconstruct_iw=logits_x_iw
        
        adjacency_orig_dense=adj_train+torch.eye(adj_train.shape[0])
        adj_orig_tile=torch.unsqueeze(adjacency_orig_dense, -1)
        adj_orig_tile=torch.tile(adj_orig_tile, mutiplies=[1,1,K]) #(N,N,K)
        
        pos_weight = float(self.adj_matrix.shape[0] * self.adj_matrix.shape[0] - self.adj_matrix.sum()) / self.adj_matrix.sum()
        norm = self.adj_matrix.shape[0] * self.adj_matrix.shape[0] / float((self.adj_matrix.shape[0] * self.adj_matrix.shape[0] - self.adj_matrix.sum()) * 2)
        
        #weighted_cross_entropy_with_logits
        #logit : reconstruct_iw
        
        log_lik_iw = -1 * norm * torch.mean(
            (1-adj_orig_tile)*reconstruct_iw + 
            (1+(pos_weight-1)*adj_orig_tile)*
            torch.log(1+torch.exp(-1*abs(reconstruct_iw))+
            max(-1*reconstruct_iw,0)),axis=[0,1]) #K
        
        loss_iw0=-torch.log(torch.sum(torch.exp(log_lik_iw+(log_prior_iw-log_H_iw)*0/N))) + torch.log(K.type(torch.Float32))
        loss_iw=loss_iw0
        #BCE=nn.Functional.binary_cross_entropy(reconstructed_input, input.view(-1, input.size(0)), reduction='mean')
        #pz=torch.normal.Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
        #KLD=torch.distributions.kl_divergence(qz, pz).mean()
        
        #return BCE + KLD
        BCE=nn.Functional.binary_cross_entropy(reconstructed_input, input.view(-1, input.size(0)), reduction='sum')
        pz=torch.normal.Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
        KLD=torch.distributions.kl_divergence(qz, pz).sum()
        
        return BCE + KLD
        '''