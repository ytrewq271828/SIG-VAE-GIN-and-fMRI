import torch
import numpy as np

'''
def get_rec(norm_constant, weight, adj_matrix, generated_prob):
        adj_matrix=adj_matrix.to('cuda')
        generated_prob=generated_prob.to('cuda')
        log_likelihood=torch.nn.functional.binary_cross_entropy_with_logits(generated_prob, adj_matrix, weight=weight)
        rec=norm_constant * log_likelihood
        #log_likelihood=norm_constant*(weight*adj_matrix*torch.log(generated_prob)+(1-adj_matrix)*torch.log(1-generated_prob))
        return rec.mean()

#SIG-VAE loss : https://github.com/YH-UtMSB/sigvae-torch/blob/master/optimizer.py
def loss(generated_prob, adj_matrix, mu, sigma, Z, epsilon, latent_representation, norm, weight, device='cuda'):
        #mean(logp(zj))
    N=adj_matrix.size(0)
    J, N, z_dim=Z.shape
    K=mu.size(0)-J
    mu_mix=mu[:K, :]
    mu_emb=mu[K:, :]
    sigma_mix=sigma[:K, :]
    sigma_emb=sigma[K:, :]
    generated_prob=torch.clamp(generated_prob, min=1e-6, max=1-(1e-6))
    rec_costs=torch.stack([get_rec(norm, weight, adj_matrix, lr) for lr in torch.unbind(generated_prob, dim=0)], dim=0)
    rec_cost=rec_costs.mean()
    
    log_prior_kernel=torch.sum(Z.pow(2)/-2.0, dim=[1,2]).mean()
    
    Zcopy=torch.clone(Z)
    Zcopy=Zcopy.view(J, 1, N, z_dim)
    mu_mix=mu_mix.view(1, K, N, z_dim)
    sigma_mix=sigma_mix.view(1, K, N, z_dim)
    
    log_post_kernel_JK=-torch.sum(
        (((Zcopy-mu_mix)/(sigma_mix+1e-6))**2)/2, dim=[-2, -1]
    )
    
    log_post_kernel_JK += -torch.sum(
        (sigma_mix+1e-6).log(), dim=[-2, -1]
    )
    
    log_post_kernel_J = - torch.sum(
    epsilon.pow(2)/2, dim=[-2,-1]
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
def loss(generated_prob, adj_matrix, mu, sigma, Z, epsilon, latent_representation, K, J, WU, norm, weight, dropout, device='cuda'):   
    eps=1e-10
    train_xs_shape=1433 #number of features of Cora dataset
    N=adj_matrix.size(0)
    z_dim=mu.shape[2]
    adj_train=adj_matrix
    
    #Drawing K samples from q(psi|X,A) - final shape : (N, K, J+1, 16=output_dim_mu=output_dim_sigma)
    #J+1 copy of samples of shape (N, K, 16)
    #z_logv : output of GIN network
    #sigma_iw1 : sigma of qphi - since z_logv=log(sigma^2) 
    #psi_iw : output of GIN network - mu of qphi
    #Not explicitly evaluating qphi - implicit approach 
    
    #z_logv=sigma #(N, output_dim_sigma[-1])
    #z_logv_iw=torch.tile(torch.unsqueeze(z_logv, 1), [1,K,1]) #(N, K, output_dim_mu[-1])
    
    sigma_iw1=sigma[:K, :, :] #(K, N, z_dim) 
    z_logv_iw=torch.log(sigma_iw1+eps)*2
    sigma_iw2=torch.tile(torch.unsqueeze(sigma_iw1, axis=0), [J+1, 1, 1, 1]) #(J+1, K, N, z_dim)
    
    psi_iw=mu[:K, :, :] #(K, N, z_dim)
    psi_iw_vec=torch.mean(psi_iw, axis=0) #(N, z_dim)
    param=torch.normal(mean=0.0, std=1.0, size=(K, N, z_dim)).to('cuda')
    
    #sampling Z from q(Z|psi) - z_sample_iw is equal to the Z in the paper
    z_sample_iw=psi_iw+param*sigma_iw1 #(K, N, z_dim)
    z_sample_iw2=torch.tile(torch.unsqueeze(z_sample_iw, axis=0), [J+1, 1, 1, 1]) #(J+1, K, N, z_dim)

    #reuse=True => sample from same distribution => another sample of mu of qphi
    psi_iw_star=mu[K:, :, :] #(J, N, z_dim)  
    psi_iw_star1=torch.tile(torch.unsqueeze(psi_iw_star,axis=1), [1,K,1,1]) #(J, K, N, z_dim)
    psi_iw_star2=torch.cat([psi_iw_star1, torch.unsqueeze(psi_iw, axis=0)],0) #(J+1, K, N, z_dim)
    
    #Bayesian kernel of normal distribution
    #https://namyoungkim.github.io/statistics/2017/09/18/probability/
    ker=torch.exp(-0.5 * torch.sum(torch.square(z_sample_iw2-psi_iw_star2)/(torch.square(sigma_iw2)+eps),3)) #(J+1, K, N)
    
    
    log_H_iw_vec=torch.log(torch.mean(ker, axis=0)+eps)-0.5*torch.sum(z_logv_iw, 2) #(K, N)
    log_H_iw=torch.mean(log_H_iw_vec, axis=1) #(K)
    
    log_prior_iw_vec=-0.5*torch.sum(torch.square(z_sample_iw),2) #(K, N)
    log_prior_iw=torch.mean(log_prior_iw_vec, axis=1)  #(K)
    
    x=torch.randn(adj_matrix.size(0), train_xs_shape)
    x_iw=torch.tile(torch.unsqueeze(x, axis=0),[K,1,1]) #(K, N, 1433)
    
    for i in range(K):
        input_=torch.squeeze(z_sample_iw[i, :,:])
        rk=torch.cuda.FloatTensor(torch.Size([z_dim]))
        torch.nn.init.uniform_(rk, a=-6., b=0.)
        rk=torch.sigmoid(rk)
        dropoutlayer=torch.nn.Dropout(p=dropout)
        input_=dropoutlayer(input_)
        x=torch.transpose(input_, 0, 1)
        x=torch.matmul(torch.diag(rk), x)
        x=torch.matmul(input_, x)
        x=torch.clamp(x, min=float('-inf'), max=10)
        logits_x=1-torch.exp(-torch.exp(x))
        #logits_x=torch.nn.Sigmoid(self.decoder.runDecoder(input_))
        if i==0:
            outputs=torch.unsqueeze(logits_x, axis=0)
        else:
            outputs=torch.cat([outputs, torch.unsqueeze(logits_x, axis=0)], axis=0)
    
    logits_x_iw=outputs #(K, N, N) 
    reconstruct_iw=logits_x_iw.to('cuda')
    
    adjacency_orig_dense=adj_train+torch.eye(adj_train.shape[0])
    adj_orig_tile=torch.unsqueeze(adjacency_orig_dense, 0)
    adj_orig_tile=torch.tile(adj_orig_tile, (K, 1, 1)).to('cuda') #(K, N, N)
    
    #pos_weight = float(self.adj_matrix.shape[0] * self.adj_matrix.shape[0] - self.adj_matrix.sum()) / self.adj_matrix.sum()
    #norm = self.adj_matrix.shape[0] * self.adj_matrix.shape[0] / float((self.adj_matrix.shape[0] * self.adj_matrix.shape[0] - self.adj_matrix.sum()) * 2)
    
    #weighted_cross_entropy_with_logits
    #logit : reconstruct_iw
    '''
    log_lik_iw = -1 * norm * torch.mean(
        (1-adj_orig_tile)*reconstruct_iw + 
        (1+(weight-1)*adj_orig_tile)*
        torch.log(1+torch.exp(-1*abs(reconstruct_iw))+
        max(-1*reconstruct_iw,0)),axis=[1,2]) #K
    '''
    log_lik_iw = norm * torch.mean(
      adj_orig_tile * torch.log(reconstruct_iw + eps) *
      weight + (1-adj_orig_tile) * torch.log(1 - reconstruct_iw + eps), axis=[1,2])

    loss_iw0 = -torch.logsumexp(log_lik_iw+(log_prior_iw-log_H_iw)*WU/N+eps, dim=0, keepdim=False) + torch.log(torch.cuda.FloatTensor([K]))
    #BCE=nn.Functional.binary_cross_entropy(reconstructed_input, input.view(-1, input.size(0)), reduction='mean')
    #pz=torch.normal.Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
    #KLD=torch.distributions.kl_divergence(qz, pz).mean()
    
    #return BCE + KLD
    #BCE=torch.nn.Functional.binary_cross_entropy(reconstructed_input, input.view(-1, input.size(0)), reduction='sum')
    #pz=torch.normal.Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
    #KLD=torch.distributions.kl_divergence(qz, pz).sum()
    
    return loss_iw0