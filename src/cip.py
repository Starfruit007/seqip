import torch

class CIP(torch.nn.Module):
    ''' 
        Random Variable Explanation
            Input sample: Xk
            Model outputted random variables: Z = (z1, z2, ..., zN)
            Label of input sample: Yl
    '''
    def __init__(self,forget_num = 2000, stable_num =  1e-20, monte_carlo_num = 2,top_p = None):
        super().__init__()

        self.forget_num = forget_num
        self.stable_num = stable_num
        self.monte_carlo_num = monte_carlo_num
        self.top_p = top_p
        self.distribution = torch.distributions.Normal(0, 1)


    def load_params(self,params_T,y_true_T):
        self.params_T = params_T
        self.y_true_T = y_true_T


    def forward(self, logits):
        '''
            calculation of the loss and the inference probability $P^{Z}(Yl|Xk)$ with posterior.
            Args:
                logits: output nodes of neural network as paramters of some prior distribution, such as Gaussian distribution.
                y_true: label of input samples

            shape symbols:
                b --> batch size                       = logits[0].shape[0]
                s --> sequence length                  = logits[0].shape[1]
                n --> dimension of variables           = logits[0].shape[2]
                t --> forget number                    = self.forget_num
                m --> Monte Carlo number               = self.monte_carlo_num
                y --> output dimension of variables    = y_true_T.shape[-1] = n
        '''

        # Take, gaussian distribution for example, the model outputted logits is the parameters $\mu, \sigma$.
        params = logits #[b,s,n]

        # random samples and reparamterization trick to get latent variable: Z = (z1,z2,...,zN)
        # shape: [b,m,n]
        z = self.gaussian_sample_and_reparameterize(params)

        # oberservation and inference to get the posterior: $P^{Z}(Yl \mid Xk)$
        # shape: [b,y]
        probability = self.observation_and_inference(z, self.params_T, self.y_true_T)
        
        return probability
    
    def de_params(self,params):
        mean, log_var = params # shape: [b,s,n] or [b,s,t]
        return mean, log_var

    def gaussian_sample_and_reparameterize(self, params):
        ''' random samples from normal gaussian distribution (noise distribution)
            and reparameterization trick to have the algorithm differentiable'''
        mean,log_var = self.de_params(params) # shape: [b,s,n]
        # random samples
        eps = self.distribution.sample([self.monte_carlo_num] + list(log_var.shape)).to(mean.device)  # shape: [m,b,n]
        std = torch.exp(log_var/2)

        # reparameterization trick.
        z = mean + eps * std
        z = torch.permute(z,(1,0,2,3)) # shape: [m,b,s,n] --> [b,m,s,n]
        return z

    def observation_and_inference(self,z,params_T,y_true_T):
        ''' oberservation and inference to get the posterior: $P^{Z}(Yl \mid Xk)
            Because the posterior is formulated as expectation and uses monte carlo method to approXkmate it,
            the observation and inference phase calculation is then not able to be sperated like the algorithm in IPNN. '''
        
        # input shape: latent_vars - [b,m,s,n], params_T - [t,s,n]
        # output shape: [b,m,t]
        joint_probs = self.product_gaussian(z,params_T)
        
        num_y_joint = torch.einsum('bmt,tsy->bmsy',joint_probs,y_true_T) # shape: [b,m,s,y]
        num_joint = torch.unsqueeze(torch.unsqueeze(torch.einsum('bmt->bm',joint_probs),dim=-1),dim=-1) # shape: [b,m,1,1]

        probs = torch.clamp_min(num_y_joint,self.stable_num)/torch.clamp_min(num_joint,self.stable_num) # shape: [b,m,s,y]

        probability = torch.sum(probs,dim=1) / num_joint.shape[1] # shape:  [b,m,s,y] -->  [b,s,y]

        probability = torch.clamp(probability,0,1)

        return probability


    def product_gaussian(self, z, params):
        ''' substitute latent variable z into gausstion distribution function to get the probability. '''
        mean,log_var = self.de_params(params) # shape: [t,s,n]
        var = torch.clamp_min(torch.exp(log_var),1e-20)
        z_ = torch.unsqueeze(z,dim=2) # shape: [b,m,s,n] --> [b,m,1,s,n]
        
        # substitute z value into gaussian distribution function to get the probability.
        # shape of z_ - mean: [b,m,1,s,n] - [t,s,n] --> [b,m,t,s,n]
        p = 1/torch.sqrt(2*torch.pi*var) * torch.exp(-torch.square(z_-mean)/(2*var))

        # expectation scaler factor for large latent space.
        std = torch.exp(log_var[0,0,:]/2)
        p *= 4.13273*std

        if self.top_p is not None:
            sorted_outs, _ = torch.sort(torch.reshape(p,list(p.shape)[:3]+[-1]), descending=True)
            sorted_reserved = sorted_outs[:,:,:,:int(self.top_p*sorted_outs.shape[-1])]
            joint_probs = torch.prod(sorted_reserved,dim=-1)
        else:
            joint_probs = torch.prod(torch.prod(p,dim=-1),dim=-1) # shape: [b,m,t,s,n] --> [b,m,t]

        joint_probs = torch.clamp_max(joint_probs,1e20) # avoid inf value.

        return joint_probs
