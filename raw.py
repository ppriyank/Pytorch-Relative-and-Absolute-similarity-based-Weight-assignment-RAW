class RAW_LOSS(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma =0.5, use_gpu=True):
        super(RAW_LOSS, self).__init__()
        self.use_gpu = use_gpu
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, x, labels):
        '''
        x : feature vector : (n x d)
        labels : (n,)
        '''
        x = nn.functional.normalize(x, p=2, dim=1) # normalize the features
        n = x.size(0)
        
        #pair wise dot product
        s = torch.mm(x,x.t()) 
        
        if not self.use_gpu:
            p_mask = labels.expand(n, n).eq(labels.expand(n, n).t()).float() 
        else:
            p_mask = labels.expand(n, n).eq(labels.expand(n, n).t()).float().cuda()
        n_mask = 1 -  p_mask
        
        #cosine distance 
        dist = 1- s
        #farthest point in x with positive label 
        _, indices  = (dist *p_mask).sort()
        Hardest_positive = indices[:,-1]
        #closest point in x with negative label
        _, indices  = (dist * n_mask + 10 * p_mask).sort()
        Hardest_negative = indices[:,0]
        
        #outer circle Hardest positive 
        dist_pos_limit = dist.gather(1, Hardest_positive.view(-1,1))
        
        #Smaller Circle Hardest negative  
        dist_neg_limit = dist.gather(1, Hardest_negative.view(-1,1))
        
        #Valid Triplet Loss
        valid_neg = (dist < dist_pos_limit).float() * n_mask
        valid_pos = (dist < dist_neg_limit).float() * p_mask
        
        A = 1 + torch.exp((-1 * alpha *(s  - gamma) * valid_pos).sum(-1))
        B = 1 + torch.exp((  beta *(s  - gamma) * valid_neg).sum(-1))
        
        RAW_error = (torch.log(A) / alpha + torch.log(B) / beta).mean()
        
        return RAW_error 



