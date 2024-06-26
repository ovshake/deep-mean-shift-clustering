import torch

class Mean_Shift(torch.nn.Module):

    def __init__(self, delta, eta, ms_iter, use_gpu):
        super(Mean_Shift, self).__init__()
        self.delta = delta
        self.eta = eta
        self.ms_iter = ms_iter

    def mean_shift_once(self, X):
        S = torch.mm(X.t(), X)
        K = torch.exp(self.delta * self.delta * S)
        
        N = list(X.size())[1]
        d = torch.mm(K.t(), torch.ones(N, 1).cuda())
        
        q = 1 / d
        D_inv = torch.diagflat(q)
        
        eye = torch.eye(N)
        if use_gpu:
            eye = eye.cuda()
        
        P = ((1-self.eta) * eye) + (self.eta * torch.mm(K, D_inv))
        
        new_X = torch.mm(X, P)
        
        norm = new_X.norm(p=2, dim=0, keepdim=True).detach()
        new_norm_X = new_X.div(norm.expand_as(new_X))
        
        return new_norm_X
    
    def forward(self, z):
        # z dims: z_len x batch_size
        
        trajects = self.mean_shift_once(z).unsqueeze(0)
        for it in range(1, self.ms_iter):
            trajects = torch.cat((trajects, self.mean_shift_once(trajects[it-1]).unsqueeze(0)))
        
        trajects = trajects.permute(2, 0, 1)  
        # trajects: batch_size x ms_iter x z_len
        return trajects

if __name__ == '__main__':
    
    z_len = 64
    batch_size = 216
    delta = 0.5
    eta = 1
    ms_iter = 3
    
    z = torch.randn(batch_size, z_len)
    print('z size', z.size())
    
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        z = z.cuda()
    
    mean_shift = Mean_Shift(delta, eta, ms_iter, use_gpu)
    
    trajects = mean_shift(z.t())
    print('trajects size', trajects.size())
    print('embed unit norm', trajects[0, 0, :].norm(p=2))