import torch
import torch.nn as nn
import numpy as np



class Listwise(nn.Module):
    """ Listwise LTR model """

    def __init__(self, input_size=501):
        """ Initialize model """
        super(Listwise, self).__init__()

        # assert metric in ['NDCG', 'ERR']

        self.nnet = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

        # self.metric = metric

    def forward(self, x):
        """ Forward pass """
        return self.nnet(x)


def delta_ndcg(y_hat, y):
    
    indices = y_hat.sort(descending=True).indices
    R_true = y.sort(descending=True).values
    R = y[indices]

    alpha = ( (2 ** R_true - 1) / torch.arange(2, len(y)+2).float().log2().to(DEVICE) ).sum()

    if alpha.item() == 0:
        return torch.ones_like(R)
    else:
        Nd = len(y_hat)

        Delta  = ((2**R-1).unsqueeze(1) / torch.arange(2, len(R)+2).float().log2().cuda() )
        ddcg = (Delta.sum(dim=1) + Delta.sum(dim=0) - Delta.diag() * Nd - Delta.diag().sum())/alpha
    
        return ddcg[indices] / alpha
    
    
def delta_err(y_hat, y):
    R = y[y_hat.sort(descending=True).indices]
    Nd = len(R)
    D = torch.zeros(Nd, Nd, Nd).to(DEVICE)
    
    phi = torch.arange(1,Nd+1).to(DEVICE)
    p = (2**R-1) / 2**4
    
    for i in range(Nd):
        for j in range(i+1, Nd):
            
            # swap indices
            p = p.clone()
            p[i], p[j] = p[j].item(), p[i].item()
                        
            prob_stopping = torch.cumprod(1-p, 0) / (1-p[0]) * p
            D[i,j,:] = D[j,i,:] = prob_stopping
    
    D = (D / phi).sum(dims=[1,2])
            
    return D


class LambdaRankFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, y_hat, y, sigma, metric):
        
        S = (y.unsqueeze(1) - y.t()).clamp(-1, 1)
        D = y_hat.unsqueeze(1) - y_hat.t()
        
        ctx.save_for_backward(S, D, y_hat, y)
        ctx.sigma = sigma
        ctx.metric = metric
        
        loss = .5 * (1-S) * sigma * D + torch.log2(1+ torch.exp(-sigma*D))
            
        return loss.sum()
    
    @staticmethod
    def backward(ctx, grad_output):
        S, D, y_hat, y = ctx.saved_tensors
        
        Lambda = ctx.sigma * (.5 * (1-S) - 1 / (1+torch.exp(-ctx.sigma*D)))
        
        if ctx.metric == 'NDCG':
            dirm = delta_ndcg(y_hat, y)
        elif ctx.metric == 'ERR':
            dirm = delta_err(y_hat, y)
        else:
            raise NotImplementedError()
        
        dy_hat = Lambda.sum(dim=1) * dirm.abs()
        
        return dy_hat, None, None, None


class LambdaRankLoss(torch.nn.Module):
    '''
    input: predictions and target
    output: scalar loss 
    '''
    
    def __init__(self, sigma, metric):
        assert metric in ['NDCG', 'ERR']
        
        super(LambdaRankLoss, self).__init__()
        self.sigma = sigma
        self.metric = metric
        
    
    def forward(self, y_hat, y):
        return LambdaRankFunction.apply(y_hat, y, self.sigma, self.metric)

    
    



if __name__ == "__main__":
    pass
