import torch
import copy
from scipy.special import gamma

def gamma_func(x):
    return torch.tensor(gamma(x))

def noise_estimation_loss(score_model,
                          sde,
                          levy,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e_B: torch.Tensor,
                          e_L: torch.Tensor,
                          b: torch.Tensor, keepdim = False):

    # aa = (1-b).cumprod(dim=0).index_select(0, t) #.view(-1, 1, 1, 1)   # a(t)^2

    # sigma1 = w_b * torch.pow(1.0 - aa, 0.5)
    # sigma2 = w_c * torch.pow(1.0 - torch.pow(aa, sde.alpha/2), 1/sde.alpha)

    aa = sde.diffusion_coeff(t)
    sigma1, sigma2 = sde.marginal_std(t)

    # Combined ( aa.sqrt() = a(t) )
    x_t = x0 * aa.sqrt().view(-1,1,1,1) + e_B * sigma1.view(-1,1,1,1) + e_L * sigma2.view(-1,1,1,1)
    
    # in case of 'sigma_2 = 0' : only brownian
    # score = - (x_t - a * x0) / torch.pow(sigma1, 2)

    # # in case of 'sigma_1 = 0' : only levy
    # score = - (gamma_func(3/sde.alpha)/gamma_func(1/sde.alpha)) * (x_t - aa.sqrt() * x0) / torch.pow(sigma2, 2)

    # in case of 'sigma_1 neq 0 && sigma_2 neq 0' : brownian + levy
    output = score_model(x_t, t.float())

    if sde.alpha == 2.0:
        sigma_score = -1 / 2 * (e_L)
    else:
        sigma_score = levy.score(e_L, sde.alpha)

    weighted_loss = (sigma2[:,None,None,None]*output - sigma_score)

    if keepdim:
        return (weighted_loss).square().sum(dim=(1, 2, 3))
    else:
        return (weighted_loss).square().sum(dim=(1, 2, 3)).mean(dim=0)

loss_registry = {
    'simple': noise_estimation_loss,
}
