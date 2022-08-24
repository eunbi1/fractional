import torch
import math

class VPSDE:
    def __init__(self, w_b, w_c, alpha, beta_min=0.1, beta_max=20, cosine_s=0.008, cosine_beta_max=999., schedule='linear', T=1.):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.cosine_s = cosine_s
        self.cosine_beta_max = cosine_beta_max
        self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
        self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
        self.schedule = schedule

        self.alpha = alpha
        self.b = w_b
        self.c = w_c
        
        if schedule == 'cosine':
            self.T = 0.9946
        else:
            self.T = T

    def beta(self, t):
        return (self.beta_1 - self.beta_0) * t + self.beta_0

    def marginal_log_mean_coeff(self, t):
        # linear 외에는 아직 사용x 
        if self.schedule == 'linear':
            log_alpha_t = - 1/4 * (t ** 2) * (self.beta_1 - self.beta_0) - 1/2 * t * self.beta_0
            return log_alpha_t
            # return (-1 / (2*alpha)* t ** 2 * (self.beta_1 - self.beta_0) - 1/alpha * t * self.beta_0) * 1/2 
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0
            t_max_vec = self.cosine_t_max * torch.ones_like(t).to(t)
            log_alpha_t_max =  log_alpha_fn(t_max_vec) - self.cosine_log_alpha_0
            return log_alpha_t
            # return torch.where(t >= self.cosine_t_max, log_alpha_t_max - 0.5 *  (t - t_max_vec) * self.cosine_beta_max, log_alpha_t)
            # return torch.where(t >= self.cosine_t_max, log_alpha_t_max + 0.5 *  (t - t_max_vec) * self.cosine_beta_max, log_alpha_t)
        else:
            raise ValueError("Unsupported ")
    
    def diffusion_coeff(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        # actually std->scale 
        # return torch.pow(1. - torch.exp(alpha*self.marginal_log_mean_coeff(t, alpha)), 1/alpha)
        # return torch.pow(2 * (1. - torch.exp(alpha*self.marginal_log_mean_coeff(t, alpha))), 1/alpha)
        sigma1 = self.b * torch.pow(1. - torch.exp(2*self.marginal_log_mean_coeff(t)), 1/2)
        sigma2 = self.c * torch.pow(1. - torch.exp(self.alpha*self.marginal_log_mean_coeff(t)), 1/self.alpha)        

        return sigma1, sigma2


    def marginal_lambda(self, t):
        # lambda(t) = \log a(t)/sigma(t). marginal_lambda does not change even if we change alpha, 
        #which mean \lambda_{\alpha_1}(t) = \lambda_{\alpha_2}(t) 
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_sigma1 = torch.log(self.b * torch.pow(1. - torch.exp(2*log_mean_coeff), 1/2))
        log_sigma2 = torch.log(self.c * torch.pow(1. - torch.exp(self.alpha*log_mean_coeff), 1/self.alpha))
        # log_std = 1/alpha * torch.log(2 * (1. - torch.exp(alpha * log_mean_coeff)))

        lambda1 = log_mean_coeff - log_sigma1
        lambda2 = log_mean_coeff - log_sigma2
        return lambda1, lambda2

    def inverse_lambda(self, lamb):
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2))
            t_max_vec = self.cosine_t_max * torch.ones_like(lamb).to(lamb)
            log_alpha_t_max =  log_alpha_fn(t_max_vec) - self.cosine_log_alpha_0
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            # t = torch.where(log_alpha <= log_alpha_t_max, t_max_vec + (log_alpha_t_max - log_alpha) * 2. / self.cosine_beta_max, t_fn(log_alpha))
            # t = torch.where(log_alpha <= log_alpha_t_max, (log_alpha_t_max - log_alpha) * 2. * self.cosine_beta_max, t_fn(log_alpha))
            return t