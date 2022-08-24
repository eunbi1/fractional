import torch
import torch.nn.functional as F
import math
import scipy
from scipy.special import gamma


def cond_fn(x, t_discrete, y, classifier, classifier_scale=1.):
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t_discrete)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale


def dpm_solver_sample(x, model_fn, steps, w_b, w_c, alpha, h, sde, levy,
        skip_type='logSNR', method='ddim_second', eps=1e-4, T=1., total_N=1000,
        schedule='linear', classifier=None, cond_class=False,
        num_classes=1000, classifier_scale=1., fixed_class=None,
        Predictor=True, Corrector=False, LM_steps=200):
    
    device = x.device
    if cond_class:
        if fixed_class is None:
            classes = torch.randint(low=0, high=num_classes, size=(x.shape[0],)).to(device)
        else:
            classes = torch.randint(low=fixed_class, high=fixed_class + 1, size=(x.shape[0],)).to(device)
    else:
        classes = None

    def model(x, t_discrete):
        if cond_class:
            noise_uncond = model_fn(x, t_discrete, classes)
            cond_grad = cond_fn(x, t_discrete, classes, classifier, classifier_scale=classifier_scale)
            sigma_t1, sigma_t2 = sde.marginal_std(get_continuous_time(t_discrete))[:,None,None,None]
            return noise_uncond - sigma_t * cond_grad
        else:
            return model_fn(x, t_discrete)

    def get_discrete_time(t):
        # Type-1
        return 1000. * torch.max(t - 1. / total_N, torch.zeros_like(t).to(t))
        # Type-2
        # max_N = (total_N - 1) / total_N * 1000.
        # return max_N * t

    def get_continuous_time(t):
        max_N = (total_N - 1) / total_N * 1000.
        return t / max_N

    def get_time_steps_by_logSNR(T=sde.T, t0=eps, N=steps):
        logSNR_steps = torch.linspace(sde.marginal_lambda(torch.tensor(T).to(device)), sde.marginal_lambda(torch.tensor(t0).to(device)), N + 1).to(device)
        return get_time_by_logSNR(logSNR_steps)

    def get_time_steps_by_logSNR_quadratic(T=sde.T, t0=eps, N=steps):
        lambda_T = sde.marginal_lambda(torch.tensor(T).to(device))
        lambda_0 = sde.marginal_lambda(torch.tensor(t0).to(device))
        logSNR_steps = torch.linspace(torch.zeros_like(lambda_T).to(device), torch.sqrt(lambda_0 - lambda_T), N + 1).to(device)
        return get_time_by_logSNR(torch.square(logSNR_steps) + lambda_T)

    def get_time_steps_by_quadratic(T=sde.T, t0=eps, N=steps):
        t = torch.linspace(t0, T, 10000000).to(device)
        quadratic_t = torch.sqrt(t)
        quadratic_steps = torch.linspace(quadratic_t[0], quadratic_t[-1], N + 1).to(device)
        return torch.flip(torch.cat([t[torch.searchsorted(quadratic_t, quadratic_steps)[:-1]], T * torch.ones((1,)).to(device)], dim=0), dims=[0])

    def get_time_steps_for_fast_sampling(T=sde.T, t0=eps, N=steps):
        K = N // 3 + 1
        if N % 3 == 0:
            first_steps = K - 2
        else:
            first_steps = K - 1
        return first_steps, get_time_steps_by_logSNR(N=K)

    def get_time_by_logSNR(lamb):
        return sde.inverse_lambda(lamb)
    
    # def gamma_func(x):
    #     return torch.tensor(gamma(x))

    def gamma_func(x):
        return torch.tensor(gamma(x)).to(device)

    def ddim_update(x, s, t, return_noise=False, denoise=False):
        """
        input: x_s, s, t
        output: x_t
        """
        noise_s = model(x, get_discrete_time(s))
        log_alpha_s, log_alpha_t = sde.marginal_log_mean_coeff(s), sde.marginal_log_mean_coeff(t)
        lambda_s, lambda_t = sde.marginal_lambda(s), sde.marginal_lambda(t)
        sigma_s, sigma_t = sde.marginal_std(s), sde.marginal_std(t)
        if denoise:
            x_coeff = torch.exp(-log_alpha_s)
            noise_coeff = sigma_s * x_coeff
        else:
            h = lambda_t - lambda_s
            x_coeff = torch.exp(log_alpha_t - log_alpha_s)
            noise_coeff = sigma_t * (torch.expm1(h))
        x_t =  x_coeff[:,None,None,None] * x - noise_coeff[:,None,None,None] * noise_s
        if return_noise:
            return x_t, noise_s
        else:
            return x_t

    def ddim_second_update(x, s, t, r1=0.5, noise_s=None, return_noise=False):
        lambda_s, lambda_t = sde.marginal_lambda(s), sde.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = get_time_by_logSNR(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = sde.marginal_log_mean_coeff(s), sde.marginal_log_mean_coeff(s1), sde.marginal_log_mean_coeff(t)
        sigma_s1, sigma_t = sde.marginal_std(s1), sde.marginal_std(t)
        
        if noise_s is None:
            noise_s = model(x, get_discrete_time(s))
        x_s1 = (
            torch.exp(log_alpha_s1 - log_alpha_s)[:,None,None,None] * x
            - (sigma_s1 * torch.expm1(r1 * h))[:,None,None,None] * noise_s
        )
        noise_s1 = model(x_s1, get_discrete_time(s1))
        x_t = (
            torch.exp(log_alpha_t - log_alpha_s)[:,None,None,None] * x
            - (sigma_t * torch.expm1(h))[:,None,None,None] * noise_s
            - (0.5 / r1) * (sigma_t * torch.expm1(h))[:,None,None,None] * (noise_s1 - noise_s)
        )
        if return_noise:
            return x_t, noise_s, noise_s1
        else:
            return x_t

    def ddim_third_update(x, s, t, r1=1./3., r2=2./3., noise_s=None, noise_s1=None, noise_s2=None):
        lambda_s, lambda_t = sde.marginal_lambda(s), sde.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = get_time_by_logSNR(lambda_s1)
        s2 = get_time_by_logSNR(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = sde.marginal_log_mean_coeff(s), sde.marginal_log_mean_coeff(s1), sde.marginal_log_mean_coeff(s2), sde.marginal_log_mean_coeff(t)
        sigma_s1, sigma_s2, sigma_t = sde.marginal_std(s1), sde.marginal_std(s2), sde.marginal_std(t)

        phi_11 = torch.expm1(r1 * h)
        phi_12 = torch.expm1(r2 * h)
        phi_1 = torch.expm1(h)
        phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
        phi_2 = torch.expm1(h) / h - 1.

        if noise_s is None:
            noise_s = model(x, get_discrete_time(s))
        if noise_s1 is None:
            x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s)[:,None,None,None] * x
                - (sigma_s1 * phi_11)[:,None,None,None] * noise_s
            )
            noise_s1 = model(x_s1, get_discrete_time(s1))
        if noise_s2 is None:
            x_s2 = (
                torch.exp(log_alpha_s2 - log_alpha_s)[:,None,None,None] * x
                - (sigma_s2 * phi_12)[:,None,None,None] * noise_s
                - r2 / r1 * (sigma_s2 * phi_22)[:,None,None,None] * (noise_s1 - noise_s)
            )
            noise_s2 = model(x_s2, get_discrete_time(s2))
        x_t = (
            torch.exp(log_alpha_t - log_alpha_s)[:,None,None,None] * x
            - (sigma_t * phi_1)[:,None,None,None] * noise_s
            - (1. / r2) * (sigma_t * phi_2)[:,None,None,None] * (noise_s2 - noise_s)
        )
        return x_t

    def dpm_solver_adaptive_12(x, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9):
        s = sde.T * torch.ones((x.shape[0],)).to(x)
        lambda_s = sde.marginal_lambda(s)
        lambda_eps = sde.marginal_lambda(eps * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        while torch.abs((s - eps)).mean() > 1e-5:
            t = get_time_by_logSNR(lambda_s + h)
            print(s.mean(), t.mean())
            x_first, noise_s = ddim_update(x, s, t, return_noise=True)
            x_second = ddim_second_update(x, s, t, noise_s=noise_s)
            delta = torch.max(torch.ones_like(x_first).to(x) * atol, rtol * torch.max(torch.abs(x_first), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_first - x_second) / delta).max()
            if torch.all(E <= 1.):
                x = x_second
                s = t
                x_prev = x_first
                lambda_s = sde.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -0.5).float(), lambda_eps - lambda_s)[0]
            nfe += 2
        print('nfe', nfe)
        return x

    def dpm_solver_adaptive_23(x, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9):
        s = sde.T * torch.ones((x.shape[0],)).to(x)
        lambda_s = sde.marginal_lambda(s)
        lambda_eps = sde.marginal_lambda(eps * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        r1, r2 = 1. / 3., 2. / 3.
        while torch.abs((s - eps)).mean() > 1e-5:
            t = get_time_by_logSNR(lambda_s + h)
            print(s.mean(), t.mean())
            x_second, noise_s, noise_s1 = ddim_second_update(x, s, t, r1=r1, return_noise=True)
            x_third = ddim_third_update(x, s, t, r1=r1, r2=r2, noise_s=noise_s, noise_s1=noise_s1)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_second), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_third - x_second) / delta).max()
            if torch.all(E <= 1.):
                x = x_third
                s = t
                x_prev = x_second
                lambda_s = sde.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / 3.).float(), lambda_eps - lambda_s)[0]
            nfe += 3
        print('nfe', nfe)
        return x

    def ddim_levy_update(x, s, t, return_noise=False, denoise=False):
        """
        input: x_s, s, t
        output: x_t
        """
        # scheduled_alpha_s = torch.tensor([2.0 if _s < 0.1 else 1.9 for _s in s]).to(device)
        # scheduled_alpha_t = torch.tensor([2.0 if _t < 0.1 else 1.9 for _t in t]).to(device)
        # alpha_s = scheduled_alpha_s.cpu().clone()
        alpha = 2.0
        noise_s = model(x, get_discrete_time(s))
        # log_alpha_s, log_alpha_t = sde.marginal_log_mean_coeff(s, scheduled_alpha_s), sde.marginal_log_mean_coeff(t, scheduled_alpha_t)
        # lambda_s, lambda_t = sde.marginal_lambda(s, scheduled_alpha_s), sde.marginal_lambda(t, scheduled_alpha_t)
        # sigma_s, sigma_t = sde.marginal_std(s, scheduled_alpha_s), sde.marginal_std(t, scheduled_alpha_t)
        log_alpha_s, log_alpha_t = sde.marginal_log_mean_coeff(s), sde.marginal_log_mean_coeff(t)
        lambda_s, lambda_t = sde.marginal_lambda(s), sde.marginal_lambda(t)
        sigma_s, sigma_t = sde.marginal_std(s), sde.marginal_std(t)

        if denoise:
            x_coeff = torch.exp(-log_alpha_s)
            noise_coeff = sigma_s * x_coeff
            raise Exception("denoise case")
        else:
            h = lambda_t - lambda_s
            x_coeff = torch.exp(log_alpha_t - log_alpha_s)
            # F_coeff = sigma_t * (torch.expm1(h)) * alpha * torch.pow(sigma_s, alpha-1)
            # F_s = F(x, get_discrete_time(s), h, sigma_s)
            # noise_coeff = sigma_t * (torch.expm1(h)) * torch.pow(sigma_s, scheduled_alpha_s-2) * scheduled_alpha_s * \
            #             gamma_func(alpha_s-1) * gamma_func(3/alpha_s) / torch.pow(gamma_func(alpha_s/2),2) / gamma_func(1/alpha_s)
            noise_coeff = sigma_t / torch.pow(h, alpha-2) * (torch.expm1(h)) * torch.pow(sigma_s, alpha-2) * alpha * \
                        gamma_func(alpha-1) * gamma_func(3/alpha) / torch.pow(gamma_func(alpha/2),2) / gamma_func(1/alpha)

        # x_t = x_coeff[:,None,None,None] * x + F_coeff[:,None,None,None] * F_s
        x_t =  x_coeff[:,None,None,None] * x - noise_coeff[:,None,None,None] * noise_s

        if return_noise:
            return x_t, noise_s
        else:
            return x_t

    def ddim_score_update(x, s, t, h, return_noise=False, denoise=False):
        """
        input: x_s, s, t
        output: x_t
        """
        # scheduled_alpha_s = torch.tensor([2.0 if _s < 0.1 else 1.9 for _s in s]).to(device)
        # scheduled_alpha_t = torch.tensor([2.0 if _t < 0.1 else 1.9 for _t in t]).to(device)
        # alpha_s = scheduled_alpha_s.cpu().clone()

        # log_alpha_s, log_alpha_t = sde.marginal_log_mean_coeff(s, scheduled_alpha_s), sde.marginal_log_mean_coeff(t, scheduled_alpha_t)
        # lambda_s, lambda_t = sde.marginal_lambda(s, scheduled_alpha_s), sde.marginal_lambda(t, scheduled_alpha_t)
        # sigma_s, sigma_t = sde.marginal_std(s, scheduled_alpha_s), sde.marginal_std(t, scheduled_alpha_t)
        log_alpha_s, log_alpha_t = sde.marginal_log_mean_coeff(s), sde.marginal_log_mean_coeff(t)
        lambda1_s, lambda2_s = sde.marginal_lambda(s)
        lambda1_t, lambda2_t = sde.marginal_lambda(t)
        sigma1_s, sigma2_s = sde.marginal_std(s)
        sigma1_t, sigma2_t = sde.marginal_std(t)

        score_s = model(x, s)
        # print(f"score : {score_s}")

        if denoise:
            x_coeff = torch.exp(-log_alpha_s)
            noise_coeff = sigma_s * x_coeff
            raise Exception("denoise case")
        else:
            h = torch.tensor(h).to(device)
            h1 = lambda1_t - lambda1_s
            h2 = lambda2_t - lambda2_s
            x_coeff = torch.exp(log_alpha_t - log_alpha_s)
            # F_coeff = sigma_t * (torch.expm1(h)) * alpha * torch.pow(sigma_s, alpha-1)
            # F_s = F(x, get_discrete_time(s), h, sigma_s)
            # noise_coeff = sigma_t * (torch.expm1(h)) * torch.pow(sigma_s, scheduled_alpha_s-2) * scheduled_alpha_s * \
            #             gamma_func(alpha_s-1) * gamma_func(3/alpha_s) / torch.pow(gamma_func(alpha_s/2),2) / gamma_func(1/alpha_s)

            # noise_coeff = sigma_t / torch.pow(h, alpha-2) * (torch.expm1(h)) * torch.pow(sigma_s, alpha-2) * alpha * \
            #             gamma_func(alpha-1) * gamma_func(3/alpha) / torch.pow(gamma_func(alpha/2),2) / gamma_func(1/alpha)
            
            coeff1 = sigma1_t * sigma1_s * torch.expm1(h1)
            coeff2 = sigma2_t * torch.pow(sigma2_s, alpha-1) * alpha * torch.expm1(h2) \
                     * gamma_func(alpha-1) / torch.pow(gamma_func(alpha/2),2) / torch.pow(h, alpha-2)
            # score_coeff = coeff1 + coeff2
            score_coeff = coeff2

        # x_t = x_coeff[:,None,None,None] * x + F_coeff[:,None,None,None] * F_s
        # x_t =  x_coeff[:,None,None,None] * x - noise_coeff[:,None,None,None] * noise_s
        x_t =  x_coeff[:,None,None,None] * x + score_coeff[:,None,None,None] * score_s

        if return_noise:
            return x_t, score_s
        else:
            return x_t
    
    def sde_score_update(x, s, t, h):
        step_size = torch.abs(t-s)
        score_s = model(x, s)
        h = torch.tensor(h).to(device)

        x_coeff = 1 + sde.beta(s) / sde.alpha * step_size
        score_coeff = 2 * sde.beta(s) * step_size * gamma_func(sde.alpha-1) / torch.pow(gamma_func(sde.alpha/2), 2) \
                      / torch.pow(h, sde.alpha-2)
        noise_coeff = torch.pow(sde.beta(s) * step_size, 1 / sde.alpha)

        e_L = levy.sample(sde.alpha, 0, x.shape).to(device)

        x_t = x_coeff[:, None, None, None] * x + score_coeff[:, None, None, None] * score_s + noise_coeff[:, None, None, None] * e_L

        return x_t


    if method == 'ddim_fast':
        first_steps, timesteps = get_time_steps_for_fast_sampling(N=steps)
        
        with torch.no_grad():
            for i in range(first_steps):
                vec_s, vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i], torch.ones((x.shape[0],)).to(device) * timesteps[i + 1]
                x = ddim_third_update(x, vec_s, vec_t)
            if steps % 3 == 0:
                vec_s, vec_t = torch.ones(x.shape[0]).to(device) * timesteps[first_steps], torch.ones(x.shape[0]).to(device) * timesteps[first_steps + 1]
                x = ddim_second_update(x, vec_s, vec_t)
                vec_s, vec_t = torch.ones(x.shape[0]).to(device) * timesteps[first_steps + 1], torch.ones(x.shape[0]).to(device) * timesteps[first_steps + 2]
                x = ddim_update(x, vec_s, vec_t)
            elif steps % 3 == 1:
                vec_s, vec_t = torch.ones(x.shape[0]).to(device) * timesteps[first_steps], torch.ones(x.shape[0]).to(device) * timesteps[first_steps + 1]
                x = ddim_update(x, vec_s, vec_t)
            else:
                vec_s, vec_t = torch.ones(x.shape[0]).to(device) * timesteps[first_steps], torch.ones(x.shape[0]).to(device) * timesteps[first_steps + 1]
                x = ddim_second_update(x, vec_s, vec_t)
    elif method == 'dpm_solver_12':
        with torch.no_grad():
            x = dpm_solver_adaptive_12(x)
    elif method == 'dpm_solver_23':
        with torch.no_grad():
            x = dpm_solver_adaptive_23(x)
    else:
        # timesteps length = NFE + 1
        if skip_type == 'logSNR':
            timesteps = get_time_steps_by_logSNR(N=steps)
        elif skip_type == 'time' or skip_type == 'uniform':
            timesteps = torch.linspace(sde.T, eps, steps + 1).to(device)
        elif skip_type == 'quad' or skip_type == 'quadratic':
            timesteps = get_time_steps_by_quadratic(N=steps)
        elif skip_type == 'logSNR_quad':
            timesteps = get_time_steps_by_logSNR_quadratic(N=steps)
        
        with torch.no_grad():
            for i in range(steps):
                print(f"--------- step : {i}")
                # vector length = sampling.batch_size
                vec_s, vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i], torch.ones((x.shape[0],)).to(device) * timesteps[i + 1]
                if method == 'ddim':
                    x = ddim_update(x, vec_s, vec_t)
                elif method == 'ddim_levy':
                    x = ddim_levy_update(x, vec_s, vec_t)
                elif method == 'ddim_score':
                    x = ddim_score_update(x, vec_s, vec_t, h)
                elif method == 'ddim_second':
                    x = ddim_second_update(x, vec_s, vec_t)
                elif method == 'ddim_third':
                    x = ddim_third_update(x, vec_s, vec_t)
                elif method == 'pc_score':
                    step_size = timesteps[0] - timesteps[1]
                    if Corrector:
                        for j in range(LM_score):
                            grad = model(x, vec_t)
                            e_L = levy.sample(sde.alpha, 0, x.shape).to(device)

                            x = x + step_size * gamma_func(sde.alpha-1) / (gamma_func(sde.alpha/2)**2) \
                                * grad + torch.pow(step_size, 1/sde.alpha) * e_L
                    if Predictor:
                        x = sde_score_update(x, vec_s, vec_t, h)
    
    return x
