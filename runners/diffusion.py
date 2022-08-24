import os
import logging
import time
import glob
from tkinter import E

import numpy as np
import tqdm
import torch
import math 
import torch.utils.data as data 

from models.diffusion import Model, get_score_fn
from models.improved_ddpm.unet import UNetModel as ImprovedDDPM_Model
from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from models.guided_diffusion.unet import EncoderUNetModel as GuidedDiffusion_Classifier
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from functions.sde import VPSDE
from functions.levy_stable_pytorch import LevyStable
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from evaluate.fid_score import calculate_fid_given_paths

from scipy.stats import levy_stable

import torchvision.utils as tvu


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.alpha = self.config.diffusion.alpha
        self.w_b = self.config.model.b
        self.w_c = self.config.model.c
        self.sde = VPSDE(self.w_b, self.w_c, self.alpha, schedule=self.config.sampling.schedule)
        self.levy = LevyStable()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        torch.multiprocessing.set_start_method('spawn')
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            # num_workers=config.data.num_workers,
            num_workers=0,
            generator=torch.Generator(device=self.device),
        )

        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        # model = Model(config)
        # if args.resume_training_cpkt:
        #     ckpt = get_ckpt_path(f"ema_cifar10")
        #     print("Loading checkpoint {}".format(ckpt))
        #     model.load_state_dict(torch.load(ckpt, map_location=self.device))
        #     model.to(self.device)
        #     model = torch.nn.DataParallel(model)
        # else:
        #     model = model.to(self.device)
        #     model = torch.nn.DataParallel(model)

        optimizer= get_optimizer(self.config, model.parameters())
        optimizer.param_groups[0]["capturable"] = True

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)

                # Brownian motion
                e_B = torch.randn_like(x)

                # Levy motion
                e_L = levy_stable.rvs(alpha=self.alpha, beta=0, loc=0, scale=1, size=x.shape)
                e_L = torch.Tensor(e_L).to(self.device)

                b = self.betas

                # antithetic sampling
                # t = torch.randint(
                #     low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                # ).to(self.device)
                # t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                t = torch.rand(size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                # # alpha scheduling
                # scheduled_alpha = torch.tensor([2.0 if _t < 50 else alpha for _t in t]).to(self.device)
                score_model = get_score_fn(model, self.levy, self.sde)
                loss = loss_registry[config.model.type](score_model, self.sde, self.levy, x, t, e_B, e_L, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()


    def sample(self):
        if self.config.model.model_type == 'improved_ddpm':
            model = ImprovedDDPM_Model(
                in_channels=self.config.model.in_channels,
                model_channels=self.config.model.model_channels,
                out_channels=self.config.model.out_channels,
                num_res_blocks=self.config.model.num_res_blocks,
                attention_resolutions=self.config.model.attention_resolutions,
                dropout=self.config.model.dropout,
                channel_mult=self.config.model.channel_mult,
                conv_resample=self.config.model.conv_resample,
                dims=self.config.model.dims,
                use_checkpoint=self.config.model.use_checkpoint,
                num_heads=self.config.model.num_heads,
                num_heads_upsample=self.config.model.num_heads_upsample,
                use_scale_shift_norm=self.config.model.use_scale_shift_norm
            )
        elif self.config.model.model_type == "guided_diffusion":
            model = GuidedDiffusion_Model(
                image_size=self.config.model.image_size,
                in_channels=self.config.model.in_channels,
                model_channels=self.config.model.model_channels,
                out_channels=self.config.model.out_channels,
                num_res_blocks=self.config.model.num_res_blocks,
                attention_resolutions=self.config.model.attention_resolutions,
                dropout=self.config.model.dropout,
                channel_mult=self.config.model.channel_mult,
                conv_resample=self.config.model.conv_resample,
                dims=self.config.model.dims,
                num_classes=self.config.model.num_classes,
                use_checkpoint=self.config.model.use_checkpoint,
                use_fp16=self.config.model.use_fp16,
                num_heads=self.config.model.num_heads,
                num_head_channels=self.config.model.num_head_channels,
                num_heads_upsample=self.config.model.num_heads_upsample,
                use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                resblock_updown=self.config.model.resblock_updown,
                use_new_attention_order=self.config.model.use_new_attention_order,
            )
        else:
            model = Model(self.config)

        if "ckpt_dir" in self.config.model.__dict__.keys():
            ckpt_dir = self.config.model.ckpt_dir
            states = torch.load(
                ckpt_dir,
                map_location=self.config.device,
            )
            model = model.to(self.device)
            if self.config.model.model_type == 'improved_ddpm' or self.config.model.model_type == 'guided_diffusion':
                model.load_state_dict(states, strict=True)
                if self.config.model.use_fp16:
                    model.convert_to_fp16()
                model = torch.nn.DataParallel(model)
            else:
                model = torch.nn.DataParallel(model)
                model.load_state_dict(states[0], strict=True)

            if self.config.model.ema: # for celeba 64x64 in DDIM
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)

            else:
                ema_helper = None

            if self.config.sampling.cond_class:
                classifier = GuidedDiffusion_Classifier(
                    image_size=self.config.classifier.image_size,
                    in_channels=self.config.classifier.in_channels,
                    model_channels=self.config.classifier.model_channels,
                    out_channels=self.config.classifier.out_channels,
                    num_res_blocks=self.config.classifier.num_res_blocks,
                    attention_resolutions=self.config.classifier.attention_resolutions,
                    channel_mult=self.config.classifier.channel_mult,
                    use_fp16=self.config.classifier.use_fp16,
                    num_head_channels=self.config.classifier.num_head_channels,
                    use_scale_shift_norm=self.config.classifier.use_scale_shift_norm,
                    resblock_updown=self.config.classifier.resblock_updown,
                    pool=self.config.classifier.pool
                )
                ckpt_dir = self.config.classifier.ckpt_dir
                states = torch.load(
                    ckpt_dir,
                    map_location=self.config.device,
                )
                classifier = classifier.to(self.device)
                classifier.load_state_dict(states, strict=True)
                if self.config.classifier.use_fp16:
                    classifier.convert_to_fp16()
                classifier = torch.nn.DataParallel(classifier)
            else:
                classifier = None
        else:
            classifier = None
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()
        score_model = get_score_fn(model, self.levy, self.sde)

        if self.args.fid:
            self.sample_fid(score_model, classifier=classifier)
            # if not os.path.exists(os.path.join(self.args.exp, "fid.npy")):
            #     logging.info("Begin to compute FID...")
            #     fid = calculate_fid_given_paths((self.config.sampling.fid_stats_dir, self.args.image_folder), batch_size=self.config.sampling.fid_batch_size, device='cuda:0', dims=2048, num_workers=8)
            #     logging.info("FID: {}".format(fid))
            #     np.save(os.path.join(self.args.exp, "fid"), fid)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model, classifier=None):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000 # total num of datasamples (cifar10 has 50000 training dataset)
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        # fixed random seed to comparison
        # temp = [10487, 29703,  4537, 36273,  7303, 23166, 39392,  6310, 21725, 19929,
        #  1785, 17223, 46623, 26445,  2998, 48249, 12721, 33664,  6364,  8865,
        # 12088, 19320, 39021, 40723, 16111]
        temp = [i for i in range(25)]

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x_shape = (n, config.data.channels, config.data.image_size, config.data.image_size)

                x_B = torch.randn(x_shape, device=self.device)
                x_L = levy_stable.rvs(alpha=self.alpha, beta=0, loc=0, scale=1, size=x_shape)
                x_L = torch.Tensor(x_L).to(self.device)
                x = self.w_b * x_B + self.w_c * x_L

                x = self.sample_image(x, model, self.alpha, classifier=classifier)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    if img_id in temp:
                        temp_path = self.args.image_folder + "/../tests"
                        if not os.path.exists(temp_path):
                            os.makedirs(temp_path)
                        tvu.save_image(
                            x[i], os.path.join(temp_path, f"test_{img_id}.png")
                        )
                    img_id += 1
                
                if img_id > 25:
                    break
    

    def sample_sequence(self, model, classifier=None):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False, classifier=classifier)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []
        print(f"x : {x.shape}")

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, alpha, last=True, classifier=None):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        elif self.args.sample_type in ['ddim', 'ddim_second', 'ddim_third', 'ddim_fast', 'dpm_solver_12', 'dpm_solver_23', 'ddim_levy', 'ddim_score', 'pc_score']:
            from functions.sampler import dpm_solver_sample
            def model_fn(x, t, y=None):
                if self.config.sampling.cond_class:
                    out = model(x, t, y)
                else:
                    out = model(x, t)
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out

            x = dpm_solver_sample(x, model_fn, self.args.timesteps, self.w_b, self.w_c, self.alpha, self.config.sampling.h, self.sde, self.levy, skip_type=self.args.skip_type, method=self.args.sample_type, eps=self.args.start_time, total_N=self.config.sampling.total_N, schedule=self.config.sampling.schedule, classifier=classifier, cond_class=self.config.sampling.cond_class, num_classes=self.config.data.num_classes, classifier_scale=self.config.sampling.classifier_scale, fixed_class=self.args.fixed_class)
            return x.cpu()
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
