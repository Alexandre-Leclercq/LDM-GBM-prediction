"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import lightning as L
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from typing import Literal, List
from torchvision.utils import make_grid
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig

from utils.autoloading import load_model
from models.ldm_stable.util import exists, default, count_params, instantiate_from_config
from models.ldm_stable.modules.ema import LitEma
from models.ldm_stable.modules.distributions.distributions import DiagonalGaussianDistribution
from models.ldm_stable.models.autoencoder import IdentityFirstStage, AutoencoderKL
from models.ldm_stable.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from models.ldm_stable.models.diffusion.ddim import DDIMSampler
from utils.metrics import ssim


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(L.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 make_it_fit=False,
                 ucg_training=None,
                 reset_ema=False,
                 reset_num_ema_updates=False,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        self.make_it_fit = make_it_fit
        if reset_ema: assert exists(ckpt_path)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
            if reset_ema:
                assert self.use_ema
                print(f"Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint.")
                self.model_ema = LitEma(self.model)
        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema
            self.model_ema.reset_num_updates()

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        self.ucg_training = ucg_training or dict()
        if self.ucg_training:
            self.ucg_prng = np.random.RandomState()

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if self.make_it_fit:
            n_params = len([name for name, _ in
                            itertools.chain(self.named_parameters(),
                                            self.named_buffers())])
            for name, param in tqdm(
                    itertools.chain(self.named_parameters(),
                                    self.named_buffers()),
                    desc="Fitting old weights to new weights",
                    total=n_params
            ):
                if not name in sd:
                    continue
                old_shape = sd[name].shape
                new_shape = param.shape
                assert len(old_shape) == len(new_shape)
                if len(new_shape) > 2:
                    # we only modify first two axes
                    assert new_shape[2:] == old_shape[2:]
                # assumes first axis corresponds to output dim
                if not new_shape == old_shape:
                    new_param = param.clone()
                    old_param = sd[name]
                    if len(new_shape) == 1:
                        for i in range(new_param.shape[0]):
                            new_param[i] = old_param[i % old_shape[0]]
                    elif len(new_shape) >= 2:
                        for i in range(new_param.shape[0]):
                            for j in range(new_param.shape[1]):
                                new_param[i, j] = old_param[i % old_shape[0], j % old_shape[1]]

                        n_used_old = torch.ones(old_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_old[j % old_shape[1]] += 1
                        n_used_new = torch.zeros(new_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_new[j] = n_used_old[j % old_shape[1]]

                        n_used_new = n_used_new[None, :]
                        while len(n_used_new.shape) < len(new_shape):
                            n_used_new = n_used_new.unsqueeze(-1)
                        new_param /= n_used_new

                    sd[name] = new_param

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        """
        forward diffusion process.
        diffused x_start at the t-steps in the diffusion sequence.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_v(self, x, noise, t):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[None, :]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True, sync_dist=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""

    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 source_keys: List[str],
                 target_key: str,
                 noised_source: bool = False,
                 source_noised_function: Literal["vanilla_noise", "skip_noise", "reverse_skip_noise"] = None,
                 source_noised_gamma: float = 0,
                 ddim_steps=None,
                 ddim_unconditional_scale=1.0,
                 condition_key=None,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 temperature=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        assert ((concat_mode and noised_source) or
                (concat_mode and not noised_source) or
                (not concat_mode and not noised_source))
        # assert that all input modalities have corresponding encoder in first_stage_config
        assert all(key in first_stage_config for key in source_keys)
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        reset_ema = kwargs.pop("reset_ema", False)
        reset_num_ema_updates = kwargs.pop("reset_num_ema_updates", False)
        ignore_keys = kwargs.pop("ignore_keys", [])
        # if the in_channel is different from base channels, it means unet in and out channel have already been processed.
        if concat_mode and kwargs['channels'] == kwargs['unet_config']['params']['in_channels']:
            kwargs['unet_config']['params']['in_channels'] *= len(source_keys) + 1
            if noised_source:
                kwargs['unet_config']['params']['out_channels'] *= len(source_keys) + 1
                # number of modalities to denoise
                kwargs['unet_config']['params']['number_modalities'] = len(source_keys) + 1
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key

        self.first_stage_model = nn.ModuleDict({})
        for key in first_stage_config:
            self.instantiate_first_stage(first_stage_config, key)
        self.instantiate_cond_stage(cond_stage_config)

        self.source_keys = source_keys
        self.target_key = target_key
        self.noised_source = noised_source  # activate or deactivate the noised add to src image.

        # the type of noised_function used in generation to specify the amount of noise to add to the source domain.
        self.source_noised_function = source_noised_function
        # parameter used to indicate the amount of noise we don't have in comparison to the noise add to the sampling.
        # over the generation processed.
        self.source_noised_gamma = source_noised_gamma
        self.condition_key = condition_key

        self.temperature = temperature

        # Metrics result buffer
        self.register_buffer("psnr_results", torch.empty([0]), persistent=False)
        self.register_buffer("local_psnr_results", torch.empty([0]), persistent=False)
        self.register_buffer("ssim_results", torch.empty([0]), persistent=False)
        self.register_buffer("local_ssim_results", torch.empty([0]), persistent=False)
        self.register_buffer("lpips_results", torch.empty([0]), persistent=False)
        self.register_buffer("global_mse_results", torch.empty([0]), persistent=False)
        self.register_buffer("local_mse_results", torch.empty([0]), persistent=False)

        try:
            self.num_downs = self.model.encoder.num_resolutions - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
            if reset_ema:
                assert self.use_ema
                print(
                    f"Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint.")
                self.model_ema = LitEma(self.model)
        if reset_num_ema_updates:
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema
            self.model_ema.reset_num_updates()
        self.ddim_steps = ddim_steps
        self.ddim_unconditional_scale = ddim_unconditional_scale

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x_src = super().get_input(batch, self.source_keys)
            x_tgt = super().get_input(batch, self.target_key)
            x_src = x_src.to(self.device)
            x_tgt = x_tgt.to(self.device)

            encoder_posterior_src, _, _ = self.first_stage_model.encode(x_src)
            z_src = self.get_first_stage_encoding(encoder_posterior_src).detach()

            encoder_posterior_tgt, _, _ = self.first_stage_model.encode(x_tgt)
            z_tgt = self.get_first_stage_encoding(encoder_posterior_tgt).detach()

            z = torch.cat([z_src, z_tgt], dim=1)

            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def load_weights(self, pretrained_weights, strict=True, **kwargs):
        filter = kwargs.get('filter', lambda key: key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {key: value for key, value in pretrained_weights.items() if filter(key)}
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=strict)
        return self

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config, key):
        if type(config[key]) == str:
            # reuse same model
            self.first_stage_model[key] = self.first_stage_model[config[key]]
        elif type(config[key]) == DictConfig:
            model = load_model(config[key]['dir'], config[key]['checkpoint'])
            self.first_stage_model[key] = model.to(self.device).eval()
            self.first_stage_model[key] = self.first_stage_model[key].to(self.device)
            self.first_stage_model[key].train = disabled_train
            for param in self.first_stage_model[key].parameters():
                param.requires_grad = False
        else:
            raise TypeError(f'Exprected either a dict model configuration for {key} modality or str to map to another model previously known')

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = load_model(config.dir, config.checkpoint)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c, _, _ = self.cond_stage_model.encode(c)
                if len(c.shape) == 4:
                    c = rearrange(c, 'b c h w -> b c (h w)')
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    @torch.no_grad()
    def get_input(self, batch, return_first_stage_outputs=False, cond_key=None, bs=None, return_x=False,
                  return_extra_data=False, return_gtv=False):
        sources = self.source_keys
        target = self.target_key

        x_src = {}
        for key in sources:
            x_src[key] = super().get_input(batch, key)

        x_tgt = super().get_input(batch, target)

        if bs is not None:
            for key, elem in x_src.items():
                x_src[key] = elem[:bs]
            x_tgt = x_tgt[:bs]

        for key in sources:
            x_src[key] = x_src[key].to(self.device)
        x_tgt= x_tgt.to(self.device)

        z_src = {}
        for key in sources:
            z_src[key], _, _ = self.encode_first_stage(x_src[key], key)
            z_src[key] = self.get_first_stage_encoding(z_src[key]).detach()

        z_tgt, _, _ = self.encode_first_stage(x_tgt, target)
        z_tgt = self.get_first_stage_encoding(z_tgt).detach()

        if cond_key is not None:
            xc = super().get_input(batch, cond_key)
            if not self.cond_stage_trainable:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}
        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z_src, z_tgt, c]

        if return_first_stage_outputs:
            xrec_src = {}
            for key in sources:
                xrec_src[key] = self.decode_first_stage(z_src[key], key)
            xrec_tgt = self.decode_first_stage(z_tgt, target)
            out.extend([x_src, x_tgt, xrec_src, xrec_tgt, sources, target])

        if return_extra_data:
            out.extend([batch['patient'], batch['slice']])
        if return_gtv:
            out.extend([batch['gtv']])

        return out

    @torch.no_grad()
    def decode_first_stage(self, z, key, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model[key].quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        return self.first_stage_model[key].decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x, key):
        """
        :param x: Tensor of dim [N, C, H, W] in pixel space
        :param key: indicate which modality encoder are required
        :return: return latent space representation of the x modality
        """
        return self.first_stage_model[key].encode(x)

    def shared_step(self, batch, **kwargs):
        x_src, x_tgt, c = self.get_input(batch, cond_key=self.condition_key)
        loss = self(x_src, x_tgt, c)
        return loss

    def forward(self, x_src, x_tgt, c, *args, **kwargs):
        t_tgt = torch.randint(0, self.num_timesteps, (x_tgt.shape[0],), device=self.device).long()

        if self.noised_source:
            t_src = {}
            for key in self.source_keys:
                t_src[key] = torch.randint(0, self.num_timesteps, (x_tgt.shape[0],), device=self.device).long()
        else:
            t_src = None

        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)

        return self.p_losses(x_src, x_tgt, c, t_src, t_tgt, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        return self.model(x_noisy, t, **cond)



    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_losses(self, x_src: dict, x_tgt: torch.Tensor,  cond, t_src: dict, t_tgt: torch.Tensor, noise=None):

        noise_tgt = torch.randn_like(x_tgt)
        x_noisy_tgt = self.q_sample(x_start=x_tgt, t=t_tgt, noise=noise_tgt)

        if t_src is not None:
            t_all = torch.empty((x_tgt.shape[0], 1), device=self.device)
            noisy_src_all = torch.empty((x_tgt.shape[0], 1, x_tgt.shape[2], x_tgt.shape[3]), device=self.device)
            x_noisy_src = {}
            for key in self.source_keys:
                noise_src = torch.randn_like(x_src[key])
                x_noisy_src[key] = self.q_sample(x_start=x_src[key], t=t_src[key], noise=noise_src)
                noisy_src_all = torch.cat((noisy_src_all, noise_src), dim=1)
                t_all = torch.hstack((t_all, t_src[key][:, None]))
            t_all = t_all[:, 1:]
            noisy_src_all = noisy_src_all[:, 1:, :, :]  # remove tmp val
            noise = torch.cat((noisy_src_all, noise_tgt), dim=1)
            t_all = torch.hstack((t_all, t_tgt[:, None]))
        else:
            t_all = t_tgt[:, None]
            noise = noise_tgt

        if self.concat_mode:
            x_all = torch.empty((x_tgt.shape[0], 1, x_tgt.shape[2], x_tgt.shape[3]), device=self.device)
            for key in self.source_keys:
                if t_src is not None:
                    x_all = torch.cat([x_all, x_noisy_src[key]], dim=1)
                else:
                    x_all = torch.cat([x_all, x_src[key]], dim=1)
            x_all = torch.cat((x_all, x_noisy_tgt), dim=1)
            x_all = x_all[:, 1:, :, :]  # remove tmp val
        else:
            x_all = x_tgt
        model_output = self.apply_model(x_all, t_all, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            if self.concat_mode:
                target = torch.empty((x_tgt.shape[0], 1), device=self.device)
                for key in self.source_keys:
                    target = torch.cat([target, x_src[key]], dim=1)
                target = torch.cat([x_src, x_tgt], dim=1)
            else:
                target = x_tgt
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            if self.noised_source:
                raise NotImplementedError()
            else:
                target = self.get_v(x_tgt, noise, t_tgt)
        else:
            raise NotImplementedError()
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        self.logvar = self.logvar.to(self.device)
        logvar_tgt = self.logvar[t_tgt]
        if t_src is not None:
            logvar_src = torch.zeros_like(logvar_tgt)
            for key in self.source_keys:
                logvar_src += self.logvar[t_src[key]]
            logvar_t = (logvar_src + logvar_tgt) / (len(t_src)+1)
        else:
            logvar_t = logvar_tgt
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        if t_src is not None:
            loss_vlb_weight = self.lvlb_weights[t_tgt]
            for key in self.source_keys:
                loss_vlb_weight += self.lvlb_weights[t_src[key]]
            loss_vlb_weight /= len(t_src) + 1
        else:
            loss_vlb_weight = self.lvlb_weights[t_tgt]
        loss_vlb = (loss_vlb_weight * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        t_tgt = t[:, -1:].squeeze()  # last t vector is for target domain
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)
        if self.concat_mode:
            model_out = model_out[:, -4:, :, :]  # 4 last channel are target prediction

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t_tgt, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x[:,model_out.shape[1]:], t=t_tgt, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x[:,model_out.shape[1]:], t=t_tgt)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(model_mean.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        t_tgt = t[:, -1:].squeeze()
        nonzero_mask = (1 - (t_tgt == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.cat([x0.to(self.device), torch.randn_like(x0).to(self.device)], dim=1)
        else:
            img = torch.cat([x0.to(self.device), x_T.to(self.device)], dim=1)
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            x_prev, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)

            img = torch.cat([x0.to(self.device), x_prev], dim=1)

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    def phi(self, t):
        t_transformed = t.clone().detach()
        if self.source_noised_function == 'vanilla_noise':
            # follow the noise of the origin backward diffusion process
            return t_transformed
        elif self.source_noised_function == "skip_noise":
            # follow the noise apply to the generation but applied proportionally  to gamma a lower amount.
            t_transformed = t_transformed - self.num_timesteps * self.source_noised_gamma
            t_transformed[t_transformed < 0] = 1
            t_transformed[t_transformed > self.num_timesteps] = self.num_timesteps
            return t_transformed
        elif self.source_noised_function == "reverse_skip_noise":
            t_transformed = self.num_timesteps * (1 - self.source_noised_gamma) - t_transformed
            t_transformed[t_transformed < 0] = 1
            t_transformed[t_transformed > self.num_timesteps] = self.num_timesteps - 1
            return t_transformed
        elif self.source_noised_function == "no_noise":
            t_transformed[:] = 1
            return t_transformed
        else:
            raise NotImplementedError

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      x0=None, img_callback=None, start_T=None, log_every_t=None):
        """
        reverse process of DDPM
        """

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            x_T = torch.randn_like(x0).to(self.device)
        else:
            x_T = x_T.to(self.device)

        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)

        if self.noised_source:
            t_src = torch.full((b,), timesteps, device=device, dtype=torch.long)
            t_src = self.phi(t_src).long()
            x0 = self.q_sample(x_start=x0, t=t_src)
        if self.concat_mode:
            img = torch.cat([x0.to(self.device), x_T], dim=1)
        else:
            img = x_T

        intermediates = [img]

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.noised_source:
                tc = self.phi(ts).long()
                t_all = torch.vstack((tc, ts)).T
            else:
                t_all = ts

            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            x_prev = self.p_sample(img, cond, t_all,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if self.noised_source:
                t_src = self.phi(ts).long()
                x0 = self.q_sample(x_start=x0, t=t_src)
            if self.concat_mode:
                img = torch.cat([x0.to(self.device), x_prev], dim=1)
            else:
                img = x_prev

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        output = img[:, -4:, :, :] if self.concat_mode else img
        if return_intermediates:
            return output, intermediates
        return output

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised, x0=x0)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps=None, **kwargs):
        if ddim_steps is None:
            ddim_steps = self.ddim_steps
        ddim = True if self.ddim_steps is not None else False
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, cond, verbose=False, **kwargs)
        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def generate(self, x_src, condition=None, ddim=True, ddim_steps=None, ddim_unconditional_scale=None):
        """
        :param x_src: a dictionnary of Tensor modality
        :param condition: [N] Tensor of class label
        :param ddim:
        :param ddim_steps:
        :param ddim_unconditional_scale: scale us for classifier-free guidance.
        """

        assert len(x_src) == len(self.source_keys)

        if ddim_unconditional_scale is None:
            ddim_unconditional_scale = self.ddim_unconditional_scale

        z_src = {}
        for key in self.source_keys:
            z_src[key], _, _ = self.encode_first_stage(x_src[key], key)
            z_src[key] = self.get_first_stage_encoding(z_src[key]).detach()

        batch_size = z_src[next(iter(z_src))].shape[0]

        if self.model.conditioning_key is not None:
            assert condition is not None
            if self.cond_stage_trainable:
                condition = self.get_learned_conditioning(condition)
            # use if we are doing a classifier-free guidance
            unconditinal_class_label = self.cond_stage_model.get_unconditional_class(batch_size).to(self.device)
            unconditional_conditioning = self.get_learned_conditioning(unconditinal_class_label)
        else:
            unconditional_conditioning = None

        with self.ema_scope():
            samples, _ = self.sample_log(cond=condition, batch_size=batch_size, ddim=ddim, ddim_steps=ddim_steps,
                                         ddim_unconditional_scale=ddim_unconditional_scale,
                                         unconditional_conditioning=unconditional_conditioning, eta=1.0, x0=z_src,
                                         temperature=self.temperature)
        x_samples = self.decode_first_stage(samples, self.target_key)

        return x_samples

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        z_src, z_tgt, c, x_src, x_tgt, xrec_src, xrec_tgt, source, target = self.get_input(batch,
                                                                                           return_first_stage_outputs=True,
                                                                                           cond_key=self.condition_key)

        """ we stop generating images for the validation step as it rise time to process one epoch 4 times.
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)

        ddim = True if self.ddim_steps is not None else False
        with self.ema_scope():
            samples, _ = self.sample_log(cond=c, batch_size=z_tgt.shape[0], ddim=ddim, ddim_steps=self.ddim_steps, eta=1.0,
                                         x0=z_src, temperature=self.temperature)
        x_samples = self.decode_first_stage(samples, self.target_key)

        psnr = PeakSignalNoiseRatio().to(self.device)
        psnr_ = psnr(x_samples, x_tgt)
        self.log('val/psnr', psnr_, on_step=False, on_epoch=True, sync_dist=True, batch_size=len(x_samples))

        ssim = StructuralSimilarityIndexMeasure().to(self.device)
        ssim_ = ssim(x_samples, x_tgt)
        self.log('val/ssim', ssim_, on_step=False, on_epoch=True, sync_dist=True, batch_size=len(x_samples))
        """

    def on_test_epoch_start(self):
        self.psnr_results = torch.empty([0], device=self.device)
        self.local_psnr_results = torch.empty([0], device=self.device)
        self.ssim_results = torch.empty([0], device=self.device)
        self.local_ssim_results = torch.empty([0], device=self.device)
        self.lpips_results = torch.empty([0], device=self.device)
        self.global_mse_results = torch.empty([0], device=self.device)
        self.local_mse_results = torch.empty([0], device=self.device)

    def test_step(self, batch, batch_idx):
        """
        compute loss in comparison to expected MRI generate.
        """

        z_src, z_tgt, c, x_src, x_tgt, xrec_src, xrec_tgt, source, target, gtv = self.get_input(batch,
                                                                                        return_first_stage_outputs=True,
                                                                                        cond_key=self.condition_key,
                                                                                        return_gtv=True)

        ddim = True if self.ddim_steps is not None else False

        batch_size = z_tgt.shape[0]

        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
                # use if we are doing a classifier-free guidance
            unconditional_class_label = self.cond_stage_model.get_unconditional_class(batch_size).to(self.device)
            unconditional_conditioning = self.get_learned_conditioning(unconditional_class_label)
        else:
            unconditional_conditioning = None

        samples, _ = self.sample_log(cond=c, batch_size=batch_size, ddim=ddim, eta=1.0, x0=z_src,
                                     temperature=self.temperature, ddim_unconditional_scale=self.ddim_unconditional_scale,
                                     unconditional_conditioning=unconditional_conditioning)
        x_samples = self.decode_first_stage(samples, self.target_key)

        masked_x_samples = x_samples * gtv.int().float()
        masked_x_tgt = x_tgt * gtv.int().float()

        # ------------------------- Compute Loss ---------------------------
        # apply gtv mask to x_samples and x_tgt to only keep the tumor area segment from preMRI
        global_mse_scores = F.mse_loss(x_samples, x_tgt, reduction='none').sum(dim=(1, 2, 3))
        global_mse_scores /= x_samples.size(1) * x_samples.size(2) * x_samples.size(3)
        self.global_mse_results = torch.cat([self.global_mse_results, global_mse_scores], dim=0)

        local_mse_scores = F.mse_loss(masked_x_samples, masked_x_tgt, reduction='none').sum(dim=(1, 2, 3))
        # local_mse_scores correspond to the mse_loss between each sample.
        # Thus, we divide by the masked area to get the mean mse on the masked area.
        n_pixel_mask = (gtv == 1).sum(dim=(3, 2, 1))
        local_mse_scores /= n_pixel_mask

        self.local_mse_results = torch.cat([self.local_mse_results, local_mse_scores], dim=0)

        psnr_scores = peak_signal_noise_ratio(x_samples, x_tgt, reduction='none', dim=(1, 2, 3), data_range=(-1, 1))
        self.psnr_results = torch.cat([self.psnr_results, psnr_scores], dim=0)

        # data range for target images are [-1; 1] which gives a range of 2 thus 2^2 = 4
        local_psnr_scores = 10 * torch.log10(4/(local_mse_scores))
        self.local_psnr_results = torch.cat([self.local_psnr_results, local_psnr_scores], dim=0)

        ssim_scores = structural_similarity_index_measure(x_samples, x_tgt, reduction='none')
        self.ssim_results = torch.cat([self.ssim_results, ssim_scores], dim=0)

        _, ssim_map = ssim(x_samples, x_tgt, return_full_image=True)
        local_ssim_scores = (ssim_map * gtv.int().float()).sum(dim=(1, 2, 3)) / n_pixel_mask
        self.local_ssim_results = torch.cat([self.local_ssim_results, local_ssim_scores], dim=0)

        for i in range(x_samples.size(0)):
            # lpips used a classifier train on rgb images to compute distance
            max_val = max(abs(x_samples[i].max()), abs(x_samples[i].min()))
            max_val = max_val if max_val > 1 else 1

            lpips_scores = learned_perceptual_image_patch_similarity(img1=x_samples[i].repeat(1, 3, 1, 1) / max_val,
                                                                     img2=x_tgt[i].repeat(1, 3, 1, 1), reduction='none')
            lpips_scores = lpips_scores.reshape(-1)
            self.lpips_results = torch.cat([self.lpips_results, lpips_scores], dim=0)

    def on_test_epoch_end(self):
        self.log_dict({
            'psnr_mean': self.psnr_results.mean().item(),
            'psnr_std': self.psnr_results.std().item(),
            'local_psnr_mean': self.local_psnr_results.mean().item(),
            'local_psnr_std': self.local_psnr_results.std().item(),
            'ssim_mean': self.ssim_results.mean().item(),
            'ssim_std': self.ssim_results.std().item(),
            'local_ssim_mean': self.local_ssim_results.mean().item(),
            'local_ssim_std': self.local_ssim_results.std().item(),
            'lpips_mean': self.lpips_results.mean().item(),
            'lpips_std': self.lpips_results.std().item(),
            'global_mse_mean': self.global_mse_results.mean().item(),
            'global_mse_std': self.global_mse_results.std().item(),
            'local_mse_mean': self.local_mse_results.mean().item(),
            'local_mse_std': self.local_mse_results.std().item(),
        })

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=50, return_keys=None,
                   quantize_denoised=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=True, **kwargs):
        print('starting logging phase')
        use_ddim = True if self.ddim_steps is not None else False

        log = dict()
        (z_src, z_tgt, c, x_src, x_tgt, xrec_src, xrec_tgt, source,
         target, patient, slice_) = self.get_input(batch, return_first_stage_outputs=True, return_extra_data=True,
                                                   cond_key=self.condition_key, bs=N)

        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)

        log['patient'] = patient
        log['slice'] = slice_

        N = min(x_tgt.shape[0], N)
        for key in self.source_keys:
            log[f"src_{key}"] = x_src[key]
            log[f"recon_{key}"] = xrec_src[key]
        log[f"tgt_{target}"] = x_tgt
        log[f"recon_{target}"] = xrec_tgt

        if N < x_tgt.shape[0]:
            for key in self.source_keys:
                x_src[key] = x_src[key][:N]
                z_src[key] = z_src[key][:N]

        if self.condition_key is not None:
            log[f"conditioning"] = c

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=1.0, x0=z_src,
                                                         temperature=self.temperature)
            x_samples = self.decode_first_stage(samples, self.target_key)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=1.0,
                                                             quantize_denoised=True, x0=z_src,
                                                             temperature=self.temperature)
                x_samples = self.decode_first_stage(samples.to(self.device), self.target_key)
                log["samples_x0_quantized"] = x_samples


        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               x0=z_src,
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(L.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t, context=None)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=0)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out
