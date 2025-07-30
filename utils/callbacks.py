"""
source: https://github.com/CompVis/latent-diffusion/blob/main/main.py
"""

import os
import time
from typing import Literal
from omegaconf import OmegaConf
import numpy as np
import torch
import torchvision
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only, rank_zero_info
from data.MRIDataModule import MRIDataModule
from PIL import Image, ImageDraw, ImageFont
import nibabel as nib


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config, data_module: MRIDataModule):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.data_module: MRIDataModule = data_module

    def on_exception(self, trainer, pl_module, KeyboardInterrupt):
        print("Summoning checkpoint.")
        ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
        trainer.save_checkpoint(ckpt_path)
        print("weight saved.")

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

            print("Data split")
            self.data_module.save_patient_split(dir_path=self.cfgdir)
            self.data_module.save_patient_split_from_dataset(dir_path=self.cfgdir)


        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self,
                 frequence_unit: Literal['batch', 'epoch'],
                 batch_frequency_train,
                 batch_frequency_val,
                 max_images,
                 clamp=False,
                 increase_log_steps=True,
                 rescale=True,
                 disabled=False,
                 log_on_batch_idx=False,
                 log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq_train = batch_frequency_train
        self.batch_freq_val = batch_frequency_val
        self.max_images = max_images
        self.logger_log_images = {
            TensorBoardLogger: self._tensorboard,
        }
        self.log_steps_train = [2 ** n for n in range(int(np.log2(self.batch_freq_train)) + 1)]
        if not increase_log_steps:
            self.log_steps_train = [self.batch_freq_train]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.batch_freq_val)) + 1)]
        if not increase_log_steps:
            self.log_steps_val = [self.batch_freq_val]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.last_train_batch = None  # an attribute which store the last compute training batch
        self.last_val_batch = None  # an attribute which store the last compute validation batch
        self.frequence_unit = frequence_unit

    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, trainer, save_dir, split, images,
                  global_step, current_epoch, batch_idx, patient=None, slice_=None):
        root = os.path.join(save_dir, "images", split)
        padding = 2
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=len(images[k]), padding=padding)
            if self.rescale:
                grid = (grid + 1.0) / 2.0
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx
                )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            pil_image = Image.fromarray(grid)

            # will print information text inside image (patient + slice of MRI)
            if patient is not None and slice_ is not None and k in ['recon_preMRI', 'recon_postMRI', 'src_preMRI', 'tgt_postMRI', 'samples']:
                draw = ImageDraw.Draw(pil_image)
                font = ImageFont.load_default()

                for i in range(len(images[k])):
                    width = (images[k][0].shape[1] + padding) * (i + 1)
                    height = images[k][0].shape[2] + padding * 2
                    position = (width - 100, height - 20)
                    text_to_add = f"{patient[i]}-{slice_[i]}"
                    draw.text(position, text_to_add, font=font, fill=(255, 255, 255))

            pil_image.save(path)

    def log_img(self, trainer, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx or split == "val" else pl_module.global_step
        if (self.check_frequency(check_idx, split) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
                patient = images.pop('patient')
                slice_ = images.pop('slice')
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(trainer, pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx, patient, slice_)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx, split='train'):
        if self.frequence_unit == "epoch": # we don't need to validate frequency for logging every epoch
            return True
        if split == 'train':
            if ((check_idx % self.batch_freq_train) == 0 or (check_idx in self.log_steps_train)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps_train.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        if split == 'val':
            if ((check_idx % self.batch_freq_val) == 0 or (check_idx in self.log_steps_val)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps_val.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.last_train_batch = batch
        if self.frequence_unit == 'batch':
            if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
                self.log_img(trainer, pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.last_val_batch = batch
        if self.frequence_unit == 'batch':
            if not self.disabled and pl_module.global_step > 0:
                self.log_img(trainer, pl_module, batch, batch_idx, split="val")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.frequence_unit == 'epoch':
            self.log_img(trainer, pl_module, self.last_train_batch, 0, split="train")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.frequence_unit == 'epoch':
            self.log_img(trainer, pl_module, self.last_val_batch, 0, split="val")


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, L_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass