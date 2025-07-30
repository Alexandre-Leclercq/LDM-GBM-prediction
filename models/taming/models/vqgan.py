import torch
import torch.nn.functional as F
import lightning as L

from utils.autoloading import instantiate_from_config, load_model

from models.taming.modules.diffusionmodules.model import Encoder, Decoder
from models.taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class VQModel(L.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 first_stage_vae=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="mri",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()

        self.automatic_optimization = False

        if first_stage_vae is not None:
            self.first_stage_vae = load_model(dir_trained_model=first_stage_vae['dir'],
                                                  checkpoint=first_stage_vae['checkpoint'])
            self.first_stage_vae.freeze()
        else:
            self.first_stage_vae = None

        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def load_weights(self, pretrained_weights, strict=True, **kwargs):
        filter = kwargs.get('filter', lambda key: key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {key: value for key, value in pretrained_weights.items() if filter(key)}
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=strict)
        return self

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        if self.first_stage_vae is not None:
            x = self.first_stage_vae.encode(x)[0]
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        if self.first_stage_vae is not None:
            dec = self.first_stage_vae.decode(dec)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        if self.first_stage_vae is not None:
            dec = self.first_stage_vae.decode_code(dec)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        
        return x.float()

    def training_step(self, batch):
        x = self.get_input(batch, self.image_key)

        optimizer_autoencoder, optimizer_discriminator = self.optimizers()

        # train autoencoder part
        self.toggle_optimizer(optimizer_autoencoder)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")

        # ----------------- Log Scalars AE Loss ----------------------
        for metric_name, metric_val in log_dict_ae.items():
            self.log(metric_name, metric_val, prog_bar=True,
                     on_step=True, on_epoch=True, sync_dist=True, logger=True
                     )

        self.manual_backward(aeloss)
        optimizer_autoencoder.step()
        optimizer_autoencoder.zero_grad()
        self.untoggle_optimizer(optimizer_autoencoder)

        # train discriminator
        self.toggle_optimizer(optimizer_discriminator)
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")

        # ----------------- Log Scalars Discriminant Loss ----------------------
        for metric_name, metric_val in log_dict_disc.items():
            self.log(metric_name, metric_val, prog_bar=True,
                     on_step=True, on_epoch=True, sync_dist=True, logger=True
                     )

        self.manual_backward(discloss)
        optimizer_discriminator.step()
        optimizer_discriminator.zero_grad()
        self.untoggle_optimizer(optimizer_discriminator)

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        xrec, qloss = self(x)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        # ----------------- Log Scalars AE Loss ----------------------
        for metric_name, metric_val in log_dict_ae.items():
            self.log(metric_name, metric_val, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)

        ssim = StructuralSimilarityIndexMeasure().to(x.device)
        ssim_ = ssim(xrec, x)
        self.log('val/ssim', ssim_, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        # ----------------- Log Scalars Discriminant Loss ----------------------
        for metric_name, metric_val in log_dict_disc.items():
            self.log(metric_name, metric_val, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)

        return self.log_dict

    def test_step(self, batch, batch_idx):

        x = self.get_input(batch, self.image_key)
        target = x

        xrec, qloss = self(x)

        # ------------------------- Compute Loss ---------------------------
        loss, _ = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="test")

        psnr = PeakSignalNoiseRatio().to(x.device)
        psnr_ = psnr(xrec, target)

        ssim = StructuralSimilarityIndexMeasure().to(x.device)
        ssim_ = ssim(xrec, target)

        # lpips used a classifier train on rgb images to compute distance
        lpips = LearnedPerceptualImagePatchSimilarity().to(x.device)
        max_val = max(abs(xrec.max()), abs(xrec.min()))
        max_val = max_val if max_val > 1 else 1
        lpips_ = lpips(xrec.repeat(1, 3, 1, 1)/max_val, target.repeat(1, 3, 1, 1))

        log_dict = {
            'loss_ae': loss,
            'psnr': psnr_,
            'ssim': ssim_,
            'lpips': lpips_,
        }

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.AdamW(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        gtv = self.get_input(batch, 'gtv')
        xrec, _ = self(x)
        if x.shape[1] > 3:
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["input"] = x
        log["gtv"] = gtv
        log[f"reconstruction"] = xrec
        log['patient'] = batch['patient']
        log['slice'] = batch['slice']
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="mri",
                 colorize_nlabels=None,
                 monitor=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels, monitor=monitor)


    def training_step(self, batch):
        x = self.get_input(batch, self.image_key)

        optimizer_autoencoder = self.optimizers()

        # train autoencoder part
        self.toggle_optimizer(optimizer_autoencoder)

        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="train")

        # ----------------- Log Scalars AE Loss ----------------------
        for metric_name, metric_val in log_dict_ae.items():
            self.log(metric_name, metric_val, prog_bar=True,
                     on_step=True, on_epoch=True, sync_dist=True, logger=True
                     )

        self.manual_backward(aeloss)
        optimizer_autoencoder.step()
        optimizer_autoencoder.zero_grad()
        self.untoggle_optimizer(optimizer_autoencoder)

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        xrec, qloss = self(x)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="val")

        # ----------------- Log Scalars AE Loss ----------------------
        for metric_name, metric_val in log_dict_ae.items():
            self.log(metric_name, metric_val, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)

        ssim = StructuralSimilarityIndexMeasure().to(x.device)
        ssim_ = ssim(xrec, x)
        self.log('val/ssim', ssim_, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, logger=True)

        return self.log_dict


    def test_step(self, batch, batch_idx):

        x = self.get_input(batch, self.image_key)
        target = x

        xrec, qloss = self(x)

        # ------------------------- Compute Loss ---------------------------
        loss, _ = self.loss(qloss, x, xrec, 0, self.global_step, last_layer=self.get_last_layer(), split="test")

        psnr = PeakSignalNoiseRatio().to(x.device)
        psnr_ = psnr(xrec, target)

        ssim = StructuralSimilarityIndexMeasure().to(x.device)
        ssim_ = ssim(xrec, target)

        # lpips used a classifier train on rgb images to compute distance
        lpips = LearnedPerceptualImagePatchSimilarity().to(x.device)
        max_val = max(abs(xrec.max()), abs(xrec.min()))
        max_val = max_val if max_val > 1 else 1
        lpips_ = lpips(xrec.repeat(1, 3, 1, 1)/max_val, target.repeat(1, 3, 1, 1))

        log_dict = {
            'loss_ae': loss,
            'psnr': psnr_,
            'ssim': ssim_,
            'lpips': lpips_,
        }

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return log_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        gtv = self.get_input(batch, 'gtv')
        xrec, _ = self(x)
        if x.shape[1] > 3:
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["input"] = x
        log["gtv"] = gtv
        log[f"reconstruction"] = xrec
        log['patient'] = batch['patient']
        log['slice'] = batch['slice']
        return log
