import sys
import torch
import torch.nn.functional as F

from torch import nn
from models.psp.encoders import psp_encoders
from models.psp.stylegan2.model import Generator
from utils.class_registry import ClassRegistry
from utils.common_utils import get_keys
from utils.model_utils import toogle_grad
from configs.paths import DefaultPaths
from argparse import Namespace


sys.path.append("./utils")
methods_registry = ClassRegistry()


@methods_registry.add_to_registry("fse_full", stop_args=("self", "checkpoint_path"))
class FSEFull(nn.Module):
    def __init__(self,
                 device="cpu",
                 paths=DefaultPaths,
                 checkpoint_path=None,
                 inverter_pth=None):
        super(FSEFull, self).__init__()
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "stylegan_size": 1024
        }
        self.opts.update(paths)
        self.opts = Namespace(**self.opts)

        self.device = device
        self.inverter_pth = inverter_pth

        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.stylegan_size, 512, 8)
        self.latent_avg = None

        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()

    def load_weights(self):
        if self.opts.checkpoint_path != "":
            print(f"Loading from checkpoint: {self.opts.checkpoint_path}")
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
            self.inverter.load_state_dict(get_keys(ckpt, "inverter"), strict=True)
        else:
            print(f"Loading Inverter from Inverter checkpoint: {self.inverter_pth}")
            ckpt = torch.load(self.inverter_pth, map_location="cpu")
            self.inverter.load_state_dict(get_keys(ckpt, "encoder"), strict=True)

        self.inverter = self.inverter.eval().to(self.device)
        toogle_grad(self.inverter, False)

        print("Loading Decoder from", self.opts.stylegan_weights)
        ckpt = torch.load(self.opts.stylegan_weights)
        self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
        self.latent_avg = ckpt['latent_avg'].to(self.device)
        self.decoder = self.decoder.eval().to(self.device)
        toogle_grad(self.decoder, False)

        print("Loading E4E from", self.opts.e4e_path)
        ckpt = torch.load(self.opts.e4e_path, map_location="cpu")
        self.e4e_encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
        self.e4e_encoder = self.e4e_encoder.eval().to(self.device)
        toogle_grad(self.e4e_encoder, False)


    def set_encoder(self):
        self.inverter = psp_encoders.Inverter(opts=self.opts, n_styles=18) 
        self.e4e_encoder = psp_encoders.Encoder4Editing(50, "ir_se", self.opts)
        feat_editor = psp_encoders.ContentLayerDeepFast(6, 1024, 512)
        return feat_editor  # trainable part
    
    def forward(self, x, return_latents=False):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        with torch.no_grad():
            w_recon, predicted_feat = self.inverter.fs_backbone(x)
            w_recon = w_recon + self.latent_avg
                    
            _, w_feat = self.decoder(w_recon)

            fused_feat = self.inverter.fuser(torch.cat([predicted_feat, w_feat], dim=1))
            delta = torch.zeros_like(fused_feat)  # inversion case

        edited_feat = self.encoder(torch.cat([fused_feat, delta], dim=1))

        images, _ = self.decoder(w_recon, new_feature=edited_feat)

        if return_latents:
            if not self.encoder.training:
                fused_feat = fused_feat.cpu()
                predicted_feat = predicted_feat.cpu()
            return images, w_recon, fused_feat, predicted_feat
        return images
