from pathlib import Path

import torch
import torch.nn as nn

from utils.model_utils import toogle_grad
from models.psp.encoders import psp_encoders
from utils.common_utils import get_keys
import modified.onnx_module.interpolate as interpolate
from modified.onnx_module.utils import opts, export_and_validate

DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/invert.onnx'


class FSBackboneLatentAverage(nn.Module):
    def __init__(self, fs_backbone, latent_avg):
        super().__init__()
        self.fs_backbone = fs_backbone
        self.latent_avg = latent_avg

    def forward(self, x):
        w_recon, predicted_feat = self.fs_backbone(x)
        w_recon = w_recon + self.latent_avg
        return w_recon, predicted_feat


def init_model():
    inverter = psp_encoders.Inverter(opts=opts, n_styles=18)
    ckpt = torch.load(opts.checkpoint_path, map_location='cpu')
    inverter.load_state_dict(get_keys(ckpt, 'inverter'), strict=True)
    inverter = inverter.eval()
    toogle_grad(inverter, False)

    ckpt_stylegan = torch.load(opts.stylegan_weights, map_location='cpu')
    latent_avg = ckpt_stylegan['latent_avg']

    return FSBackboneLatentAverage(fs_backbone=inverter.fs_backbone, latent_avg=latent_avg).eval()


def pt_output(dummy_input=None):
    torch_model = init_model()
    if dummy_input is None:
        dummy_input = interpolate.pt_output()
    with torch.no_grad():
        w_recon_pt, predicted_feat_pt = torch_model(dummy_input)
    return w_recon_pt, predicted_feat_pt


if __name__ == "__main__":
    torch_model = init_model()
    dummy_input = interpolate.pt_output()

    output_names = ['w_recon', 'predicted_feat']

    export_and_validate(
        model=torch_model,
        dummy_input=(dummy_input,),
        output_onnx_path=MODEL_PATH,
        output_names=output_names,
        skip_export=True,
    )

# ONNX export was performed in models/psp/encoders/psp_encoders.py by:  self.avg_pool = AdaptiveAvgPool2dCustom((3, 3))
# Pytorch is launched in models/psp/encoders/psp_encoders.py by:  self.avg_pool = nn.AdaptiveAvgPool2d((3, 3))
# Mismatched elements: 8834 / 9216 (95.9%)
# Max absolute difference: 0.07346225
# Max relative difference: 18.75668
#  x: array([[[ 1.429832, -0.318392, -1.034901, ..., -2.551268,  2.810783,
#          -0.278764],
#         [-0.772323,  2.23823 ,  0.678995, ..., -0.544392, -0.148692,...
#  y: array([[[ 1.419555, -0.327271, -1.022342, ..., -2.541709,  2.802806,
#          -0.274425],
#         [-0.790261,  2.233334,  0.68954 , ..., -0.516931, -0.131541,...
