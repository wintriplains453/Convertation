from pathlib import Path

import torch
import torch.nn as nn

from models.psp.encoders import psp_encoders
from utils.model_utils import toogle_grad
from utils.common_utils import get_keys
from modified.onnx_module import interpolate
from modified.onnx_module.utils import opts, export_and_validate

DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/e4e_encoder.onnx'


class E4EEncoderLatentAverage(nn.Module):
    def __init__(self, e4e_encoder, latent_avg):
        super().__init__()
        self.e4e_encoder = e4e_encoder
        self.latent_avg = latent_avg

    def forward(self, x):
        w_e4e = self.e4e_encoder(x)
        w_e4e = w_e4e + self.latent_avg
        return w_e4e


def init_model():
    e4e_encoder = psp_encoders.Encoder4Editing(50, 'ir_se', opts)
    ckpt = torch.load(opts.e4e_path, map_location='cpu')
    e4e_encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
    e4e_encoder = e4e_encoder.eval()
    toogle_grad(e4e_encoder, False)

    ckpt_stylegan = torch.load(opts.stylegan_weights, map_location='cpu')
    latent_avg = ckpt_stylegan['latent_avg']

    return E4EEncoderLatentAverage(e4e_encoder=e4e_encoder, latent_avg=latent_avg).eval()


def pt_output(dummy_input=None):
    torch_model = init_model()
    if dummy_input is None:
        dummy_input = interpolate.pt_output()
    with torch.no_grad():
        w_e4e = torch_model(dummy_input)
    return w_e4e


if __name__ == "__main__":
    torch_model = init_model()
    dummy_input = interpolate.pt_output()

    output_names = ['w_e4e']

    export_and_validate(
        model=torch_model,
        dummy_input=(dummy_input,),
        output_onnx_path=MODEL_PATH,
        output_names=output_names,
        skip_export=False,
        dynamo=True,
    )
