from pathlib import Path

import torch

from models.psp.stylegan2.model import Generator
from modified.onnx_module.utils import export_and_validate, opts
from utils.model_utils import toogle_grad
from modified.onnx_module import encoder, fuser, decoder_without_new_feature, inverter


DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/decoder_with_new_feature.onnx'


def init_model():
    decoder = Generator(opts.stylegan_size, 512)
    ckpt = torch.load(opts.stylegan_weights, map_location='cpu')
    decoder.load_state_dict(ckpt["g_ema"], strict=False)
    decoder = decoder.eval()
    toogle_grad(decoder, False)
    return decoder


def pt_output(dummy_input=None):
    torch_model = init_model()
    if dummy_input is None:
        dummy_input = encoder.pt_output()
    with torch.no_grad():
        image = torch_model(dummy_input)
    return image


if __name__ == "__main__":
    torch_model = init_model()
    w_recon, predicted_feat = inverter.pt_output()
    _, w_feat = decoder_without_new_feature.pt_output(w_recon)
    fused_feat = fuser.pt_output(torch.cat([predicted_feat, w_feat], dim=1))
    delta = torch.zeros_like(fused_feat)
    edited_feat = encoder.pt_output(torch.cat([fused_feat, delta], dim=1))

    input_names = ['latent', 'new_feature']
    output_names = ['image']

    export_and_validate(
        model=torch_model,
        dummy_input=(w_recon, edited_feat),
        output_onnx_path=MODEL_PATH,
        output_names=output_names,
        skip_export=False,
        atol=1e-5,
        opset_version=10,
        dynamo=False,
    )
