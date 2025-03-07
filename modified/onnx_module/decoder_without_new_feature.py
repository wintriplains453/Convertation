from pathlib import Path

import torch

from models.psp.stylegan2.model import Generator
from modified.onnx_module.utils import opts
from utils.model_utils import toogle_grad
from modified.onnx_module import inverter
from modified.onnx_module.utils import export_and_validate


DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/decoder_without_new_feature_onnx.onnx'


def init_model():
    decoder = Generator(opts.stylegan_size, 512)
    ckpt = torch.load(opts.stylegan_weights, map_location='cpu')
    decoder.load_state_dict(ckpt["g_ema"], strict=False)
    decoder = decoder.eval()
    toogle_grad(decoder, False)
    return decoder


if __name__ == "__main__":
    torch_model = init_model()
    w_recon_pt, predicted_feat_pt = inverter.pt_output()

    output_names = ['image', 'feature']
    print("w_recon_pt shape:", w_recon_pt.shape)

    export_and_validate(
        model=torch_model,
        dummy_input=w_recon_pt,
        output_onnx_path=MODEL_PATH,
        output_names=output_names,
        skip_export=False,
        atol=1e-5,
        opset_version=10
    )