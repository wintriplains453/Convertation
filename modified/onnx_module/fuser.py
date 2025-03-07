from pathlib import Path

import torch

from utils.model_utils import toogle_grad
from models.psp.encoders import psp_encoders
from utils.common_utils import get_keys
from modified.onnx_module import decoder_without_new_feature, inverter
from modified.onnx_module.utils import opts, export_and_validate

DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/fuser.onnx'


def init_model():
    inverter = psp_encoders.Inverter(opts=opts, n_styles=18)
    ckpt = torch.load(opts.checkpoint_path, map_location='cpu')
    inverter.load_state_dict(get_keys(ckpt, 'inverter'), strict=True)
    inverter = inverter.eval()
    toogle_grad(inverter, False)

    return inverter.fuser


def pt_output(dummy_input=None):
    torch_model = init_model()
    if dummy_input is None:
        w_recon, predicted_feat = inverter.pt_output()
        _, w_feat = decoder_without_new_feature.pt_output()
        dummy_input = torch.cat([predicted_feat, w_feat], dim=1)
    with torch.no_grad():
        fused_feat = torch_model(dummy_input)
    return fused_feat


if __name__ == "__main__":
    torch_model = init_model()
    w_recon, predicted_feat = inverter.pt_output()
    _, w_feat = decoder_without_new_feature.pt_output()
    input_pt = torch.cat([predicted_feat, w_feat], dim=1)
    output_names = ['fused_feat']

    export_and_validate(
        model=torch_model,
        dummy_input=(input_pt,),
        output_onnx_path=MODEL_PATH,
        output_names=output_names,
        skip_export=False,
    )
