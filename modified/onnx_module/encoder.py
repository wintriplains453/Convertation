from pathlib import Path

import torch

from models.psp.encoders import psp_encoders
from utils.common_utils import get_keys
from modified.onnx_module import fuser
from modified.onnx_module.utils import opts, export_and_validate

DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/encoder.onnx'


def init_model():
    encoder = psp_encoders.ContentLayerDeepFast(6, 1024, 512)
    ckpt = torch.load(opts.checkpoint_path, map_location='cpu')
    encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
    encoder = encoder.eval()
    return encoder


def pt_output(dummy_input=None):
    torch_model = init_model()
    if dummy_input is None:
        fused_feat = fuser.pt_output()
        delta = torch.zeros_like(fused_feat)
        dummy_input = torch.cat([fused_feat, delta], dim=1)
    with torch.no_grad():
        edited_feat = torch_model(dummy_input)
    return edited_feat


if __name__ == "__main__":
    torch_model = init_model()
    fused_feat = fuser.pt_output()
    delta = torch.zeros_like(fused_feat)
    input_pt = torch.cat([fused_feat, delta], dim=1)
    output_names = ['edited_feat']

    export_and_validate(
        model=torch_model,
        dummy_input=(input_pt,),
        output_onnx_path=MODEL_PATH,
        output_names=output_names,
        skip_export=False,
        dynamo=True,
    )
