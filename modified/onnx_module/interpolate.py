from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from modified.onnx_module.utils import export_and_validate

DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/interpolate.onnx'


class Interpolate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.interpolate(
            x,
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )
        return x

def pt_output():
    torch_model = Interpolate()
    dummy_input = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        torch_output = torch_model(dummy_input)
    return torch_output


if __name__ == '__main__':
    model = Interpolate()
    dummy_input = torch.randn(1, 3, 1024, 1024)

    output_names = ['output']

    export_and_validate(
        model=model,
        dummy_input=(dummy_input,),
        output_onnx_path=MODEL_PATH,
        output_names=output_names,
        skip_export=True,
    )
