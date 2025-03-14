from pathlib import Path

import torch
import torch.nn as nn

from utils.model_utils import get_stylespace_from_w
from models.psp.stylegan2.model_original import Generator
from modified.onnx_module.utils import export_and_validate, opts
from utils.model_utils import toogle_grad


DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/decoder_stylespace.onnx'


class StyleSpaceExtractor(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G

    def forward(self, w):
        style_space, to_rgb_stylespaces = get_stylespace_from_w(w, self.G)
        return tuple(style_space + to_rgb_stylespaces)

def init_model():
    G = Generator(1024, 512, 8)
    ckpt = torch.load(opts.stylegan_weights, map_location='cpu')
    G.load_state_dict(ckpt["g_ema"], strict=False)
    G = G.eval()
    toogle_grad(G, False)
    return StyleSpaceExtractor(G=G).eval()


def pt_output():
    torch_model = init_model().eval()
    dummy_input = torch.randn(1, 18, 512)
    with torch.no_grad():
        outputs = torch_model(dummy_input)
    return list(outputs[:17]), list(outputs[17:])


if __name__ == "__main__":
    torch_model = init_model().eval()
    dummy_input = torch.randn(1, 18, 512)

    output_names = [f"style_out_{i}" for i in range(17)] + [f"rgb_out_{i}" for i in range(9)]

    export_and_validate(
        model=torch_model,
        dummy_input=(dummy_input,),
        output_onnx_path=MODEL_PATH,
        output_names=output_names,
        skip_export=False,
        dynamo=False,
    )
