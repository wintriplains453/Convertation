from pathlib import Path

import torch

from models.psp.stylegan2.model_stylespace import GeneratorWOFeature
from modified.onnx_module.utils import opts
from utils.model_utils import toogle_grad
from modified.onnx_module.utils import export_and_validate


DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/decoder_rgb_without_new_feature.onnx'


def init_model():
    decoder = GeneratorWOFeature(opts.stylegan_size, 512, 8)
    ckpt = torch.load(opts.stylegan_weights, map_location='cpu')
    decoder.load_state_dict(ckpt["g_ema"], strict=False)
    decoder = decoder.eval()
    toogle_grad(decoder, False)
    return decoder


def pt_input():
    edited_ss_list = [(1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512)]
    edited_rgb_list = [(1, 512), (1, 512), (1, 512), (1, 512), (1, 512)]

    edited_ss_list_pt = [torch.randn(*s) for s in edited_ss_list]
    edited_rgb_list_pt = [torch.randn(*s) for s in edited_rgb_list]

    return tuple(edited_ss_list_pt + edited_rgb_list_pt)


def pt_output(dummy_input=None):
    torch_model = init_model()
    if dummy_input is None:
        dummy_input = pt_input()
    with torch.no_grad():
        image = torch_model(dummy_input)
    return image


if __name__ == "__main__":
    torch_model = init_model()
    dummy_input = pt_input()

    input_names = [
        'style_1',
        'style_2',
        'style_3',
        'style_4',
        'style_5',
        'style_6',
        'style_7',
        'style_8',
        'style_9',
        'to_rgb_stylespace_1',
        'to_rgb_stylespace_2',
        'to_rgb_stylespace_3',
        'to_rgb_stylespace_4',
        'to_rgb_stylespace_5',
    ]
    output_names = ['image', 'feature']

    export_and_validate(
        model=torch_model,
        dummy_input=dummy_input,
        output_onnx_path=MODEL_PATH,
        input_names=input_names,
        output_names=output_names,
        skip_export=False,
        atol=1e-5,
        opset_version=10,
        dynamo=False,
    )

# Mismatched elements: 853 / 2097152 (0.0407%)
# Max absolute difference: 0.00140381
# Max relative difference: 4.2898216
#  x: array([[[[-4.391115e-02, -4.074199e-01, -6.839443e-01, ...,
#           -9.333425e-01, -8.007776e-01, -3.251358e-01],
#          [ 4.812788e-01, -4.635905e-01, -9.420992e-01, ...,...
#  y: array([[[[-4.391071e-02, -4.074189e-01, -6.839458e-01, ...,
#           -9.333456e-01, -8.007814e-01, -3.251367e-01],
#          [ 4.812765e-01, -4.635886e-01, -9.420974e-01, ...,...
