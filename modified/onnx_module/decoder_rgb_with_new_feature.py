from pathlib import Path

import torch

from models.psp.stylegan2.model_stylespace import Generator
from modified.onnx_module.utils import export_and_validate, opts
from utils.model_utils import toogle_grad


DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/decoder_rgb_with_new_feature.onnx'


def init_model():
    decoder = Generator(opts.stylegan_size, 512, 8)
    ckpt = torch.load(opts.stylegan_weights, map_location='cpu')
    decoder.load_state_dict(ckpt["g_ema"], strict=False)
    decoder = decoder.eval()
    toogle_grad(decoder, False)
    return decoder


def pt_input():
    edited_ss_list = [(1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512),
                      (1, 512), (1, 256), (1, 256), (1, 128), (1, 128), (1, 64), (1, 64), (1, 32)]
    edited_rgb_list = [(1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 256), (1, 128), (1, 64), (1, 32)]

    edited_ss_list_pt = [torch.randn(*s) for s in edited_ss_list]
    edited_rgb_list_pt = [torch.randn(*s) for s in edited_rgb_list]
    new_feture = torch.randn(1, 512, 64, 64)

    return tuple(edited_ss_list_pt + edited_rgb_list_pt + [new_feture])


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
        'style_10',
        'style_11',
        'style_12',
        'style_13',
        'style_14',
        'style_15',
        'style_16',
        'style_17',
        'to_rgb_stylespace_1',
        'to_rgb_stylespace_2',
        'to_rgb_stylespace_3',
        'to_rgb_stylespace_4',
        'to_rgb_stylespace_5',
        'to_rgb_stylespace_6',
        'to_rgb_stylespace_7',
        'to_rgb_stylespace_8',
        'to_rgb_stylespace_9',
        'new_feature',
    ]
    output_names = ['image']

    export_and_validate(
        model=torch_model,
        dummy_input=dummy_input,
        output_onnx_path=MODEL_PATH,
        input_names=input_names,
        output_names=output_names,
        skip_export=False,
        atol=1e-5,
        opset_version=10,
        dynamo=True,
    )

# Mismatched elements: 781 / 3145728 (0.0248%)
# Max absolute difference: 0.00036621
# Max relative difference: 13.289157
#  x: array([[[[-8.769349e-01, -1.994071e+00, -2.576848e+00, ...,
#           -4.119475e+00, -3.095610e+00, -1.557721e+00],
#          [-1.561052e+00, -2.076708e+00, -2.386229e+00, ...,...
#  y: array([[[[-8.769357e-01, -1.994073e+00, -2.576849e+00, ...,
#           -4.119474e+00, -3.095611e+00, -1.557721e+00],
#          [-1.561053e+00, -2.076710e+00, -2.386232e+00, ...,...
