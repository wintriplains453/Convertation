import numpy as np

from modified.onnx_module.utils import run_onnx, ONNX_MODELS_PATH
from modified.fse_full import forward as fse_full
from modified.latent_editor import get_edited_latent


def run_on_batch(inputs):

    images, w_recon, fused_feat, predicted_feat = fse_full(inputs)

    x = run_onnx(ONNX_MODELS_PATH / 'interpolate.onnx', (inputs,))
    x = x[0]

    w_e4e = run_onnx(ONNX_MODELS_PATH / 'e4e_encoder.onnx', (x,))
    w_e4e = w_e4e[0]


    result_batch = {
        'latents': w_recon,
        'fused_feat': fused_feat,
        'predicted_feat': predicted_feat,
        'w_e4e': w_e4e,
        'inputs': inputs
    }

    return images, result_batch

def run_editing_on_batch(method_res_batch, editing_name, editing_degree):
    latent = method_res_batch['latents']
    edited_latents = get_edited_latent(
        latent,
        editing_name,
        editing_degree
    )
    return edited_latents
