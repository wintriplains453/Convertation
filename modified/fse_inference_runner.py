import numpy as np

from modified.onnx_module.utils import run_onnx, ONNX_MODELS_PATH
from modified.fse_full import forward as fse_full
from modified.latent_editor import get_edited_latent


def run_on_batch(input):

    image, w_recon, fused_feat, predicted_feat = fse_full(input)

    x = run_onnx(ONNX_MODELS_PATH / 'interpolate.onnx', (input,))
    x = x[0]

    w_e4e = run_onnx(ONNX_MODELS_PATH / 'e4e_encoder.onnx', (x,))
    w_e4e = w_e4e[0]


    result_batch = {
        'latents': w_recon,
        'fused_feat': fused_feat,
        'predicted_feat': predicted_feat,
        'w_e4e': w_e4e,
        'input': input
    }

    return image, result_batch


def run_editing_core(latent, w_e4e, fused_feat, editing_name, editing_degree):
    edited_latents = get_edited_latent(
        latent,
        editing_name,
        editing_degree
    )

    edited_w_e4e = get_edited_latent(
        w_e4e,
        editing_name,
        editing_degree
    )

    is_stylespace = isinstance(edited_latents, tuple) # tuple[list[np.ndarray], list[np.ndarray]]

    e4e_inv, fs_x = run_onnx(ONNX_MODELS_PATH / 'decoder_without_new_feature.onnx', (w_e4e,))

    if is_stylespace:
        e4e_edit, fs_y = run_onnx(
            ONNX_MODELS_PATH / 'decoder_rgb_without_new_feature.onnx',
            tuple(edited_w_e4e[0] + edited_w_e4e[1])
        )
    else:
        e4e_edit, fs_y = run_onnx(ONNX_MODELS_PATH / 'decoder_without_new_feature.onnx', (edited_w_e4e,))
    delta = fs_x - fs_y

    edited_feat = run_onnx(
        ONNX_MODELS_PATH / 'encoder.onnx',
        (np.concatenate((fused_feat, delta), axis=1),)
    )
    edited_feat = edited_feat[0]

    if is_stylespace:
        image_edit = run_onnx(
            ONNX_MODELS_PATH / 'decoder_rgb_with_new_feature.onnx',
            tuple(edited_latents[0] + edited_latents[1] + [edited_feat])
        )
    else:
        image_edit = run_onnx(
            ONNX_MODELS_PATH / 'decoder_with_new_feature.onnx',
            tuple([edited_latents] + [edited_feat])
        )
    image_edit = image_edit[0]

    return image_edit


def run_editing_on_batch(method_res_batch, editing_name, editing_degree):
    latent = method_res_batch['latents']
    w_e4e = method_res_batch['w_e4e']
    fused_feat = method_res_batch['fused_feat']
    return run_editing_core(latent, w_e4e, fused_feat, editing_name, editing_degree)


def run_pre_editor(x):
    return run_onnx(ONNX_MODELS_PATH / 'pre_editor.onnx', (x,))
