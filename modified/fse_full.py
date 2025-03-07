import numpy as np

from modified.onnx_module.utils import run_onnx, ONNX_MODELS_PATH


def forward(x, return_latents=False):

    x = run_onnx(ONNX_MODELS_PATH / 'interpolate.onnx', (x,))
    x = x[0]

    w_recon, predicted_feat = run_onnx(ONNX_MODELS_PATH / 'invert.onnx', (x,))

    _, w_feat = run_onnx(ONNX_MODELS_PATH / 'decoder_without_new_feature.onnx', (w_recon,))

    fused_feat = run_onnx(
        ONNX_MODELS_PATH / 'fuser.onnx',
        (np.concatenate((predicted_feat, w_feat), axis=1),)
    )
    fused_feat = fused_feat[0]

    delta = np.zeros_like(fused_feat)

    edited_feat = run_onnx(
        ONNX_MODELS_PATH / 'encoder.onnx',
        (np.concatenate((fused_feat, delta), axis=1),)
    )
    edited_feat = edited_feat[0]

    images = run_onnx(
        ONNX_MODELS_PATH / 'decoder_with_new_feature.onnx',
        (w_recon, edited_feat)
    )
    images = images[0]

    return images, w_recon, fused_feat, predicted_feat
