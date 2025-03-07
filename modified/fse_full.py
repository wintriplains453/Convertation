import numpy as np

from modified.onnx_module.utils import run_onnx, ONNX_MODELS_PATH


def forward(x, return_latents=False):
    x = run_onnx(ONNX_MODELS_PATH / 'interpolate.onnx', x)
    w_recon, predicted_feat = run_onnx(ONNX_MODELS_PATH / 'invert.onnx', x[0])
    _, w_feat = run_onnx(ONNX_MODELS_PATH / 'decoder_without_new_feature_onnx.onnx', w_recon)
    fused_feat = run_onnx(
        ONNX_MODELS_PATH / 'fuser.onnx',
        np.concatenate((predicted_feat, w_feat), axis=1)
    )
    delta = np.zeros_like(fused_feat)
    return fused_feat[0]
