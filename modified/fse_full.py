from modified.onnx_module.utils import run_onnx, ONNX_MODELS_PATH


def forward(x, return_latents=False):
    x = run_onnx(ONNX_MODELS_PATH / 'interpolate.onnx', x)
    w_recon, predicted_feat = run_onnx(ONNX_MODELS_PATH / 'invert.onnx', x[0])
    _, w_feat = run_onnx(ONNX_MODELS_PATH / 'decoder_without_new_feature_onnx.onnx', w_recon)
    return x[0]
