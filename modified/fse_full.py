from modified.onnx_module.utils import run_onnx, ONNX_MODELS_PATH


def forward(x, return_latents=False):
    x = run_onnx(ONNX_MODELS_PATH / 'interpolate.onnx', x)
    w_recon, predicted_feat = run_onnx(ONNX_MODELS_PATH / 'invert.onnx', x[0])
    return x[0]
