import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

from models.psp.stylegan2.model import Generator

PARAMS_USAGE_A = {
    "return_features": True,
    "is_stylespace": False,
    "new_features": None,
    "early_stop": None,
}


class GeneratorUsageA(nn.Module):
    """
    Input: single latent w of shape [bs, 18, 512]
    """
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, w_latent):
        images, _ = self.generator(
            [w_latent],
            **PARAMS_USAGE_A
        )
        return images


ckpt = torch.load('pretrained_models/stylegan2-ffhq-config-f.pt', map_location='cpu')
pt_generator = Generator(size=1024, style_dim=512, n_mlp=8)
pt_generator.load_state_dict(ckpt["g_ema"], strict=False)
pt_generator.eval()


generator_usage_a = GeneratorUsageA(pt_generator).eval()

dummy_latent = torch.randn(1, 18, 512)

torch.onnx.export(
    generator_usage_a,
    dummy_latent,
    "generator_usage_a.onnx",
    input_names=["w_latent"],
    output_names=["output"],
    # dynamic_axes={
    #     "w_latent": {0: "batch_size"},
    #     "output": {0: "batch_size"}
    # },
    opset_version=10,
)

print("Exported generator_usage_a.onnx")


with torch.no_grad():
    pt_output_a, _ = pt_generator([dummy_latent], **PARAMS_USAGE_A)

pt_output_a = pt_output_a.numpy()
dummy_latent_np = dummy_latent.numpy()

ort_session_a = ort.InferenceSession("generator_usage_a.onnx", providers=["CPUExecutionProvider"])
w_latent_name_a = ort_session_a.get_inputs()[0].name

onnx_output_a = ort_session_a.run(None, {w_latent_name_a: dummy_latent_np})

print('ONNX A shape:', onnx_output_a[0].shape)
print('PyTorch A shape:', pt_output_a.shape)

print(np.linalg.norm(onnx_output_a[0]), np.linalg.norm(pt_output_a))
print(np.testing.assert_allclose(pt_output_a, onnx_output_a[0], rtol=1e-3, atol=1e-3))

# ONNX A shape: (1, 3, 1024, 1024)
# PyTorch A shape: (1, 3, 1024, 1024)
# 1506.7668 1506.7668
# None
