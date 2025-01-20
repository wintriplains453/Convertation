import torch
import onnxscript
from models.psp.stylegan2.model import Generator

generator = Generator(size=1024, style_dim=512, n_mlp=8)
generator.eval()

dummy_input = torch.randn(1, 1, 512)

export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
onnx_program = torch.onnx.export(generator, dummy_input, "generator.onnx", export_params=True, opset_version=11)
print(f"Модель успешно экспортирована")
