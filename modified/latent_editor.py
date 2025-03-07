import numpy as np

from modified.onnx_module.editor_interfacegan import interfacegan_directions_onnx
from modified.onnx_module.utils import run_onnx



def get_edited_latent(original_latent: np.ndarray, editing_name: str, editing_degree: float):
    editing_degree = np.array([editing_degree], dtype=np.float32)

    if editing_name in interfacegan_directions_onnx:
        edited_latents = run_onnx(
            interfacegan_directions_onnx[editing_name],
            (original_latent, editing_degree)
        )
        edited_latents = edited_latents[0]
    else:
        raise ValueError(f'Edit name {editing_name} is not available')
    return edited_latents
