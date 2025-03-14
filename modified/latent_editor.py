import numpy as np

from modified.onnx_module.editor_interfacegan import interfacegan_directions_onnx
from modified.onnx_module.utils import ONNX_MODELS_PATH
from modified.onnx_module.utils import run_onnx
from modified.styleclip.styleclip_editor import get_styleclip_global_edits


def get_edited_latent(original_latent: np.ndarray, editing_name: str, editing_degree: float) -> np.ndarray:

    if editing_name in interfacegan_directions_onnx:
        editing_degree = np.array([editing_degree], dtype=np.float32)
        edited_latents = run_onnx(
            interfacegan_directions_onnx[editing_name],
            (original_latent, editing_degree)
        )
        edited_latents = edited_latents[0]
    elif editing_name.startswith('styleclip_global_'):
        stylespace_latent = run_onnx(ONNX_MODELS_PATH / 'decoder_stylespace.onnx', (original_latent,))
        edited_latents = get_styleclip_global_edits(stylespace_latent, editing_degree, editing_name)
    else:
        raise ValueError(f'Edit name {editing_name} is not available')
    return edited_latents
