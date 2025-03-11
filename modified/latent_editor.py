import numpy as np

from modified.onnx_module.editor_interfacegan import interfacegan_directions_onnx
from modified.onnx_module.utils import ONNX_MODELS_PATH
from modified.onnx_module.utils import run_onnx



def get_edited_latent(original_latent: np.ndarray, editing_name: str, editing_degree: float) -> np.ndarray:
    editing_degree = np.array([editing_degree], dtype=np.float32)

    if editing_name in interfacegan_directions_onnx:
        edited_latents = run_onnx(
            interfacegan_directions_onnx[editing_name],
            (original_latent, editing_degree)
        )
        edited_latents = edited_latents[0]
    elif editing_name.startswith('styleclip_global_'):
        direction = editing_name.replace('styleclip_global_', '')
        stylespace_latent = run_onnx(ONNX_MODELS_PATH / 'decoder_stylespace.onnx', (original_latent,))
        stylespace_latent = (list(stylespace_latent[:17]), list(stylespace_latent[17:]))
        print('stylespace_latent[0]:', [i.shape[1] for i in stylespace_latent[0]])
        print('stylespace_latent[1]:', [i.shape[1] for i in stylespace_latent[1]])
        edited_latents = None
    else:
        raise ValueError(f'Edit name {editing_name} is not available')
    return edited_latents
