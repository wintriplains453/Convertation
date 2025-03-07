from pathlib import Path

import torch
import torch.nn as nn

from modified.onnx_module.utils import export_and_validate, EDITING_DIRECTIONS_PATH

OUTPUT_DIR_PATH = Path(__file__).parent.resolve() / 'onnx_models/editings'


interfacegan_directions_pt = {
    'age': str(EDITING_DIRECTIONS_PATH / 'interfacegan_directions/age.pt'),
    'smile': str(EDITING_DIRECTIONS_PATH / 'interfacegan_directions/smile.pt'),
    'rotation': str(EDITING_DIRECTIONS_PATH / 'interfacegan_directions/rotation.pt'),
}
interfacegan_directions_onnx = {
    editing_name: str(OUTPUT_DIR_PATH / f'interfacegan_{editing_name}.onnx')
    for editing_name in interfacegan_directions_pt
}

interfacegan_tensors = {
    name: torch.load(path, map_location='cpu')
    for name, path in interfacegan_directions_pt.items()
}



class InterfaceGanEdit(nn.Module):
    def __init__(self, direction_tensor: torch.Tensor):
        super().__init__()
        self.register_buffer('direction', direction_tensor)

    def forward(self, start_w: torch.Tensor, factor: torch.Tensor):
        # factor should be a scalar tensor or shape [N] that broadcasts properly
        # e.g., if factor is shape [1], the math works elementwise.
        edited_latent = start_w + 0.5 * factor * self.direction
        return edited_latent


if __name__ == '__main__':

    dummy_latent = torch.randn(1, 18, 512)
    dummy_factor = torch.tensor([5.0], dtype=dummy_latent.dtype)

    for editing_name, direction_tensor in interfacegan_tensors.items():
        print(f'Exporting direction: {editing_name}')
        model = InterfaceGanEdit(direction_tensor=direction_tensor).eval()
        onnx_path = interfacegan_directions_onnx[editing_name]

        export_and_validate(
            model=model,
            dummy_input=(dummy_latent, dummy_factor),
            output_onnx_path=onnx_path,
            output_names=[f'edited_{editing_name}'],
            input_names=['start_w', 'factor'],
            opset_version=11,
            skip_export=False,
        )
