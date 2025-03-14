from pathlib import Path

import torch
import torch.nn as nn

import clip
from modified.onnx_module.utils import  export_and_validate
from modified.styleclip.styleclip_editor import TEMPLATES

DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/clip_text_encoder.onnx'


class CLIPTextEncoder(nn.Module):

    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, text_inputs):
        return self.clip.encode_text(text_inputs)

def init_model():
    model, _ = clip.load('ViT-B/32', 'cpu')
    return CLIPTextEncoder(model).eval()


def pt_output(dummy_input=None, context_length=77):
    torch_model = init_model()
    if dummy_input is None:
        dummy_input = torch.cat([clip.tokenize(f"a photo of a {c}") for c in TEMPLATES.split('\n')]).to('cpu')
    with torch.no_grad():
        out = torch_model(dummy_input)
    return out


if __name__ == "__main__":
    torch_model = init_model()
    dummy_input = torch.cat([clip.tokenize(f"a photo of a {c}") for c in TEMPLATES.split('\n')]).to('cpu')
    print(dummy_input.shape)
    output_names = ['text_emb']

    export_and_validate(
        model=torch_model,
        dummy_input=(dummy_input,),
        output_onnx_path=MODEL_PATH,
        output_names=output_names,
        skip_export=False,
        dynamo=False,
        opset_version=16
    )
