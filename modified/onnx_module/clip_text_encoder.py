import pickle
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn

import clip
from modified.onnx_module.utils import  export_and_validate
from modified.styleclip.styleclip_editor import TEMPLATES

DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/clip_text_encoder.onnx'
STYLECLIP_GLOBAL_DIR = (Path(__file__).parent.parent / 'styleclip').resolve()


class CLIPTextEncoder(nn.Module):

    def __init__(self, clip, delta_i_c):
        super().__init__()
        self.clip = clip
        self.delta_i_c = delta_i_c

    def forward(self, text_inputs, beta):
        text_embeddings = self.clip.encode_text(text_inputs)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        batch_size = text_embeddings.shape[0] // 2
        text_embedding1 = text_embeddings[:batch_size].mean(dim=0)
        text_embedding1 /= text_embedding1.norm()
        text_embedding2 = text_embeddings[batch_size:].mean(dim=0)
        text_embedding2 /= text_embedding2.norm()
        delta_t = text_embedding1 - text_embedding2
        delta_i = delta_t / torch.norm(delta_t)
        r_c = torch.matmul(self.delta_i_c, delta_i)
        delta_s = r_c.clone()
        channels_to_zero = torch.abs(r_c) < beta
        delta_s[channels_to_zero] = 0
        max_channel_value = torch.abs(delta_s).max()
        delta_s /= max_channel_value
        return delta_s

def init_model():
    model, _ = clip.load('ViT-B/32', 'cpu')
    delta_i_c = torch.from_numpy(np.load(STYLECLIP_GLOBAL_DIR / 'delta_i_c.npy')).float()
    return CLIPTextEncoder(model, delta_i_c).eval()


def pt_output(dummy_input=None):
    torch_model = init_model()
    if dummy_input is None:
        dummy_input = (
            torch.cat(
                [clip.tokenize(t.format('photo')) for t in TEMPLATES.split('\n')]
                +
                [clip.tokenize(t.format('picture')) for t in TEMPLATES.split('\n')]
            ).to('cpu'),
            torch.tensor([0.1])
        )
    with torch.no_grad():
        out = torch_model(dummy_input)
    return out


if __name__ == "__main__":
    torch_model = init_model()
    dummy_input = torch.cat(
        [clip.tokenize(t.format('photo')) for t in TEMPLATES.split('\n')]
        +
        [clip.tokenize(t.format('picture')) for t in TEMPLATES.split('\n')]
    ).to('cpu')
    beta = torch.tensor([0.1])
    print(dummy_input.shape)

    output_names = ['delta_s']

    export_and_validate(
        model=torch_model,
        dummy_input=(dummy_input, beta),
        output_onnx_path=MODEL_PATH,
        output_names=output_names,
        skip_export=False,
        dynamo=False,
        opset_version=16
    )
