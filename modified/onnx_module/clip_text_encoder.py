import pickle
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn

import clip
from modified.onnx_module.utils import  export_and_validate
from modified.styleclip.styleclip_editor import TEMPLATES, STYLESPACE_DIMENSIONS, STYLESPACE_INDICES_WITHOUT_TORGB

DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/clip_text_encoder.onnx'
STYLECLIP_GLOBAL_DIR = (Path(__file__).parent.parent / 'styleclip').resolve()


class FeaturesChannelsToS(nn.Module):
    def __init__(self, s_std: torch.Tensor):
        """
        s_std should be a 1D torch.Tensor of length len(STYLESPACE_DIMENSIONS).
        """
        super().__init__()
        # Register s_std as a buffer so it's saved inside the model:
        self.register_buffer("s_std", s_std)

    def forward(self, s_without_torgb: torch.Tensor):
        result_list = []
        start_index_features = 0
        for i, dim in enumerate(STYLESPACE_DIMENSIONS):
            # If i is in the "no-toRGB" indices, pick from the flattened array
            if i in STYLESPACE_INDICES_WITHOUT_TORGB:
                end_index_features = start_index_features + dim
                # slice shape: [B, dim]
                slice_ = s_without_torgb[:, start_index_features:end_index_features]
                # scale by s_std[i]
                # s_std[i] is a scalar, so we do broadcast
                scaled = slice_ * self.s_std[i]
                start_index_features = end_index_features
            else:
                # produce zeros of shape [B, dim]
                scaled = torch.zeros(1, dim, device='cpu')
            result_list.append(scaled)

        return tuple(result_list)


class CLIPTextEncoder(nn.Module):

    def __init__(self, clip, delta_i_c, s_std):
        super().__init__()
        self.clip = clip
        self.delta_i_c = delta_i_c
        self.feats_to_s = FeaturesChannelsToS(s_std=s_std)

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
        output_s = self.feats_to_s(delta_s.unsqueeze(0))
        return output_s

def init_model():
    model, _ = clip.load('ViT-B/32', 'cpu')
    delta_i_c = torch.from_numpy(np.load(STYLECLIP_GLOBAL_DIR / 'delta_i_c.npy')).float()
    with open(STYLECLIP_GLOBAL_DIR / 'S_mean_std', 'rb') as f:
        _, s_std = pickle.load(f)
    s_std = np.concatenate(s_std).astype(np.float32)
    s_std_torch = torch.from_numpy(s_std)
    return CLIPTextEncoder(model, delta_i_c, s_std_torch).eval()


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

    output_names = [
        'o1',
        'o2',
        'o3',
        'o4',
        'o5',
        'o6',
        'o7',
        'o8',
        'o9',
        'o10',
        'o11',
        'o12',
        'o13',
        'o14',
        'o15',
        'o16',
        'o17',
        'o18',
        'o19',
        'o20',
        'o21',
        'o22',
        'o23',
        'o24',
        'o25',
        'o26',
    ]

    export_and_validate(
        model=torch_model,
        dummy_input=(dummy_input, beta),
        output_onnx_path=MODEL_PATH,
        output_names=output_names,
        skip_export=False,
        dynamo=False,
        opset_version=16
    )
