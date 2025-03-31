from pathlib import Path

import torch
import torch.nn as nn

from modified.onnx_module.utils import export_and_validate
from modified.onnx_module import (interpolate, inverter, decoder_without_new_feature, fuser, encoder,
                                  decoder_with_new_feature, e4e_encoder)

DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/pre_editor.onnx'


class PreEditor(nn.Module):
    def __init__(self):
        super().__init__()
        self.interpolate = interpolate.init_model()
        self.inverter = inverter.init_model()
        self.decoder_without_new_feature = decoder_without_new_feature.init_model()
        self.fuser = fuser.init_model()
        self.encoder = encoder.init_model()
        self.decoder_with_new_feature = decoder_with_new_feature.init_model()
        self.e4e_encoder = e4e_encoder.init_model()

    def forward(self, x):
        x = self.interpolate(x)
        w_recon, predicted_feat = self.inverter(x)
        print('w_recon', w_recon.shape)
        _, w_feat = self.decoder_without_new_feature(w_recon)
        print('w_feat', w_feat.shape)
        fused_feat = self.fuser(torch.cat([predicted_feat, w_feat], dim=1))
        delta = torch.zeros_like(fused_feat)
        print('delta', delta.shape)
        print('fused_feat', fused_feat.shape)
        edited_feat = self.encoder(torch.cat([fused_feat, delta], dim=1))
        print('edited_feat', edited_feat.shape)
        image = self.decoder_with_new_feature(w_recon, edited_feat)
        w_e4e = self.e4e_encoder(x)
        return image, w_recon, w_e4e, fused_feat


def init_model():
    return PreEditor()


def pt_output():
    torch_model = PreEditor()
    dummy_input = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        image, w_recon, w_e4e, fused_feat = torch_model(dummy_input)
    return image, w_recon, w_e4e, fused_feat


if __name__ == "__main__":
    torch_model = init_model()
    dummy_input = torch.randn(1, 3, 1024, 1024)

    output_names = ['image', 'w_recon', 'w_e4e', 'fused_feat']

    export_and_validate(
        model=torch_model,
        dummy_input=(dummy_input,),
        output_onnx_path=MODEL_PATH,
        output_names=output_names,
        skip_export=False,
        dynamo=False,
        opset_version=11,
        do_constant_folding=True,
    )

# AssertionError:
# Not equal to tolerance rtol=0.001, atol=1e-05
#
# Mismatched elements: 4 / 3145728 (0.000127%)
# Max absolute difference: 2.0086765e-05
# Max relative difference: 6.1578946
#  x: array([[[[ 0.22233 ,  0.111701, -0.033018, ...,  0.050313,  0.035754,
#            0.080918],
#          [ 0.139929, -0.006132, -0.010355, ...,  0.02712 ,  0.02239 ,...
#  y: array([[[[ 0.22233 ,  0.111702, -0.033018, ...,  0.050313,  0.035754,
#            0.080918],
#          [ 0.13993 , -0.006132, -0.010354, ...,  0.02712 ,  0.02239 ,...
