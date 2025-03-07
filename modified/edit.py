from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from modified.preprocess import preprocess_image
from modified import fse_inference_runner



def prepare_np(x):
    return (np.transpose(x[0], (1, 2, 0)) + 1) / 2

image_pth = str(Path(__file__).parent / 'tests/data/smith_aligned.jpg')

output = fse_inference_runner.run_on_batch(preprocess_image(image_pth))
# plt.imshow(prepare_np(output[0]))
# plt.savefig('np_run_on_batch.png')



# from PIL import Image
# import torchvision
# import torch.nn.functional as F
# from modified.onnx_module.inverter import init_model as inverter_init_model
#
# orig_img = Image.open(image_pth).convert("RGB")
# transforms = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.Resize(1024),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3),
#     ]
# )
# orig_img = transforms(orig_img).unsqueeze(0)
# torch_interpolated = F.interpolate(
#     orig_img,
#     size=(256, 256),
#     mode='bilinear',
#     align_corners=False
# )
# inverter_init_model(torch_interpolated)
#
# plt.imshow(prepare_np(torch_interpolated.numpy()))
# plt.savefig('pt.png')
