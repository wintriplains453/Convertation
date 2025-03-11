import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from modified.preprocess import preprocess_image
from modified import fse_inference_runner



def prepare_np(x):
    out = np.transpose(x[0], (1, 2, 0))
    out = (out + 1) / 2
    out[out < 0] = 0
    out[out > 1] = 1
    return out

image_pth = str(Path(__file__).parent / 'tests/data/smith_aligned.jpg')

def edit(
    orig_img_pth: str,
    editing_name: str,
    edited_power: float,
    save_pth: str,
    align: bool = False,
):
    save_pth = Path(save_pth)
    aligned_image_pth = orig_img_pth

    # Если захочется align вставить, то для этого заглушка:
    if align:
        aligned_image_pth = ...
        raise NotImplementedError

    start = time.time()
    orig_img = preprocess_image(aligned_image_pth)
    print('preprocess_image:', time.time() - start, 's')

    # start = time.time()
    # inv_images, inversion_results = fse_inference_runner.run_on_batch(orig_img)
    # edited_image = fse_inference_runner.run_editing_on_batch(
    #     method_res_batch=inversion_results,
    #     editing_name=editing_name,
    #     editing_degree=edited_power,
    # )
    # print('separate onnx pre editing files:', time.time() - start, 's')  # 10.74s

    start = time.time()
    image, w_recon, w_e4e, fused_feat = fse_inference_runner.run_pre_editor(orig_img)
    edited_image = fse_inference_runner.run_editing_core(
        latent=w_recon,
        w_e4e=w_e4e,
        fused_feat=fused_feat,
        editing_name=editing_name,
        editing_degree=edited_power,
    )
    print('combined onnx pre editing files:', time.time() - start, 's') # 8.62

    return edited_image


output = edit(
    orig_img_pth=image_pth,
    editing_name='styleclip_global_face with hair_face with fire hair_0.1',
    edited_power=2,
    save_pth='',
    align=False
)
# print('output type:', type(output), 'output shape:', output.shape)
# plt.imshow(prepare_np(output))
# plt.savefig(str(Path(__file__).parent /'np_run_editing_on_batch.png'))
