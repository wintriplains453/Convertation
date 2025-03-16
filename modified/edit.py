import time
from pathlib import Path

import numpy as np
from PIL import Image

from modified.preprocess import preprocess_image
from modified import fse_inference_runner

start = time.time()


def prepare_np(x):
    out = np.transpose(x[0], (1, 2, 0))
    out = (out + 1) / 2
    out[out < 0] = 0
    out[out > 1] = 1
    out = out * 255
    return Image.fromarray(out.astype("uint8"))


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

    edited_image = prepare_np(edited_image)
    edited_image.save(save_pth)

    return edited_image


if __name__ == '__main__':
    image_pth = str(Path(__file__).parent.parent / 'editing_res/scarlet/scarlet_aligned.jpg')
    output = edit(
        orig_img_pth=image_pth,
        editing_name='styleclip_global_face with hair_face with black hair_0.2',
        edited_power=3,
        save_pth=str(Path(__file__).parent / 'styleclip.png'),
        align=False
    )
print('Total:', time.time() - start, 's')