import os
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from modified.preprocess import preprocess_image
from modified.fse_full import interpolate_onnx

@pytest.fixture
def temp_image_1024(tmp_path):
    """
    Creates a random 1024x1024 image. We'll pass it through preprocess_image,
    which should output (1,3,1024,1024) in [-1, 1].
    """
    file_path = tmp_path / "random_1024.jpg"
    random_pixels = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    image = Image.fromarray(random_pixels, 'RGB')
    image.save(str(file_path))
    return str(file_path)

def test_interpolate_onnx(temp_image_1024):
    """
    Compare F.interpolate(..., size=(256,256), mode='bilinear', align_corners=False)
    with interpolate_onnx(...).
    """
    # 1) Use your existing pipeline to get (1,3,1024,1024) in [-1,1]
    #    Make sure your pipeline indeed forces a 1024x1024 output.
    input_np = preprocess_image(temp_image_1024)
    assert input_np.shape == (1, 3, 1024, 1024), (
        f"Expected shape (1,3,1024,1024) from preprocess_image, got {input_np.shape}"
    )

    # 2) Torch interpolation
    torch_input = torch.from_numpy(input_np)  # shape: (1,3,1024,1024)
    torch_interpolated = F.interpolate(
        torch_input,
        size=(256, 256),
        mode='bilinear',
        align_corners=False
    )  # shape: (1,3,256,256)
    torch_out = torch_interpolated.numpy()

    # 3) NumPy interpolation
    numpy_out = interpolate_onnx(input_np)  # shape: (1,3,256,256)

    # 4) Compare shapes
    assert numpy_out.shape == torch_out.shape, (
        f"Shape mismatch: {numpy_out.shape} vs {torch_out.shape}"
    )

    # 5) Compare pixel values within a tolerance
    #    Expect minor differences due to Pillow vs. PyTorch bilinear logic.
    np.testing.assert_allclose(numpy_out, torch_out, atol=1e-2)


    # Additional cleanup or prints as desired
    print(f"PyTorch interpolation: mean={torch_out.mean()}, std={torch_out.std()}")
    print(f"NumPy interpolation:   mean={numpy_out.mean()}, std={numpy_out.std()}")
