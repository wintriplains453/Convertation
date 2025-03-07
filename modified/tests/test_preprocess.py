import os
import glob
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import pytest

from modified.preprocess import preprocess_image

def get_torch_output(image_path, resize_size=1024, mean=0.5, std=0.5):
    """
    Uses torchvision to process the image similarly to your custom pipeline:
      - Resizes the smaller edge to 'resize_size'.
      - Converts the image to a tensor ([0, 1], shape (C, H, W)).
      - Normalizes it using (x - mean)/std.
      - Unsqueezes to add a batch dimension: (1, C, H, W).
    """
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize(resize_size),
        T.ToTensor(),
        T.Normalize([mean] * 3, [std] * 3)
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor

def get_real_image_paths():
    """
    Searches for image files in the tests/data/ directory with common image extensions.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for pattern in patterns:
        image_files.extend(glob.glob(os.path.join(data_dir, pattern)))
    return image_files

real_image_paths = get_real_image_paths()
if not real_image_paths:
    pytest.skip("No real images found in tests/data", allow_module_level=True)

@pytest.mark.parametrize("real_image_path", real_image_paths)
def test_preprocess_real_images(real_image_path):
    """
    Processes each real image from tests/data/ using both your custom pipeline and
    torchvision's transforms, then compares their outputs.
    """
    custom_output = preprocess_image(real_image_path, resize_size=1024, mean=0.5, std=0.5)
    torch_output = get_torch_output(real_image_path, resize_size=1024, mean=0.5, std=0.5).numpy()

    # Check that the shapes match.
    assert custom_output.shape == torch_output.shape, (
        f"Shapes differ for image {real_image_path}: {custom_output.shape} vs {torch_output.shape}"
    )

    # Check that pixel values are close within a tolerance.
    np.testing.assert_allclose(custom_output, torch_output, atol=1e-5)
