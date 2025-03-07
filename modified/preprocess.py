import numpy as np
from PIL import Image


def resize_image(image: Image, size: int):
    width, height = image.size
    if width < height:
        new_width = size
        new_height = int(height * size / width)
    else:
        new_height = size
        new_width = int(width * size / height)
    return image.resize((new_width, new_height), Image.BILINEAR)


def normalize_and_to_tensor(image: Image, mean: float = 0.5, std: float = 0.5):
    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = (img_np - mean) / std
    img_np = np.transpose(img_np, (2, 0, 1))
    return np.expand_dims(img_np, axis=0)


def preprocess_image(
        image_path: str,
        resize_size: int = 1024,
        mean: float = 0.5,
        std: float = 0.5
) -> np.ndarray:
    """
    Mimics:

    orig_img = Image.open(image_pth).convert("RGB")
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(1024),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5]*3, [0.5]*3),
        ]
    )
    orig_img = transforms(orig_img).unsqueeze(0)
    """
    image = Image.open(image_path).convert("RGB")
    resized_image = resize_image(image, resize_size)
    processed_tensor = normalize_and_to_tensor(resized_image, mean, std)
    return processed_tensor
