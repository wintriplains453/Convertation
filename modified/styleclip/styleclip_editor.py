import numpy as np

from modified.styleclip.simple_tokenizer import tokenize
from modified.onnx_module.utils import run_onnx
from modified.onnx_module.utils import ONNX_MODELS_PATH


TEMPLATES = """a bad photo of a {}.
a photo of the hard to see {}.
a low resolution photo of the {}.
a bad photo of the {}.
a cropped photo of the {}.
a photo of a hard to see {}.
a bright photo of a {}.
a photo of a clean {}.
a photo of a dirty {}.
a dark photo of the {}.
a photo of my {}.
a photo of the cool {}.
a close-up photo of a {}.
a black and white photo of the {}.
a pixelated photo of the {}.
a bright photo of the {}.
a cropped photo of a {}.
a photo of the dirty {}.
a jpeg corrupted photo of a {}.
a blurry photo of the {}.
a photo of the {}.
a good photo of the {}.
a photo of one {}.
a close-up photo of the {}.
a photo of a {}.
a low resolution photo of a {}.
a photo of the clean {}.
a photo of a large {}.
a photo of a nice {}.
a photo of a weird {}.
a blurry photo of a {}.
a pixelated photo of a {}.
a jpeg corrupted photo of the {}.
a good photo of a {}.
a photo of the nice {}.
a photo of the small {}.
a photo of the weird {}.
a photo of the large {}.
a black and white photo of a {}.
a dark photo of a {}.
a photo of a cool {}.
a photo of a small {}."""

STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]

TORGB_INDICES = list(range(1, len(STYLESPACE_DIMENSIONS), 3))
STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in TORGB_INDICES][:11]


def get_styleclip_global_edits(
    start_s: list[np.ndarray],
    factor: float,
    editing_name: str,
):

    direction = editing_name.replace('styleclip_global_', '')
    neutral_text, target_text, disentangle_str = direction.split("_")
    disentanglement = float(disentangle_str)

    # Compute the StyleSpace direction (list of style arrays) for adjusting
    # from `neutral_text` to `target_text`, thresholded by `beta`.
    # s_std : list[np.ndarray] - standard deviations for each StyleSpace channel.

    text_prompts_templates = TEMPLATES.split('\n')
    all_sentences = (
        [t.format(target_text) for t in text_prompts_templates]
        +
        [t.format(neutral_text) for t in text_prompts_templates]
    )
    tokens = tokenize(all_sentences)

    directions = run_onnx(
        ONNX_MODELS_PATH / 'clip_text_encoder.onnx',
        (tokens, np.array([disentanglement], dtype=np.float32))
    )

    edits_ss = [directions[i] for i in range(len(directions)) if i not in TORGB_INDICES]
    edits_rgb = [directions[i] for i in range(len(directions)) if i in TORGB_INDICES]

    # Now apply the factor to the style-space directions, but
    # use a factor of 1.0 for the to-RGB directions, dividing each by 1.5
    # (as was done in your original code).
    edited_ss_list = []
    for orig, delta in zip(start_s[:17], edits_ss):
        # Broadcast the delta to match the shape of orig, then add scaled by (factor / 1.5)
        edit_broadcast = np.broadcast_to(delta, orig.shape)
        out = orig + (factor / 1.5) * edit_broadcast
        edited_ss_list.append(out)

    edited_rgb_list = []
    for orig, delta in zip(start_s[17:], edits_rgb):
        # Similarly, but use factor=1.0 for the to-RGB layers
        edit_broadcast = np.broadcast_to(delta, orig.shape)
        out = orig + (1.0 / 1.5) * edit_broadcast
        edited_rgb_list.append(out)
    return edited_ss_list, edited_rgb_list
