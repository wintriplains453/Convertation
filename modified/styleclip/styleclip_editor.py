import pickle
from pathlib import Path

import numpy as np

from modified.styleclip.simple_tokenizer import tokenize
from modified.onnx_module.utils import run_onnx
from modified.onnx_module.utils import ONNX_MODELS_PATH


# TODO move to onnx
STYLECLIP_GLOBAL_DIR = Path(__file__).parent.resolve()

TEMPLATES = """a bad photo of a {}.
a sculpture of a {}.
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
a sketch of the {}.
a pixelated photo of a {}.
a jpeg corrupted photo of the {}.
a good photo of a {}.
a photo of the nice {}.
a photo of the small {}.
a photo of the weird {}.
a drawing of the {}.
a photo of the large {}.
a black and white photo of a {}.
a dark photo of a {}.
a photo of a cool {}.
a photo of a small {}."""

STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]

TORGB_INDICES = list(range(1, len(STYLESPACE_DIMENSIONS), 3))
STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in TORGB_INDICES][:11]


def features_channels_to_s(s_without_torgb: np.ndarray, s_std: list[np.ndarray]):
    """
    Converts a flattened per-channel style delta (s_without_torgb) into a
    list of NumPy arrays with shape (1, 1, channel_dim, 1, 1), based on
    STYLESPACE_DIMENSIONS and the standard deviation array s_std.

    Parameters
    ----------
    s_without_torgb : np.ndarray
        1D NumPy array containing style deltas for the channels that
        are not part of ToRGB layers.

    s_std : np.ndarray
        1D NumPy array of standard deviations for each StyleSpace index
        (same length as STYLESPACE_DIMENSIONS).

    Returns
    -------
    list of np.ndarray
    """
    s = []
    start_index_features = 0
    for i, dim in enumerate(STYLESPACE_DIMENSIONS):
        if i in STYLESPACE_INDICES_WITHOUT_TORGB:
            end_index_features = start_index_features + dim
            # Scale by the standard deviations
            scaled = s_without_torgb[start_index_features:end_index_features] * s_std[i]
            start_index_features = end_index_features
        else:
            scaled = np.zeros(dim, dtype=s_without_torgb.dtype)
        # Reshape to (1, 1, channel_dim, 1, 1)
        # scaled = scaled.reshape((1, 1, -1, 1, 1))
        s.append(scaled)
    return s

def load_styleclip_global(clip_onnx_path) -> 'StyleCLIPGlobalDirection':
    delta_i_c = np.load(STYLECLIP_GLOBAL_DIR / 'delta_i_c.npy')  # delta_i_c shape: (6048, 512)
    with open(STYLECLIP_GLOBAL_DIR / 'S_mean_std', 'rb') as f:
        _, s_std = pickle.load(f)
    s_std = [x for x in s_std]
    # s_std len: 26
    # s_std shapes: [(512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (512,), (256,), (256,), (256,), (128,), (128,), (128,), (64,), (64,), (64,), (32,), (32,)]
    text_prompt_templates = TEMPLATES.split('\n')
    return StyleCLIPGlobalDirection(delta_i_c, s_std, text_prompt_templates, clip_onnx_path)


class StyleCLIPGlobalDirection:
    def __init__(self, delta_i_c: np.ndarray, s_std: list[np.ndarray], text_prompts_templates, clip_onnx_path):
        """
        Parameters
        ----------
        delta_i_c : np.ndarray
            A 2D NumPy array (matrix) used for projecting the text direction into StyleSpace.

        s_std : list[np.ndarray]
            List of NumPy arrays with standard deviations for each StyleSpace channel.

        text_prompts_templates : list of str
            List of prompt templates, e.g. ["a photo of a {}", "an image of a {}", ...]

        clip_onnx_path: str or Path
            Path to CLIP encoder

        """
        self.delta_i_c = delta_i_c  # shape [num_channels, embed_dim] or similar
        self.s_std = s_std          # shape depending on your StyleSpace
        self.text_prompts_templates = text_prompts_templates
        self.clip_onnx_path = clip_onnx_path

    def get_delta_s(self, neutral_text: str, target_text: str, beta: float):
        """
        Compute the StyleSpace direction (list of style arrays) for adjusting
        from `neutral_text` to `target_text`, thresholded by `beta`.
        """
        # 1. Get the text direction (delta_i)
        #    text_prompts: (target_text, neutral_text)
        delta_i = self.get_delta_i((target_text, neutral_text)).astype(np.float32)

        # 2. Project this direction into StyleSpace
        r_c = np.matmul(self.delta_i_c, delta_i)  # shape [num_channels]

        # 3. Threshold small channels
        delta_s = r_c.copy()
        mask = np.abs(r_c) < beta
        delta_s[mask] = 0.0

        # 4. Normalize if not zero
        max_val = np.max(np.abs(delta_s))
        if max_val > 0:
            delta_s /= max_val

        # 5. Convert flattened channel deltas into StyleSpace list
        return features_channels_to_s(delta_s, self.s_std)

    def get_delta_i(self, text_prompts: tuple[str, str]) -> np.ndarray:
        """
        text_prompts: (target_text, neutral_text)
        Returns the normalized difference of their averaged text embeddings (delta_t).
        """
        # 1. Extract text embeddings: (target_text, neutral_text)
        text_features = self._get_averaged_text_features(text_prompts)

        # 2. delta_t = T_target - T_neutral
        delta_t = text_features[0] - text_features[1]

        # 3. Normalize
        norm_delta_t = np.linalg.norm(delta_t)
        if norm_delta_t > 0:
            delta_i = delta_t / norm_delta_t
        else:
            # Edge case: if vector is zero, return it as-is
            delta_i = delta_t

        return delta_i

    def _get_averaged_text_features(self, text_prompts: tuple[str, str]) -> np.ndarray:
        """
        Return a 2D array: shape = [2, embed_dim]
        Each row is the averaged text embedding for one prompt.
        text_prompts has length 2: [target_text, neutral_text].
        """
        all_sentences = (
            [t.format(text_prompts[0]) for t in self.text_prompts_templates]
            +
            [t.format(text_prompts[1]) for t in self.text_prompts_templates]
        )
        tokens = tokenize(all_sentences)
        text_emb = run_onnx(self.clip_onnx_path, (tokens,))
        return text_emb[0]


def get_styleclip_global_edits(
    start_s: list[np.ndarray],
    factor: float,
    editing_name: str,
):

    editor = load_styleclip_global(ONNX_MODELS_PATH / 'clip_text_encoder.onnx')

    direction = editing_name.replace('styleclip_global_', '')
    neutral_text, target_text, disentangle_str = direction.split("_")
    disentanglement = float(disentangle_str)

    directions = editor.get_delta_s(neutral_text, target_text, disentanglement)

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
