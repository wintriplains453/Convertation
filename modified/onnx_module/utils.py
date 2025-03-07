from pathlib import Path
from argparse import Namespace
from typing import Optional, Union

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


MODELS_PATH = (Path(__file__).parent.parent.parent / 'pretrained_models').resolve()
ONNX_MODELS_PATH = (Path(__file__).parent / 'onnx_models').resolve()


opts = Namespace(
    device='cpu',
    checkpoint_path=str(MODELS_PATH / 'sfe_editor_light.pt'),
    stylegan_size=1024,
    arcface_model_path=str(MODELS_PATH / 'iresnet50-7f187506.pth'),
    stylegan_weights=str(MODELS_PATH / 'stylegan2-ffhq-config-f.pt'),
    e4e_path=str(MODELS_PATH / 'e4e_ffhq_encode.pt'),
)


def export_to_onnx(
    model: nn.Module,
    dummy_input: tuple[torch.Tensor, ...],
    output_onnx_path: Union[str, Path],
    output_names: list[str],
    input_names: Optional[list[str]] = None,
    opset_version: int = 11,
    verbose: bool = False
):
    """
    Exports a given PyTorch model to an ONNX file.
    """
    torch.onnx.export(
        model,
        dummy_input,
        f=str(output_onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        verbose=verbose
    )

def run_onnx(
    onnx_model_path: Union[str, Path],
    input_data: tuple[np.ndarray, ...]
) -> list[np.ndarray]:
    """
    Runs inference on the given ONNX model for one (or more) inputs.
    Returns a list of all outputs in NumPy format.
    """
    session = ort.InferenceSession(str(onnx_model_path))
    # Here we assume a single input; if multiple inputs are needed, adapt accordingly
    ort_inputs = {
        session.get_inputs()[i].name: i_data
        for i, i_data in enumerate(input_data)
    }
    ort_outputs = session.run(None, ort_inputs)
    return ort_outputs

def compare_tensors(
    pt_tensors: list[np.ndarray],
    onnx_tensors: list[np.ndarray],
    names: list[str],
    rtol=1e-3,
    atol=1e-5
):
    """
    Compares a list of PyTorch output tensors with ONNX output tensors.
    Prints max & mean differences and raises if they exceed tolerances.
    """
    for pt_arr, onnx_arr, name in zip(pt_tensors, onnx_tensors, names):
        diff = np.abs(pt_arr - onnx_arr)
        print(f"[{name}] Max difference:  {diff.max()}")
        print(f"[{name}] Mean difference: {diff.mean()}")
        np.testing.assert_allclose(pt_arr, onnx_arr, rtol=rtol, atol=atol)
        print(f"{name} outputs from PyTorch and ONNX Runtime are similar!")
        print("----------------------------------------------------")

def export_and_validate(
    model: nn.Module,
    dummy_input: tuple[torch.Tensor, ...],
    output_onnx_path: Union[str, Path],
    output_names: list[str],
    input_names: Optional[list[str]] = None,
    to_numpy_fn=lambda x: x.detach().cpu().numpy(),
    rtol=1e-3,
    atol=1e-5,
    skip_export=False,
    opset_version=11,
):
    """
    High-level pipeline to:
    1) Optionally export a PyTorch model to ONNX.
    2) Run model in PyTorch -> gather outputs in NumPy.
    3) Run model in ONNX Runtime -> gather outputs in NumPy.
    4) Compare the results.

    `skip_export=True` can be used if the ONNX model is already exported.
    """
    # 1) Export
    if not skip_export:
        export_to_onnx(
            model,
            dummy_input,
            output_onnx_path,
            output_names=output_names,
            input_names=input_names,
            opset_version=opset_version,
        )

    # 2) PyTorch inference
    with torch.no_grad():
        pt_outputs = model(*dummy_input)
        # If model outputs a single tensor, wrap it in a tuple for uniformity
        if not isinstance(pt_outputs, (tuple, list)):
            pt_outputs = [pt_outputs]
        pt_outputs_np = [to_numpy_fn(o) for o in pt_outputs]

    # 3) ONNX inference
    onnx_outputs = run_onnx(output_onnx_path, tuple(i.cpu().numpy() for i in dummy_input))

    # 4) Compare
    compare_tensors(pt_outputs_np, onnx_outputs, output_names, rtol, atol)
