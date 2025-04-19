from pathlib import Path
from argparse import Namespace
from typing import Optional, Union, List, Tuple
import logging

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_PATH = (Path(__file__).parent.parent.parent / 'pretrained_models').resolve()
EDITING_DIRECTIONS_PATH = (Path(__file__).parent.parent.parent / 'editings').resolve()
ONNX_MODELS_PATH = (Path(__file__).parent / 'onnx_models').resolve()

opts = Namespace(
    device='cpu',
    checkpoint_path=str(MODELS_PATH / 'sfe_editor_light.pt'),
    stylegan_size=1024,
    arcface_model_path=str(MODELS_PATH / 'iresnet50-7f187506.pth'),
    stylegan_weights=str(MODELS_PATH / 'stylegan2-ffhq-config-f.pt'),
    e4e_path=str(MODELS_PATH / 'e4e_ffhq_encode.pt'),
)

def get_onnx_providers() -> List[str]:
    """Возвращает доступные провайдеры в правильном порядке приоритета"""
    available_providers = ort.get_available_providers()
    preferred_order = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return [p for p in preferred_order if p in available_providers]

def export_to_onnx(
    model: nn.Module,
    dummy_input: Tuple[torch.Tensor, ...],
    output_onnx_path: Union[str, Path],
    output_names: List[str],
    input_names: Optional[List[str]] = None,
    opset_version: int = 11,
    dynamo: bool = False,
    verbose: bool = False,
    do_constant_folding: bool = True,
) -> None:
    """
    Exports a given PyTorch model to an ONNX file.
    """
    logger.info(f"Exporting model to ONNX format: {output_onnx_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        f=str(output_onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            name: {0: 'batch_size'} 
            for name in (input_names or [])
        },
        verbose=verbose
    )
    
    if dynamo:
        import onnx
        model_onnx = onnx.load(output_onnx_path)
        onnx.checker.check_model(model_onnx)
        onnx.save(model_onnx, output_onnx_path)

def run_onnx(
    onnx_model_path: Union[str, Path],
    input_data: Tuple[np.ndarray, ...],
    verbose: bool = False
) -> List[np.ndarray]:
    """
    Runs inference on the given ONNX model with automatic type conversion.
    Returns a list of all outputs in NumPy format.
    
    Args:
        onnx_model_path: Path to ONNX model
        input_data: Tuple of input numpy arrays
        verbose: Whether to print debug information
        
    Returns:
        List of output numpy arrays
    """
    # 1. Initialize session
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.log_severity_level = 3 if not verbose else 1
    print(onnx_model_path);
    
    session = ort.InferenceSession(
        str(onnx_model_path),
        sess_options=session_options,
        providers=get_onnx_providers()
    )
    
    if verbose:
        logger.info(f"Running ONNX model: {onnx_model_path}")
        print_model_io_info(session)

    # 2. Prepare inputs with type checking
    ort_inputs = {}
    for i, input_info in enumerate(session.get_inputs()):
        input_array = input_data[i]
        
        # Convert input type if needed
        target_dtype = get_numpy_type_from_onnx(input_info.type)
        if input_array.dtype != target_dtype:
            if verbose:
                logger.warning(
                    f"Converting input '{input_info.name}' from {input_array.dtype} "
                    f"to {target_dtype} (required by model)"
                )
            input_array = input_array.astype(target_dtype)
        
        ort_inputs[input_info.name] = input_array

    # 3. Run inference
    ort_outputs = session.run(None, ort_inputs)
    
    if verbose:
        logger.info("Inference completed successfully")
        
    return ort_outputs


def get_numpy_type_from_onnx(onnx_type: str) -> np.dtype:
    """Maps ONNX types to numpy types"""
    type_mapping = {
        'tensor(float)': np.float32,
        'tensor(float16)': np.float16,
        'tensor(int64)': np.int64,
        'tensor(int32)': np.int32,
        'tensor(bool)': np.bool_
    }
    return type_mapping.get(onnx_type, np.float32)  # Default to float32


def print_model_io_info(session: ort.InferenceSession):
    """Prints model input/output information for debugging"""
    print("\nModel I/O Info:")
    print("Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type} (shape: {inp.shape})")
    
    print("\nOutputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type} (shape: {out.shape})")
    print()

def compare_tensors(
    pt_tensors: List[np.ndarray],
    onnx_tensors: List[np.ndarray],
    names: List[str],
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> None:
    """
    Compares PyTorch and ONNX Runtime outputs.
    """
    logger.info("Comparing PyTorch and ONNX outputs")
    
    all_ok = True
    for pt_arr, onnx_arr, name in zip(pt_tensors, onnx_tensors, names):
        diff = np.abs(pt_arr - onnx_arr)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        logger.info(f"[{name}] Max difference: {max_diff:.2e}")
        logger.info(f"[{name}] Mean difference: {mean_diff:.2e}")
        
        if max_diff > atol or np.any(np.isnan(diff)):
            logger.error(f"Large difference detected in {name}!")
            all_ok = False
        
        try:
            np.testing.assert_allclose(
                pt_arr, onnx_arr, 
                rtol=rtol, 
                atol=atol,
                err_msg=f"Output {name} differs too much"
            )
        except AssertionError as e:
            logger.error(str(e))
            all_ok = False
    
    if all_ok:
        logger.info("All outputs match within tolerances!")
    else:
        logger.warning("Some outputs differ beyond tolerances")

def export_and_validate(
    model: nn.Module,
    dummy_input: Tuple[torch.Tensor, ...],
    output_onnx_path: Union[str, Path],
    output_names: List[str],
    input_names: Optional[List[str]] = None,
    to_numpy_fn=lambda x: x.detach().cpu().numpy(),
    rtol: float = 1e-3,
    atol: float = 1e-5,
    skip_export: bool = False,
    dynamo: bool = False,
    opset_version: int = 11,
    do_constant_folding: bool = True,
) -> None:
    """
    Full pipeline for exporting and validating ONNX models.
    """
    # 1) Export
    if not skip_export:
        export_to_onnx(
            model=model,
            dummy_input=dummy_input,
            output_onnx_path=output_onnx_path,
            output_names=output_names,
            input_names=input_names,
            dynamo=dynamo,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
        )

    # 2) PyTorch inference
    with torch.no_grad():
        logger.info("Running PyTorch inference")
        pt_outputs = model(*dummy_input)
        pt_outputs = [pt_outputs] if not isinstance(pt_outputs, (tuple, list)) else pt_outputs
        pt_outputs_np = [to_numpy_fn(o) for o in pt_outputs]

    # 3) ONNX inference
    logger.info("Running ONNX inference")
    onnx_outputs = run_onnx(
        output_onnx_path,
        tuple(i.cpu().numpy() for i in dummy_input)
    )

    # 4) Compare results
    compare_tensors(
        pt_tensors=pt_outputs_np,
        onnx_tensors=onnx_outputs,
        names=output_names,
        rtol=rtol,
        atol=atol
    )