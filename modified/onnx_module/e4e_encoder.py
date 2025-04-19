from pathlib import Path
import torch
import torch.nn as nn
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, dynamic
from onnxruntime.tools import optimize_onnx_model
from onnxconverter_common.float16 import convert_float_to_float16
import onnx
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models.psp.encoders import psp_encoders
from utils.model_utils import toogle_grad
from utils.common_utils import get_keys

from modified.onnx_module import interpolate
from modified.onnx_module.utils import opts, export_and_validate

DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/e4e_encoder.onnx'

# Определение класса для калибровочного набора данных
class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        self.current_index = 0

    def get_next(self):
        if self.current_index >= len(self.image_paths):
            return None
        image_path = self.image_paths[self.current_index]
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).numpy()
        image_np = image_tensor.astype(np.float32)  # Конвертируем в float16
        self.current_index += 1
        return {'input.1': image_np}

class E4EEncoderLatentAverage(nn.Module):
    def __init__(self, e4e_encoder, latent_avg):
        super().__init__()
        self.e4e_encoder = e4e_encoder
        self.latent_avg = latent_avg

    def forward(self, x):
        w_e4e = self.e4e_encoder(x)
        w_e4e = w_e4e + self.latent_avg
        return w_e4e

def init_model():
    e4e_encoder = psp_encoders.Encoder4Editing(50, 'ir_se', opts)
    ckpt = torch.load(opts.e4e_path, map_location='cpu')
    e4e_encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
    e4e_encoder = e4e_encoder.eval()
    toogle_grad(e4e_encoder, False)

    ckpt_stylegan = torch.load(opts.stylegan_weights, map_location='cpu')
    latent_avg = ckpt_stylegan['latent_avg']

    return E4EEncoderLatentAverage(e4e_encoder=e4e_encoder, latent_avg=latent_avg).eval()

def pt_output(dummy_input=None):
    torch_model = init_model()
    if dummy_input is None:
        dummy_input = interpolate.pt_output()
    with torch.no_grad():
        w_e4e = torch_model(dummy_input)
    return w_e4e

def compress_onnx_model(input_model_path, output_model_path, quantize=True, fp16=False):
    """Оптимизирует и квантует ONNX-модель с использованием статического квантования"""
    
    # 1. Оптимизация графа
    optimized_path = str(input_model_path).replace('.onnx', '_opt.onnx')
    model = onnx.load(str(input_model_path))
    optimized_model = onnx.shape_inference.infer_shapes(model)
    onnx.save(optimized_model, optimized_path)

    # 2. FP16 преобразование
    fp16_path = str(optimized_path).replace('.onnx', '_fp16.onnx')
    if fp16:
        try:
            model_fp16 = convert_float_to_float16(onnx.load(optimized_path))
            onnx.save(model_fp16, fp16_path)
            optimized_path = fp16_path
        except ImportError:
            logger.warning("onnxconverter-common not installed, skipping FP16 conversion")
    
    # 3. Статическое INT8 квантование
    if quantize:
        # image_paths = [
        #     DIR_PATH / 'smith_aligned.jpg',
        #     DIR_PATH / 'scarlet_aligned.jpg'
        # ]
        
        # transform = transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация для модели
        # ])
        
        # Создание объекта для чтения калибровочных данных
        # print("calibration start")
        # calibration_data_reader = ImageCalibrationDataReader(image_paths, transform)
        # print("calibration true")
        
        # Выполнение статического квантования
        quantize_static(
            model_input=optimized_path,
            model_output=output_model_path,
            calibration_data_reader=calibration_data_reader,
            quant_format=QuantType.QUInt8,
            per_channel=True,  # Включение поканального квантования для весов
            weight_type=QuantType.QUInt8,
            activation_type=QuantType.QUInt8,
            extra_options = {'AddQDQPairToWeight': True},
            nodes_to_exclude=['/e4e_encoder/Resize_input_cast1', '/e4e_encoder/Resize_input_cast2']
        )
    else:
        import shutil
        shutil.copyfile(optimized_path, output_model_path)

if __name__ == "__main__":
    # Инициализация и экспорт оригинальной модели
    torch_model = init_model()
    dummy_input = interpolate.pt_output()

    output_names = ['w_e4e']
    
    # Экспорт оригинальной модели
    export_and_validate(
        model=torch_model,
        dummy_input=(dummy_input,),
        output_onnx_path=MODEL_PATH,
        output_names=output_names,
        skip_export=False,
        dynamo=False,
    )
    
    # Сжатие модели
    compressed_path = str(MODEL_PATH).replace('.onnx', '_compressed.onnx')
    compress_onnx_model(
        input_model_path=MODEL_PATH,
        output_model_path=compressed_path,
        quantize=True,    # Включить INT8 квантование
        fp16=True        # Включить FP16 преобразование
    )
    
    print(f"Original model size: {MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"Compressed model size: {Path(compressed_path).stat().st_size / (1024 * 1024):.2f} MB")