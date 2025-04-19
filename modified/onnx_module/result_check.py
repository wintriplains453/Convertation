from pathlib import Path
import numpy as np
import onnxruntime as ort
import onnx
import matplotlib.pyplot as plt
from modified.preprocess import preprocess_image
from modified.onnx_module.utils import run_onnx, ONNX_MODELS_PATH

DIR_PATH = Path(__file__).parent.resolve()
MODEL_PATH = DIR_PATH / 'onnx_models/e4e_encoder.onnx'

def get_model_io_names(model_path: Path):
    """Получаем имена входных и выходных узлов модели"""
    sess = ort.InferenceSession(str(model_path))
    inputs = {inp.name: inp.shape for inp in sess.get_inputs()}
    outputs = [out.name for out in sess.get_outputs()]
    return inputs, outputs

def compare_models(original_path: Path, compressed_path: Path, test_input: np.ndarray):
    """Сравнение моделей с автоматическим преобразованием типов"""
    # Загружаем модели и получаем информацию о входах
    sess_orig = ort.InferenceSession(str(original_path))
    sess_comp = ort.InferenceSession(str(compressed_path))
    
    orig_input = sess_orig.get_inputs()[0]
    comp_input = sess_comp.get_inputs()[0]
    
    # Преобразуем входные данные к нужному типу
    def prepare_input(data, input_info):
        if 'float16' in input_info.type:
            return data.astype(np.float16)
        elif 'int8' in input_info.type:
            return data.astype(np.int8)
        return data.astype(np.float32)
    
    # Подготавливаем входы для каждой модели
    orig_input_data = prepare_input(test_input, orig_input)
    comp_input_data = prepare_input(test_input, comp_input)
    
    # Запускаем inference
    orig_output = sess_orig.run(None, {orig_input.name: orig_input_data})[0]
    comp_output = sess_comp.run(None, {comp_input.name: comp_input_data})[0]
    
    # Конвертируем выходы к float32 для сравнения
    orig_output = orig_output.astype(np.float32)
    comp_output = comp_output.astype(np.float32)
    
    # Вычисляем различия
    abs_diff = np.abs(orig_output - comp_output)
    rel_diff = abs_diff / (np.abs(orig_output) + 1e-9)
    
    print("\nРезультаты сравнения:")
    print(f"Макс. абсолютная разница: {abs_diff.max():.2e}")
    print(f"Средняя абсолютная разница: {abs_diff.mean():.2e}")
    print(f"Макс. относительная разница: {rel_diff.max():.2e}")
    print(f"Средняя относительная разница: {rel_diff.mean():.2e}")
    
    return abs_diff, rel_diff

def main():
    # Пути к моделям
    original_path = DIR_PATH / 'onnx_models/e4e_encoder.onnx'
    compressed_path = DIR_PATH / 'onnx_models/e4e_encoder_compressed.onnx'
    
    # Проверка существования файлов
    if not original_path.exists():
        raise FileNotFoundError(f"Оригинальная модель не найдена: {original_path}")
    if not compressed_path.exists():
        raise FileNotFoundError(f"Сжатая модель не найдена: {compressed_path}")

    # Проверка модели
    model = onnx.load(str(original_path))
    onnx.checker.check_model(model)
    print(f"Модель {original_path.name} валидна")
    
    # Сравнение размеров
    orig_size = original_path.stat().st_size / (1024*1024)
    comp_size = compressed_path.stat().st_size / (1024*1024)
    print(f"\nРазмеры файлов:")
    print(f"Оригинал: {orig_size:.2f} MB")
    print(f"Сжатая: {comp_size:.2f} MB")
    print(f"Сжатие: {(1-comp_size/orig_size)*100:.1f}%")

    # Тестовый прогон
    test_image = DIR_PATH / 'smith_aligned.jpg'
    if not test_image.exists():
        print("\nТестовое изображение не найдено, использую случайные данные")
        input_shape = next(iter(get_model_io_names(original_path)[0].values()))
        test_input = np.random.randn(*input_shape).astype(np.float32)
    else:
        test_input = preprocess_image(test_image, resize_size = 256)
    
    # Сравнение моделей
    print("\nЗапуск сравнения моделей...")
    compare_models(original_path, compressed_path, test_input)

if __name__ == "__main__":
    main()



# 1) Макс. абсолютная разница: 2.73e-03 = 0.00273 (Максимальное по модулю различие между выходными значениями моделей)
# 2) Средняя абсолютная разница: 3.04e-04 = 0.000304 (Среднее отклонение по всем значениям. Указывает на высокую схожесть моделей)
# 3) Макс. относительная разница: 1.26e+01 = 1260% (При больших значениях не имеет смысла если пункт 1 мал)
# 4) Средняя относительная разница: 6.34e-03 = 0.634% (Реальный показатель схожести. Значение <1% отличный результат)