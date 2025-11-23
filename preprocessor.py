import numpy as np
import librosa
import json
from pydub import AudioSegment
import io
import tempfile
import os
#import onnxruntime as ort

def is_audio_file(content_type: str, filename: str, file_content: bytes) -> bool:
    """Проверяет, является ли файл аудио"""
    if content_type and content_type.startswith('audio/'):
        return True

    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
    file_extension = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
    if file_extension in audio_extensions:
        return True
    
    if not content_type and file_extension in audio_extensions:
        return True
    return False

def convert_audio_to_wav(audio_data: bytes, filename: str) -> bytes:
    """Конвертирует аудио в WAV формат (16kHz, mono)"""
    try:
        file_extension = filename.split('.')[-1].lower()
        
        audio_buffer = io.BytesIO(audio_data)
        audio = AudioSegment.from_file(audio_buffer, format=file_extension)
        
        # Приводим к стандартному формату
        audio = audio.set_channels(1)  # mono
        audio = audio.set_frame_rate(16000)  # 16kHz
        audio = audio.set_sample_width(2)  # 16-bit

        output_buffer = io.BytesIO()
        audio.export(output_buffer, format="wav")
        
        return output_buffer.getvalue()  
    except Exception as e:
        raise Exception(f"Ошибка конвертации аудио: {str(e)}")

def load_audio_from_bytes(audio_data: bytes, filename: str = None) -> np.ndarray:
    """Загружает аудио из bytes и возвращает numpy массив"""
    try:
        # Создаем временный файл для загрузки
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        # Загружаем аудио с помощью librosa
        audio, sr = librosa.load(temp_file_path, sr=16000, mono=True)
        
        print(f"📊 Аудио загружено: {len(audio)} samples, {len(audio)/sr:.2f} секунд, SR: {sr}Hz")
        
        return audio
        
    except Exception as e:
        raise Exception(f"Ошибка загрузки аудио: {str(e)}")
    finally:
        # Удаляем временный файл
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

class AudioPreprocessor:
    """Препроцессор идентичный NeMo"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400
        self.hop_length = 160
        self.n_mels = 80
        self.window = 'hann'
        self.f_min = 0
        self.f_max = 8000
        self.dither = 1e-05
        self.preemph = 0.97
        self.log_zero_guard_value = 2**-24
        
    def compute_mel_spectrogram(self, audio):
        """Вычисление Mel-спектрограммы"""
        # Пре-эмфаза
        audio = np.append(audio[0], audio[1:] - self.preemph * audio[:-1])
        
        # STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True
        )
        
        # Амплитудный спектр
        magnitude = np.abs(stft)
        
        # Mel-фильтры
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            norm='slaney'
        )
        
        # Применение Mel-фильтров
        mel_spectrogram = np.dot(mel_basis, magnitude)
        
        # Логарифмирование
        log_mel = np.log(np.clip(mel_spectrogram, a_min=self.log_zero_guard_value, a_max=None))
        
        return log_mel
    
    def normalize_batch(self, features, seq_len):
        """Нормализация признаков"""
        mean = features.mean(axis=2, keepdims=True)
        std = features.std(axis=2, keepdims=True)
        normalized = (features - mean) / (std + 1e-5)
        return normalized, seq_len
    
    def __call__(self, audio_signal, audio_length):
        """Основной метод препроцессинга"""
        audio_signal = np.array(audio_signal, dtype=np.float32)
        
        batch_size = audio_signal.shape[0]
        features_list = []
        features_lengths = []
        
        for i in range(batch_size):
            audio = audio_signal[i]
            length = audio_length[i]
            audio = audio[:length]
            
            mel_spec = self.compute_mel_spectrogram(audio)
            features_list.append(mel_spec)
            features_lengths.append(mel_spec.shape[1])
        
        # Собираем батч
        max_length = max(features_lengths)
        batch_features = np.zeros((batch_size, self.n_mels, max_length), dtype=np.float32)
        
        for i, feat in enumerate(features_list):
            batch_features[i, :, :feat.shape[1]] = feat
        
        features_lengths = np.array(features_lengths, dtype=np.int64)
        
        # Нормализация
        batch_features, features_lengths = self.normalize_batch(batch_features, features_lengths)
        
        return batch_features, features_lengths

def save_input_data_for_go(audio_signal, length, filename="go_input_data.json"):
    """Сохраняет входные данные для использования в Go (точная копия из оригинального кода)"""
    
    input_data = {
        "audio_signal": {
            "data": audio_signal.flatten().tolist(),
            "shape": list(audio_signal.shape),
            "dtype": str(audio_signal.dtype)
        },
        "length": {
            "data": length.flatten().tolist(),
            "shape": list(length.shape),
            "dtype": str(length.dtype)
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(input_data, f, indent=2)
    
    print(f"💾 Данные для Go сохранены в: {filename}")
    print(f"📊 Audio signal shape: {audio_signal.shape}")
    print(f"📊 Length: {length}")
    
    # Также сохраняем в бинарном формате для удобства
    audio_signal.astype(np.float32).tofile('audio_signal.bin')
    length.astype(np.float32).tofile('length.bin')
    
    print("💾 Бинарные файлы сохранены: audio_signal.bin, length.bin")

def process_audio_file_for_onnx(audio_data: bytes, filename: str, content_type: str = None, model_path: str = None) -> tuple:
    """
    Основная функция обработки аудио файла для ONNX модели
    Возвращает: (input_dict, audio_info)
    """
    # Проверяем, что файл является аудио
    if not is_audio_file(content_type, filename, audio_data):
        raise ValueError(f"Файл не является аудио: {filename}, content-type: {content_type}")
    
    # Конвертируем в WAV если нужно
    if not filename.lower().endswith('.wav'):
        print(f"🔧 Конвертируем {filename} в WAV...")
        audio_data = convert_audio_to_wav(audio_data, filename)
    
    # Загружаем аудио как numpy массив
    audio_array = load_audio_from_bytes(audio_data, filename)
    
    # Подготавливаем входные данные для препроцессора
    audio_batch = np.expand_dims(audio_array, axis=0).astype(np.float32)
    audio_length = np.array([audio_batch.shape[1]], dtype=np.int64)
    
    # Запускаем препроцессор
    preprocessor = AudioPreprocessor()
    processed_audio, processed_audio_length = preprocessor(audio_batch, audio_length)
    
    print(f"📊 Mel-спектрограмма: {processed_audio.shape}")

    # Получаем информацию о входах модели для правильных типов
    if model_path and os.path.exists(model_path):
        model = ort.InferenceSession(model_path)
        model_inputs = model.get_inputs()
        
        input_dict = {}
        for input_info in model_inputs:
            if input_info.name == 'audio_signal':
                input_dict[input_info.name] = processed_audio.astype(np.float32)
            elif input_info.name == 'length':
                # Преобразуем в float если модель ожидает float
                if 'float' in input_info.type:
                    input_dict[input_info.name] = processed_audio_length.astype(np.float32)
                else:
                    input_dict[input_info.name] = processed_audio_length.astype(np.int64)
            else:
                input_dict[input_info.name] = processed_audio.astype(np.float32)
        
        # Отладочная информация
        print("Типы входных данных для модели:")
        for input_name, input_data in input_dict.items():
            print(f"  {input_name}: {input_data.dtype}, shape: {input_data.shape}")
    else:
        # Если модель не указана, используем стандартные типы
        input_dict = {
            'audio_signal': processed_audio.astype(np.float32),
            'length': processed_audio_length.astype(np.float32)  # по умолчанию float
        }
    
    # Сохраняем данные для Go
    save_input_data_for_go(
        input_dict['audio_signal'], 
        input_dict['length']
    )
    
    # Информация об аудио
    audio_info = {
        'original_filename': filename,
        'content_type': content_type,
        'audio_samples': len(audio_array),
        'duration_seconds': len(audio_array) / 16000,
        'features_shape': processed_audio.shape,
        'features_length': int(processed_audio_length[0])
    }
    
    return input_dict, audio_info

def load_audio_file(file_path: str):
    """Загружает аудио файл с диска и возвращает bytes"""
    try:
        with open(file_path, 'rb') as f:
            audio_data = f.read()
        print(f"📁 Файл загружен: {file_path} ({len(audio_data)} bytes)")
        return audio_data
    except Exception as e:
        raise Exception(f"Ошибка загрузки файла {file_path}: {e}")

# Пример использования с реальным файлом
if __name__ == "__main__":
    # Укажите путь к вашему реальному аудио файлу
    audio_file_path = "audio.wav"  # измените на путь к вашему файлу
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Файл {audio_file_path} не найден!")
        print("Создаю тестовый аудио файл...")
        
        # Создаем тестовый файл если его нет
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_signal = np.sin(2 * np.pi * 440 * t)  # тон 440 Гц
        
        # Сохраняем как WAV файл
        audio_segment = AudioSegment(
            (audio_signal * 32767).astype(np.int16).tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=1
        )
        audio_segment.export(audio_file_path, format="wav")
        print(f"✅ Создан тестовый файл: {audio_file_path}")
    
    try:
        # Загружаем реальный аудио файл
        audio_data = load_audio_file(audio_file_path)
        
        # Обрабатываем аудио
        input_dict, audio_info = process_audio_file_for_onnx(
            audio_data=audio_data,
            filename=os.path.basename(audio_file_path),
            content_type="audio/wav",
            model_path="model_fixed.onnx" if os.path.exists("model_fixed.onnx") else None
        )
        
        print("\n📊 Результаты обработки:")
        print(f"Входное аудио: {audio_info['audio_samples']} samples, {audio_info['duration_seconds']:.2f}s")
        print(f"Выходные фичи: {audio_info['features_shape']}")
        
        # Проверяем сохраненный файл
        if os.path.exists("go_input_data.json"):
            with open("go_input_data.json", "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            
            print(f"\n💾 Проверка сохраненного файла:")
            print(f"Audio signal shape: {saved_data['audio_signal']['shape']}")
            print(f"Length: {saved_data['length']['data']}")
            print(f"Типы данных: audio_signal={saved_data['audio_signal']['dtype']}, length={saved_data['length']['dtype']}")
            
            # Показываем первые несколько значений для проверки
            print(f"\n🔍 Первые 5 значений audio_signal:")
            print(saved_data['audio_signal']['data'][:5])
            
            # Показываем структуру
            print(f"\n📋 Структура JSON (первые 200 символов):")
            print(json.dumps(saved_data, indent=2)[:200] + "...")
        
    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")
