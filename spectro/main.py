import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from matplotlib.animation import FuncAnimation
import time
import torch
import torch.fft
from collections import deque

# Конфигурация
RT_CONFIG = {
    'n_fft': 2048,  # Уменьшено для производительности
    'hop_length': 256,
    'sr': 44100,
    'channels': 1,
    'cmap': 'inferno',
    'downsample': 4,
    'db_range': 80,
    'buffer_seconds': 2,
    'window': 'hann',
    'update_interval': 20,  # мс
    'use_gpu': True  # Переключение GPU/CPU
}


class RealtimeSpectrogram:
    def __init__(self):
        # Инициализация буферов
        self.audio_buffer = np.zeros(RT_CONFIG['sr'] * RT_CONFIG['buffer_seconds'], dtype=np.float32)
        self.spec_buffer = deque(maxlen=1)  # Для хранения последнего спектра
        self.time_buffer = deque()  # Для хранения временных меток

        # Размеры спектрограммы
        self.freq_bins = (RT_CONFIG['n_fft'] // 2) // RT_CONFIG['downsample']
        self.time_bins = int(RT_CONFIG['sr'] * RT_CONFIG['buffer_seconds'] / RT_CONFIG['hop_length'])

        # Инициализация спектрограммы
        self.spectrogram = np.full((self.freq_bins, self.time_bins), -RT_CONFIG['db_range'], dtype=np.float32)

        # Настройка GPU
        self.use_gpu = RT_CONFIG['use_gpu'] and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        print(f"Using {'GPU' if self.use_gpu else 'CPU'}")

        if self.use_gpu:
            # Создаем окно на GPU
            self.window = torch.hann_window(RT_CONFIG['n_fft'], device=self.device)

        # Создаем фигуру
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        max_freq = RT_CONFIG['sr'] // 2
        self.img = self.ax.imshow(
            self.spectrogram,
            aspect='auto',
            origin='lower',
            cmap=RT_CONFIG['cmap'],
            extent=[0, RT_CONFIG['buffer_seconds'], 0, max_freq],
            vmin=-RT_CONFIG['db_range'],
            vmax=0
        )
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')
        cbar = self.fig.colorbar(self.img, format='%+2.0f dB')
        cbar.set_label('dB Relative Scale')

        # Настройки темного фона
        self.fig.patch.set_facecolor('#000000')
        self.ax.set_facecolor('#000000')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')

        # Статистика производительности
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.processing_times = []

        # Поток аудио
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=RT_CONFIG['channels'],
            samplerate=RT_CONFIG['sr'],
            blocksize=RT_CONFIG['hop_length'],
            dtype=np.float32
        )

    def audio_callback(self, indata, frames, time_info, status):
        # Добавляем новые данные в буфер
        self.audio_buffer = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata[:, 0]

        # Обрабатываем данные сразу
        self.process_audio()

    def process_audio(self):
        """Обработка аудио и вычисление спектра"""
        # Берем последние n_fft сэмплов
        start_idx = -RT_CONFIG['n_fft']
        segment = self.audio_buffer[start_idx:] if start_idx < 0 else np.zeros(RT_CONFIG['n_fft'])

        # Обработка на GPU
        if self.use_gpu:
            segment_tensor = torch.tensor(segment, device=self.device, dtype=torch.float32)
            windowed = segment_tensor * self.window
            stft = torch.fft.rfft(windowed)
            magnitude = torch.abs(stft)
            db = 20 * torch.log10(torch.clamp(magnitude, min=1e-10))
            db = torch.clamp(db, min=-RT_CONFIG['db_range'], max=0)
            db = db[::RT_CONFIG['downsample']].cpu().numpy()

        # Обработка на CPU
        else:
            window = np.hanning(RT_CONFIG['n_fft'])
            windowed = segment * window
            stft = np.fft.rfft(windowed)
            magnitude = np.abs(stft)
            db = 20 * np.log10(np.maximum(magnitude, 1e-10))
            db = np.clip(db, -RT_CONFIG['db_range'], 0)
            db = db[::RT_CONFIG['downsample']]

        # Сохраняем результат с временной меткой
        timestamp = time.time()
        self.spec_buffer.append(db)
        self.time_buffer.append(timestamp)

    def update(self, frame):
        start_time = time.perf_counter()

        # Обновляем спектрограмму только при наличии новых данных
        if not self.spec_buffer:
            return [self.img]

        # Берем последний спектр
        db = self.spec_buffer.popleft()
        timestamp = self.time_buffer.popleft()

        # Удаляем устаревшие данные
        current_time = time.time()
        max_age = RT_CONFIG['buffer_seconds']

        # Обновляем спектрограмму
        self.spectrogram = np.roll(self.spectrogram, -1, axis=1)

        # Вставляем новые данные
        valid_bins = min(db.shape[0], self.spectrogram.shape[0])
        self.spectrogram[:valid_bins, -1] = db[:valid_bins]

        # Заполняем остаток минимальным значени
        if valid_bins < self.spectrogram.shape[0]:
            self.spectrogram[valid_bins:, -1] = -RT_CONFIG['db_range']

        self.img.set_data(self.spectrogram)

        self.frame_count += 1
        processing_time = time.perf_counter() - start_time
        self.processing_times.append(processing_time)

        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            avg_time = np.mean(self.processing_times) * 1000 if self.processing_times else 0
            self.frame_count = 0
            self.last_fps_time = current_time
            self.processing_times = []

        return [self.img]

    def start(self):
        self.stream.start()
        self.last_fps_time = time.time()

        self.ani = FuncAnimation(
            self.fig,
            self.update,
            interval=RT_CONFIG['update_interval'],
            blit=True,
            cache_frame_data=False

        )
        plt.tight_layout()
        plt.show(block=True)


# Запуск
rt_spec = RealtimeSpectrogram()
rt_spec.start()
