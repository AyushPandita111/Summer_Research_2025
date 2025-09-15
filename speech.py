
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import os
import time
import threading
import pynvml
import csv


def get_gpu_power():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    power = pynvml.nvmlDeviceGetPowerUsage(handle)
    pynvml.nvmlShutdown()
    return power / 1000.0

class PowerLogger(threading.Thread):
    def _init_(self, interval=0.1):
        super(PowerLogger, self)._init_()
        self.interval = interval
        self.running = False
        self.data = []

    def run(self):
        self.running = True
        while self.running:
            power = get_gpu_power()
            timestamp = time.time()
            self.data.append((timestamp, power))
            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def save_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'power_watts'])
            writer.writerows(self.data)

# Model
def create_speech_cnn_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


data_dir = tf.keras.utils.get_file(
    'speech_commands_v0.02',
    origin='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
    untar=True)
data_dir = os.path.join(os.path.dirname(data_dir), 'speech_commands_v0.02')


def decode_audio(file_path):
    audio = tfio.audio.AudioIOTensor(file_path)
    audio_tensor = tf.cast(audio.to_tensor(), tf.float32) / 32768.0
    audio_tensor = tf.squeeze(audio_tensor, axis=-1)
    audio_tensor = tfio.audio.resample(audio_tensor, rate_in=16000, rate_out=8000)
    spectrogram = tf.signal.stft(audio_tensor, frame_length=256, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.image.resize(tf.expand_dims(spectrogram, -1), (64, 64))
    return spectrogram


yes_files = tf.io.gfile.glob(os.path.join(data_dir, 'yes/*.wav'))[:512]
x_train = np.stack([decode_audio(f).numpy() for f in yes_files])
y_train = np.zeros(len(x_train), dtype=int)


model = create_speech_cnn_model((64, 64, 1), 10)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

logger = PowerLogger()
logger.start()

start = time.time()
model.fit(x_train, y_train, batch_size=32, epochs=3)
end = time.time()

logger.stop()
logger.join()
logger.save_to_csv("speech_commands_gpu_power.csv")

print(f"\nTraining finished in {end - start:.2f} sec, GPU power log saved.")