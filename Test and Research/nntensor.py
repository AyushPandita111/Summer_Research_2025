import tensorflow as tf
from tensorflow.keras import layers, models
import time
from pynvml import *


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


x_train = x_train.astype('float32') / 255.0 * 2.0 - 1.0
x_test = x_test.astype('float32') / 255.0 * 2.0 - 1.0


model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  

power_samples = []

def record_power():
    power = nvmlDeviceGetPowerUsage(handle) / 1000.0  
    power_samples.append(power)


batch_size = 64
epochs = 5

start_time = time.time()

class PowerLogger(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        record_power()

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    verbose=1,
    callbacks=[PowerLogger()]
)

end_time = time.time()
training_duration = end_time - start_time


average_power = sum(power_samples) / len(power_samples) if power_samples else 0
energy_consumed = average_power * training_duration / 3600  # in watt-hours (Wh)

print(f"Training finished in {training_duration:.2f} seconds")
print(f"Average GPU power draw: {average_power:.2f} W")
print(f"Estimated GPU energy consumed: {energy_consumed:.4f} Wh")

nvmlShutdown()
