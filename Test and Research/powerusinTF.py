import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from pynvml import *

# Initialize NVML (NVIDIA Management Library)
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # GPU 0

def get_power():
    # Returns power in watts
    return nvmlDeviceGetPowerUsage(handle) / 1000.0

# Load pretrained MobileNetV2 model + weights
model = tf.keras.applications.MobileNetV2(weights='imagenet')
model.trainable = False

# Create dummy input matching the model's expected input shape
dummy_input = tf.random.normal([1, 224, 224, 3])

# Warm-up (1st pass is always slower due to graph building)
_ = model(dummy_input, training=False)

# Inference + Power measurement loop
power_log = []
start = time.time()

for _ in range(50):
    power = get_power()
    power_log.append(power)
    
    _ = model(dummy_input, training=False)
    time.sleep(0.2)  # Sampling interval

end = time.time()

# Shutdown NVML
nvmlShutdown()

# Reporting
avg_power = sum(power_log) / len(power_log)
total_energy = avg_power * (end - start)  # in Joules

print(f"Average Power: {avg_power:.2f} W")
print(f"Total Energy Used: {total_energy:.2f} J")
