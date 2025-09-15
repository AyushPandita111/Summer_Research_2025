import pandas as pd
import numpy as np
import time
from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0
power_samples = []

def record_power():
    power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
    power_samples.append(power)

def measure_task(task_fn, description):
    global power_samples
    power_samples = [] 

    print(f"\nRunning Task: {description}")
    start_time = time.time()

   
    for _ in range(3):
        record_power()
        time.sleep(0.1) 

    task_fn()

    for _ in range(3):
        record_power()
        time.sleep(0.1)

    end_time = time.time()

    duration = end_time - start_time
    average_power = sum(power_samples) / len(power_samples)
    energy = average_power * duration / 3600  # in Wh

    print(f"Duration: {duration:.4f} sec | Avg Power: {average_power:.2f} W | Energy: {energy:.6f} Wh")


num_rows = 1_000_000
df = pd.DataFrame({
    'A': np.random.rand(num_rows),
    'B': np.random.rand(num_rows),
    'C': np.random.randint(0, 100, num_rows),
    'D': np.random.choice(['X', 'Y', 'Z'], num_rows)
})



tasks = [
    (lambda: df['A'].sum(), "Sum of column A"),
    (lambda: df[['A', 'B']].mean(), "Mean of A and B"),
    (lambda: df.groupby('D')['A'].sum(), "GroupBy D and Sum A"),
    (lambda: df[df['C'] > 50], "Filter rows where C > 50"),
    (lambda: df.sort_values(by='B'), "Sort by column B"),
    (lambda: df.describe(), "Describe dataframe"),
]

for task_fn, desc in tasks:
    measure_task(task_fn, desc)

nvmlShutdown()
