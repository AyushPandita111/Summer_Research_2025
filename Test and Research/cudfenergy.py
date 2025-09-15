import time
import cudf
from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # GPU 0


def get_gpu_power():
    return nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W


def log_gpu_power(operation_name, func, *args, **kwargs):
    print(f"Starting: {operation_name}")
    power_log = []

    start_time = time.time()

    for _ in range(5):
        power_log.append(get_gpu_power())
        time.sleep(0.05)

    result = func(*args, **kwargs)

    
    for _ in range(5):
        power_log.append(get_gpu_power())
        time.sleep(0.05)

    end_time = time.time()

    avg_power = sum(power_log) / len(power_log)
    total_energy = avg_power * (end_time - start_time)

    print(f"{operation_name} => Avg Power: {avg_power:.2f}W | Energy: {total_energy:.2f}J\n")
    return result

# ------------------- Start GPU Workflow ----------------------

print(">>> cuDF GPU Power Consumption Monitor <<<\n")

df = log_gpu_power("Load CSV", cudf.read_csv, '../Datasets/drugs.csv')


log_gpu_power("Check Nulls", df.isnull)
log_gpu_power("Drop Nulls", df.dropna)
log_gpu_power("Fill Nulls", df.fillna, 0)
log_gpu_power("Replace Values", df.replace, '?', 'X')


log_gpu_power("Drop Column", df.drop, columns=['drugName'])
log_gpu_power("Group By Rating", df.groupby, 'rating')
log_gpu_power("Sort By Rating", df.sort_values, by='rating')


log_gpu_power("Count", df.count)
log_gpu_power("Sum usefulCount", df['usefulCount'].sum)
log_gpu_power("Mean Rating", df['rating'].mean)
log_gpu_power("Min usefulCount", df['usefulCount'].min)
log_gpu_power("Max usefulCount", df['usefulCount'].max)
log_gpu_power("Unique Conditions", df['condition'].unique)

nvmlShutdown()
print(">>> Monitoring Complete <<<")
