# import pandas as pd
# import time
# from pynvml import *

# # Initialize NVML
# nvmlInit()
# handle = nvmlDeviceGetHandleByIndex(0)  # GPU 0

# def record_power():
#     power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
#     return power

# # Load the dataset
# url = 'file:///D:/ML/individual+household+electric+power+consumption/household_power_consumption.txt'
# print("Downloading and loading dataset...")
# df = pd.read_csv(url, sep=';', low_memory=False, na_values='?')

# # Convert 'Date' and 'Time' to datetime
# print("Processing datetime...")
# df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
# df.set_index('Datetime', inplace=True)

# # Convert relevant columns to numeric
# cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
#         'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
# df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

# # Drop rows with NaN values
# df.dropna(inplace=True)

# # Define tasks to perform
# tasks = {
#     'Sum': lambda x: x.sum(),
#     'Mean': lambda x: x.mean(),
#     'GroupBy Hour': lambda x: x.resample('H').mean(),
#     'GroupBy Day': lambda x: x.resample('D').mean(),
#     'Rolling Mean': lambda x: x.rolling(window=60).mean()
# }

# # Measure power consumption for each task
# for task_name, task_func in tasks.items():
#     print(f"\nPerforming task: {task_name}")
#     power_readings = []
#     start_time = time.time()
    
#     # Start measuring power
#     for _ in range(5):  # Take 5 readings during the task
#         power = record_power()
#         power_readings.append(power)
#         time.sleep(0.1)  # Sleep to simulate time between readings

#     # Perform the task
#     result = task_func(df['Global_active_power'])

#     end_time = time.time()
#     duration = end_time - start_time
#     avg_power = sum(power_readings) / len(power_readings)
#     energy_consumed = avg_power * duration / 3600  # in Wh

#     print(f"Task '{task_name}' completed in {duration:.2f} seconds.")
#     print(f"Average GPU power draw: {avg_power:.2f} W")
#     print(f"Estimated GPU energy consumed: {energy_consumed:.4f} Wh")

# # Shutdown NVML
# nvmlShutdown()




import pandas as pd
import time
from pynvml import *

# Initialize NVML for GPU power measurement
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

def record_power():
    """Get current GPU power usage in Watts"""
    return nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W

def perform_task(column_name, operation_name, operation_func):
    print(f"\nProcessing {operation_name} on '{column_name}'...")
    
    power_readings = []
    start_time = time.time()

    # Take power readings during task
    for _ in range(5):
        power_readings.append(record_power())
        time.sleep(0.1)

    # Run the operation
    result = operation_func(df[column_name])

    end_time = time.time()
    duration = end_time - start_time
    avg_power = sum(power_readings) / len(power_readings)
    energy_consumed = avg_power * duration / 3600  # Wh

    print(f"Time taken: {duration:.2f} sec")
    print(f"Avg GPU power: {avg_power:.2f} W")
    print(f"Estimated energy used: {energy_consumed:.4f} Wh\n")

# Load the dataset
print("Loading dataset...")
url = 'file:///D:/ML/individual+household+electric+power+consumption/household_power_consumption.txt'
df = pd.read_csv(url, sep=';', low_memory=False, na_values='?')

# Convert date and time to datetime index
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
df.set_index('Datetime', inplace=True)

# Convert numeric columns
columns_to_use = ['Global_active_power', 'Global_reactive_power', 'Voltage']
df[columns_to_use] = df[columns_to_use].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=columns_to_use, inplace=True)

# Define operations
operations = {
    'Sum': lambda x: x.sum(),
    'Mean': lambda x: x.mean(),
    'GroupBy Hour': lambda x: x.resample('H').mean(),
    'GroupBy Day': lambda x: x.resample('D').mean(),
    'Rolling Mean (60)': lambda x: x.rolling(window=60).mean()
}

# Run tasks on each column separately
for col in columns_to_use:
    for op_name, op_func in operations.items():
        perform_task(col, op_name, op_func)

# Shutdown NVML
nvmlShutdown()

