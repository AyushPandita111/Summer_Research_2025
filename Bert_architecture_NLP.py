import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage


nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

def get_gpu_power():
    return nvmlDeviceGetPowerUsage(handle) / 1000  


BERT_DIR = "D:/ML/models/bert-base-uncased"   
DATA_DIR = "D:/ML/data"                   
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
model = BertForSequenceClassification.from_pretrained(BERT_DIR, num_labels=2).to(DEVICE)
model.eval()


class TextDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
       
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) == 0:
            raise ValueError(f"No text columns found in {csv_file}")
        self.texts = df[text_cols[0]].astype(str).tolist()
        print(f"Loaded {len(self.texts)} samples from {csv_file} using column '{text_cols[0]}'")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def run_inference(csv_path, batch_size=16):
    dataset = TextDataset(csv_path)
    if len(dataset) == 0:
        print(f"Dataset {csv_path} is empty, skipping")
        return 0, 0, 0

    dataloader = DataLoader(dataset, batch_size=batch_size)

    powers = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch_texts in enumerate(dataloader):
            print(f"Processing batch {batch_idx+1}/{len(dataloader)}")
            encoding = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
            power = get_gpu_power()
            powers.append(power)
            _ = model(**encoding)

    end_time = time.time()
    avg_power = np.mean(powers)
    duration = end_time - start_time
    energy = avg_power * duration

    print(f"{os.path.basename(csv_path)} completed")
    print(f"→ Avg Power: {avg_power:.2f} W | Time: {duration:.2f} s | Energy: {energy:.2f} J")

    return avg_power, duration, energy


dataset_files = [
    "imdb_test.csv",
    "ag_news_test.csv",
    "yelp_polarity_test.csv",
]

print("\n📊 Power Consumption Results")
print("| Dataset             | Power (W) | Time (s) | Energy (J) |")
print("|---------------------|-----------|----------|------------|")

for filename in dataset_files:
    full_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(full_path):
        print(f"File missing: {full_path}")
        continue
    power, duration, energy = run_inference(full_path)
    print(f"| {filename:20} | {power:9.2f} | {duration:8.2f} | {energy:10.2f} |")
