import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage


# ---------------- GPU Power Measurement ----------------
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

def get_gpu_power():
    """Return GPU power usage in Watts."""
    return nvmlDeviceGetPowerUsage(handle) / 1000  


# ---------------- Device ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ---------------- Simple CNN for Audio ----------------
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # assuming 64x64 mel-spectrogram input
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------- Audio Transform ----------------
transform = T.MelSpectrogram(
    sample_rate=16000,
    n_mels=64,
    n_fft=400,
    hop_length=160
)

def collate_fn(batch):
    """Convert raw waveform -> MelSpectrogram -> (1,64,64) for CNN"""
    specs, labels = [], []
    for waveform, label in batch:
        if waveform.size(0) > 1:  # stereo -> mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        spec = transform(waveform)
        spec = T.Resize((64, 64))(spec)  # resize to 64x64
        specs.append(spec)
        labels.append(label)
    specs = torch.stack(specs)
    labels = torch.tensor(labels)
    return specs, labels


# ---------------- Load Dataset ----------------
def get_dataset(name, root="./data"):
    if name == "speechcommands":
        dataset = torchaudio.datasets.SPEECHCOMMANDS(root=root, download=True, subset="testing")
        num_classes = len(dataset._labels)
    elif name == "librispeech":
        dataset = torchaudio.datasets.LIBRISPEECH(root=root, url="test-clean", download=True)
        num_classes = 10  # dummy, since it's ASR
    elif name == "commonvoice":
        dataset = torchaudio.datasets.COMMONVOICE(root=root, tsv="test.tsv", url="en", download=True)
        num_classes = 10  # dummy label space
    else:
        raise ValueError("Unknown dataset: " + name)
    return dataset, num_classes


# ---------------- Benchmark ----------------
def run_inference(dataset_name, batch_size=16, limit_batches=50):
    dataset, num_classes = get_dataset(dataset_name)

    # Replace final layer if needed
    model = AudioCNN(num_classes=num_classes).to(DEVICE)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    powers = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (specs, labels) in enumerate(dataloader):
            if limit_batches and batch_idx >= limit_batches:
                break
            print(f"[{dataset_name}] Processing batch {batch_idx+1}/{len(dataloader)}")

            specs, labels = specs.to(DEVICE), labels.to(DEVICE)

            # measure GPU power
            power = get_gpu_power()
            powers.append(power)

            _ = model(specs)

    end_time = time.time()
    avg_power = np.mean(powers)
    duration = end_time - start_time
    energy = avg_power * duration

    print(f"✅ {dataset_name} completed")
    print(f"→ Avg Power: {avg_power:.2f} W | Time: {duration:.2f} s | Energy: {energy:.2f} J")

    return avg_power, duration, energy


# ---------------- Run Tests ----------------
datasets = ["speechcommands", "librispeech", "commonvoice"]

print("\n📊 Power Consumption Results")
print("| Dataset        | Power (W) | Time (s) | Energy (J) |")
print("|----------------|-----------|----------|------------|")

for dset in datasets:
    try:
        power, duration, energy = run_inference(dset, batch_size=16, limit_batches=50)
        print(f"| {dset:14} | {power:9.2f} | {duration:8.2f} | {energy:10.2f} |")
    except Exception as e:
        print(f"| {dset:14} |   Error: {e}")
