import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Subset
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage


class AudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # assuming input (1,64,64)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

def get_gpu_power():
    return nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts


def audio_collate(batch):
    """Convert waveform -> mel spectrogram -> (1,64,64)."""
    specs, labels = [], []
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_mels=64)
    resize = torchaudio.transforms.Resize((64, 64))

    for i, (waveform, sample_rate, label, *_rest) in enumerate(batch):
        if sample_rate != 8000:
            waveform = torchaudio.transforms.Resample(sample_rate, 8000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        spec = mel_spec(waveform)
        spec = spec.unsqueeze(0)  # add channel
        spec = resize(spec)
        specs.append(spec)
        labels.append(0 if isinstance(label, str) else int(label))

    return torch.cat(specs, dim=0), torch.tensor(labels)


def load_yesno(root="./data"):
    return torchaudio.datasets.YESNO(root=root, download=True)

def load_synthetic(root="./data", n_samples=100):
    """Generate synthetic sine wave dataset."""
    data = []
    for i in range(n_samples):
        freq = np.random.choice([200, 400, 600, 800])
        t = torch.linspace(0, 1, 8000)
        waveform = torch.sin(2 * np.pi * freq * t).unsqueeze(0)
        label = 0 if freq < 500 else 1
        data.append((waveform, 8000, label))
    return data


def run_inference(name, dataset, num_classes=2, batch_size=8, limit_batches=10, device="cuda"):
    model = AudioCNN(num_classes=num_classes).to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=audio_collate, num_workers=0)

    powers, start_time = [], time.time()
    with torch.no_grad():
        for i, (specs, labels) in enumerate(dataloader):
            if i >= limit_batches:
                break
            specs = specs.to(device)
            _ = model(specs)
            powers.append(get_gpu_power())
    end_time = time.time()

    if not powers:
        return None, None, None
    avg_power = np.mean(powers)
    duration = end_time - start_time
    energy = avg_power * duration
    return avg_power, duration, energy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    datasets = {
        "YESNO": (load_yesno, 2),
        "Synthetic": (load_synthetic, 2),
    }

    print("\n📊 Power Consumption Results")
    print("| Dataset     | Power (W) | Time (s) | Energy (J) |")
    print("|-------------|-----------|----------|------------|")

    for name, (fn, num_classes) in datasets.items():
        try:
            ds = fn()
            if hasattr(ds, "__len__") and len(ds) > 50:
                ds = Subset(ds, range(50))  # small subset
            power, duration, energy = run_inference(name, ds, num_classes, batch_size=8, limit_batches=10, device=device)
            if power is None:
                print(f"| {name:<11} | Skipped (no usable batches) |")
            else:
                print(f"| {name:<11} | {power:9.2f} | {duration:8.2f} | {energy:10.2f} |")
        except Exception as e:
            print(f"| {name:<11} | Skipped (error: {str(e)[:40]}) |")
