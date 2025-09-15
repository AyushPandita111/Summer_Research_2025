import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage


nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

def get_gpu_power():
    """Return GPU power usage in Watts."""
    return nvmlDeviceGetPowerUsage(handle) / 1000  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


model = resnet18(weights="IMAGENET1K_V1")   
model.fc = nn.Linear(model.fc.in_features, 1000)  
model.to(DEVICE)
model.eval()


transform = transforms.Compose([
    transforms.Resize(224),        # resize for ResNet
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_dataset(name, root="./data"):
    """Return dataset and num_classes based on name."""
    if name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        num_classes = 10
    elif name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
        num_classes = 100
    elif name == "svhn":  
        dataset = torchvision.datasets.SVHN(root=root, split="test", download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError("Unknown dataset: " + name)
    return dataset, num_classes

def run_inference(dataset_name, batch_size=64, limit_batches=None):
    dataset, num_classes = get_dataset(dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

 
    if model.fc.out_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features, num_classes).to(DEVICE)
        model.eval()

    powers = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if limit_batches and batch_idx >= limit_batches:
                break
            print(f"[{dataset_name}] Processing batch {batch_idx+1}/{len(dataloader)}")
            images = images.to(DEVICE)

     
            power = get_gpu_power()
            powers.append(power)

            _ = model(images)  # forward pass

    end_time = time.time()
    avg_power = np.mean(powers)
    duration = end_time - start_time
    energy = avg_power * duration

    print(f"✅ {dataset_name} completed")
    print(f"→ Avg Power: {avg_power:.2f} W | Time: {duration:.2f} s | Energy: {energy:.2f} J")

    return avg_power, duration, energy


datasets = ["cifar10", "cifar100", "svhn"]

print("\n📊 Power Consumption Results")
print("| Dataset   | Power (W) | Time (s) | Energy (J) |")
print("|-----------|-----------|----------|------------|")

for dset in datasets:
    try:
        power, duration, energy = run_inference(dset, batch_size=64, limit_batches=100)  
        print(f"| {dset:9} | {power:9.2f} | {duration:8.2f} | {energy:10.2f} |")
    except Exception as e:
        print(f"| {dset:9} |   Error: {e}")
