from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MNISTModel


BATCH_SIZE = 1
NUM_WARMUP_RUNS = 20
NUM_MEASURED_RUNS = 1000

script_directory = Path(__file__).resolve().parent

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# MNIST input preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(
    root=script_directory / "data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Load trained model
model = MNISTModel().to(device)

model.load_state_dict(
    torch.load(
        script_directory / "mnist_model.pth",
        map_location=device
    )
)

model.eval()


# Get one input batch
images, labels = next(iter(test_loader))
images = images.to(device)


def synchronize_device():
    """Wait for asynchronous accelerator operations to complete."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# Warm up the device
with torch.inference_mode():
    for _ in range(NUM_WARMUP_RUNS):
        model(images)

synchronize_device()


# Measure inference time
start_time = time.perf_counter()

with torch.inference_mode():
    for _ in range(NUM_MEASURED_RUNS):
        outputs = model(images)

synchronize_device()

end_time = time.perf_counter()


# Calculate performance
total_time = end_time - start_time
average_time = total_time / NUM_MEASURED_RUNS
throughput = (
    NUM_MEASURED_RUNS * BATCH_SIZE
) / total_time

prediction = outputs.argmax(dim=1)


print(f"Prediction: {prediction[0].item()}")
print(f"Actual: {labels[0].item()}")
print(f"Average latency: {average_time * 1000:.4f} ms")
print(f"Throughput: {throughput:.2f} images/second")