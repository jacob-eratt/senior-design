from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MNISTModel


# Get the directory containing this script
script_directory = Path(__file__).resolve().parent

# Select CPU or GPU
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# Define the same transformation used during training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# Load the MNIST test dataset
test_dataset = datasets.MNIST(
    root=script_directory / "data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True
)


# Create and load the trained model
model = MNISTModel().to(device)

model_path = script_directory / "mnist_model.pth"

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.eval()
print("Model loaded successfully!")


# Get one batch of test images
images, labels = next(iter(test_loader))
images = images.to(device)


# Make predictions
with torch.no_grad():
    outputs = model(images)
    probabilities = torch.softmax(outputs, dim=1)
    predictions = outputs.argmax(dim=1)


# Move results back to the CPU for visualization
images = images.cpu()
predictions = predictions.cpu()
probabilities = probabilities.cpu()


# Display the first 10 images
fig, axes = plt.subplots(2, 5, figsize=(11, 6))

for index, axis in enumerate(axes.flat):
    predicted_digit = predictions[index].item()
    actual_digit = labels[index].item()
    confidence = probabilities[index, predicted_digit].item() * 100

    # Undo normalization for display
    displayed_image = images[index].squeeze() * 0.3081 + 0.1307

    axis.imshow(displayed_image, cmap="gray")
    axis.set_title(
        f"Prediction: {predicted_digit}\n"
        f"Actual: {actual_digit}\n"
        f"Confidence: {confidence:.1f}%"
    )

    axis.axis("off")

plt.tight_layout()
plt.show()