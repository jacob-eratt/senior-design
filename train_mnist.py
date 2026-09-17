import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ----------------------------
# 1. Configuration
# ----------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# ----------------------------
# 2. Load MNIST
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ----------------------------
# 3. Define the CNN
# ----------------------------
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Input: [batch, 1, 28, 28]
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [batch, 32, 14, 14]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # [batch, 64, 7, 7]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = MNISTModel().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ----------------------------
# 4. Training loop
# ----------------------------
for epoch in range(NUM_EPOCHS):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous iteration
        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = loss_function(predictions, labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()

        predicted_labels = predictions.argmax(dim=1)
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / len(train_loader)
    training_accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Training accuracy: {training_accuracy:.2f}%"
    )


# ----------------------------
# 5. Evaluate on test data
# ----------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        predicted_labels = predictions.argmax(dim=1)

        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"Test accuracy: {test_accuracy:.2f}%")


# ----------------------------
# 6. Save the trained model
# ----------------------------
torch.save(model.state_dict(), "mnist_model.pth")
print("Model saved as mnist_model.pth")