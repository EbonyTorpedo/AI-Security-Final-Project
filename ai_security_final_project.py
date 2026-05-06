"""
AI Security Final Project
Author: Jeffery L Parker
Course: CTEC450
Instructor: Professor Carter
Assignment: Adversarial Machine Learning (FGSM Attack & Defense)

====================================================
README / HOW TO RUN
====================================================

Purpose:
This project demonstrates how machine learning models can be vulnerable to
adversarial attacks and how defensive techniques can improve model robustness.
A Convolutional Neural Network (CNN) is trained on the MNIST dataset to classify
handwritten digits. The model is then intentionally attacked using the Fast
Gradient Sign Method (FGSM) to observe how accuracy is affected. Finally,
adversarial training is applied as a defense mechanism to restore performance.

The project highlights the importance of securing AI systems and understanding
how attackers can exploit weaknesses in machine learning models.

----------------------------------------------------

Requirements:
- Python 3.10 or higher
- torch (PyTorch)
- torchvision
- matplotlib
- numpy

Install dependencies using:
pip install -r requirements.txt

----------------------------------------------------

How it works:

1. Build Baseline Model:
- A CNN model is trained on the MNIST dataset
- The model learns to classify digits (0–9)
- Baseline accuracy is measured on test data

2. Adversarial Attack (FGSM):
- Small perturbations are added to input images
- These changes are not visible to humans
- The model is evaluated on adversarial examples
- Accuracy drops significantly, demonstrating vulnerability

3. Defense (Adversarial Training):
- The model is retrained using adversarial examples
- This improves the model’s ability to resist attacks
- Accuracy is evaluated again after defense

----------------------------------------------------

Key Concept (FGSM Attack):

x_adv = x + ε * sign(∇x J(θ, x, y))

Where:
- x = original input image
- ε (epsilon) = attack strength
- ∇x J = gradient of the loss function
- sign() = direction of the gradient

----------------------------------------------------

Results Summary:

- Baseline Accuracy: ~98%
- Under Attack (FGSM): ~3–6%
- After Defense: ~98%

These results show that:
- Machine learning models can be easily fooled
- Adversarial attacks are highly effective
- Defensive techniques can significantly improve robustness

----------------------------------------------------

Ethical Use:

This project is intended for educational purposes only.
Adversarial machine learning techniques should only be used in controlled
environments for research, testing, and improving system security.

Do NOT use these techniques for malicious purposes.

----------------------------------------------------

Notes:

- The MNIST dataset is automatically downloaded on first run
- Training may take 1–3 minutes depending on system performance
- A graph is generated comparing model performance across all stages

====================================================
END OF README / COMMENT BLOCK
====================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

# Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train
def train(model, loader, epochs=2):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluate
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"Accuracy: {acc*100:.2f}%")
    return acc

# FGSM Attack
def fgsm_attack(image, epsilon, gradient):
    perturbation = epsilon * gradient.sign()
    adv_image = image + perturbation
    return torch.clamp(adv_image, 0, 1)

def eval_fgsm(model, loader, epsilon=0.25):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        gradient = images.grad.data
        adv_images = fgsm_attack(images, epsilon, gradient)

        outputs = model(adv_images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"FGSM Accuracy (epsilon={epsilon}): {acc*100:.2f}%")
    return acc

# Defense (Adversarial Training)
def adversarial_train(model, loader, epochs=2, epsilon=0.25):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            images.requires_grad = True

            outputs = model(images)
            loss = criterion(outputs, labels)

            model.zero_grad()
            loss.backward()

            gradient = images.grad.data
            adv_images = fgsm_attack(images, epsilon, gradient)

            optimizer.zero_grad()
            outputs = model(adv_images.detach())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Defense Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ---- RUN EVERYTHING ----
print("\n--- Training Baseline Model ---")
train(model, train_loader, epochs=2)

print("\n--- Baseline Evaluation ---")
baseline_acc = evaluate(model, test_loader)

print("\n--- Under Attack ---")
fgsm_acc = eval_fgsm(model, test_loader, epsilon=0.25)

print("\n--- Applying Defense ---")
adversarial_train(model, train_loader, epochs=2)

print("\n--- After Defense ---")
defended_acc = evaluate(model, test_loader)

# Graph Results
labels = ['Baseline', 'FGSM Attack', 'Defended']
values = [baseline_acc, fgsm_acc, defended_acc]

colors = ['green', 'red', 'green']

plt.figure(figsize=(8,5))
plt.bar(labels, values, color=colors)
plt.title("MNIST CNN Performance Under FGSM Attack and Defense", fontsize=14)
plt.ylabel("Accuracy", fontsize=12)

# Add percentage labels on top
for i, v in enumerate(values):
    plt.text(i, v + 0.015, f"{v*100:.2f}%", ha='center', fontsize=11)

plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig("model_performance.png", dpi=300, bbox_inches="tight")
plt.show()