import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import DeepfakeDataset
from models import get_model
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os

def train_model(data_dir="data", epochs=7, batch_size=16, lr=1e-5):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    # Datasets
    train_dataset = DeepfakeDataset(os.path.join(data_dir, "train"))
    val_dataset   = DeepfakeDataset(os.path.join(data_dir, "val"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load EfficientNet Model
    model = get_model().to(device)

    # Optimizer (only trainable params)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds, all_labels = [], []

        print(f"\nEpoch {epoch+1}/{epochs}")

        for imgs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):

            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)

        # Save model
        torch.save(model.state_dict(), f"efficientnet_epoch_{epoch+1}.pth")

        print(f"Loss: {train_loss/len(train_loader):.4f} | Accuracy: {train_acc*100:.2f}%")

if __name__ == "__main__":
    train_model()
