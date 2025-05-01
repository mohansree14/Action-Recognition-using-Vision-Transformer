import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarnings

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
import logging
import json
import sys

# Add the src directory to sys.path
sys.path.append('/content/drive/MyDrive/action-recognition-vit/src')

from models.vit import load_vit_model
from training.dataset import get_dataloader

# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(data_dir, epochs=10, batch_size=4, learning_rate=1e-4):
    train_loader, val_loader = get_dataloader(data_dir, batch_size)
    processor, model = load_vit_model()
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # For mixed precision training
    writer = SummaryWriter()
    metrics = {"loss": []}

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            with autocast():  # Enable mixed precision
                outputs = model(pixel_values=inputs).logits
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        metrics["loss"].append(epoch_loss)
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

    # Save metrics to a JSON file
    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f)

    writer.close()
    torch.save(model.state_dict(), "vit_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train_model(data_dir="/content/HMDB_simp")