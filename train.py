import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Import datasets first to avoid DLL conflict crashes on Windows
from datasets import load_dataset
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Disable Matplotlib/Seaborn to prevent system import crashes
matplotlib_available = False

# Import preprocessing pipelines
from preprocess import get_train_transforms, get_val_transforms

# Define the CNN Model Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv Block 1: 32 channels, 3x3 kernel
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Conv Block 2: 64 channels, 3x3 kernel
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Conv Block 3: 128 channels, 3x3 kernel
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.4)
        
        # Fully Connected Block
        # CIFAR-10 images are 32x32. After 3 MaxPools of 2x2, spatial size is 4x4.
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # Block 1
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(self.pool1(x))
        
        # Block 2
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.dropout2(self.pool2(x))
        
        # Block 3
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.dropout3(self.pool3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC Block
        x = torch.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

def train_model():
    print("DEBUG: Entered train_model()", flush=True)
    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for training: {device}", flush=True)
    
    # 2. Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    
    # Adapt configuration if running on CPU to keep time reasonable
    if device.type == 'cpu':
        print("Warning: CUDA is not available. Training on CPU might be slow.")
        print("Optimizing parameters: using a subset of data and training for more epochs.")
        epochs = 15
        use_subset = True
    else:
        epochs = 15
        use_subset = False
        
    # 3. Load Datasets
    from datasets import load_dataset
    print("Loading CIFAR-10 from Hugging Face...")
    hf_train_dataset = load_dataset("uoft-cs/cifar10", split="train")
    hf_test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    
    # Split train set indices into 80% train and 20% validation
    num_train = len(hf_train_dataset)
    indices = list(range(num_train))
    import random
    random.seed(42)
    random.shuffle(indices)
    
    split = int(0.8 * num_train)
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # If on CPU, take a smaller subset of train/val to train within 2-3 mins
    if use_subset:
        train_indices = train_indices[:10000]
        val_indices = val_indices[:2000]
        print(f"Using subset sizes: Train={len(train_indices)}, Val={len(val_indices)}")
        
    class HFIndexedDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, indices, transform=None):
            self.hf_dataset = hf_dataset
            self.indices = indices
            self.transform = transform
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            item = self.hf_dataset[real_idx]
            img = item['img']
            label = item['label']
            if self.transform:
                img = self.transform(img)
            return img, label
            
    # Wrap datasets
    train_dataset = HFIndexedDataset(hf_train_dataset, train_indices, get_train_transforms())
    val_dataset = HFIndexedDataset(hf_train_dataset, val_indices, get_val_transforms())
    test_dataset = HFIndexedDataset(hf_test_dataset, list(range(len(hf_test_dataset))), get_val_transforms())
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 4. Instantiate Model, Loss, Optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Print model summary
    print("\nModel Architecture Summary:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Trainable Parameters: {total_params:,}\n")
    
    # 5. Training Loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    model_save_path = './best_cnn_model.pth'
    
    print(f"Starting model training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track training metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = 100.0 * correct_train / total_train
        
        # Validation evaluation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = 100.0 * correct_val / total_val
        
        # Store history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
              
        # Save model if validation accuracy improves
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Best model saved with validation accuracy: {best_val_acc:.2f}%")
            
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nTraining completed in {duration:.2f} seconds.")
    print(f"Best Validation Accuracy achieved: {best_val_acc:.2f}%")
    
    # Load the best model for final evaluation on the test set
    best_model = SimpleCNN().to(device)
    if os.path.exists(model_save_path):
        best_model.load_state_dict(torch.load(model_save_path, map_location=device))
    best_model.eval()
    
    # 6. Evaluate on Test Dataset
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = best_model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\nTest Dataset Evaluation:")
    print(classification_report(all_targets, all_preds, target_names=classes))
    
    # 7. Generate and save visualizations
    if matplotlib_available:
        print("Generating training graphs and figures...")
        os.makedirs('./static', exist_ok=True)
        
        # Loss Curve
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', color='#4f46e5', marker='o')
        plt.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', color='#f43f5e', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./static/loss_curve.png', dpi=150)
        plt.close()
        
        # Accuracy Curve
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Acc', color='#06b6d4', marker='o')
        plt.plot(range(1, epochs + 1), history['val_acc'], label='Val Acc', color='#10b981', marker='o')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./static/accuracy_curve.png', dpi=150)
        plt.close()
        
        # Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=True)
        plt.title('Confusion Matrix on Test Dataset')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('./static/confusion_matrix.png', dpi=150)
        plt.close()
        
        print("Graphs saved to './static/loss_curve.png', './static/accuracy_curve.png', and './static/confusion_matrix.png'.")
    else:
        # If Matplotlib is not available, print confusion matrix in text format
        print("\nConfusion Matrix (Text format):")
        cm = confusion_matrix(all_targets, all_preds)
        print("     " + " ".join([f"{c[:4]:>5}" for c in classes]))
        for idx, row in enumerate(cm):
            row_str = " ".join([f"{val:>5d}" for val in row])
            print(f"{classes[idx][:4]:<4} | {row_str}")
        print("\nVisual plots were skipped because Matplotlib/Seaborn is unavailable on this environment.")

if __name__ == '__main__':
    train_model()
