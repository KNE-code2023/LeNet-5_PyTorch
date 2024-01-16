# ======================
# file: train.py
# author: KONI-code2023@github
# date: 2024-01-16
# ======================
import sys
import json
import platform
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

from model.LeNet import LeNet 
from dataset import get_data_loaders

# 訓練函數，用於訓練模型
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(total=len(train_loader), desc='Training') as pbar:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.update(1)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    return train_loss, train_acc

# 驗證函數，用於評估模型性能
def validate(model, test_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with tqdm(total=len(test_loader), desc='Validation') as pbar:
        with torch.no_grad():
            for val_inputs, val_labels in test_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()

                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()

                pbar.update(1)

    val_loss /= len(test_loader)
    val_acc = val_correct / val_total

    return val_loss, val_acc

# 繪製訓練曲線的函數
def plot_results(train_losses, train_accuracies, val_losses, val_accuracies, save_path=None):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# 主函數
def main():
    # 獲取系統和庫的版本信息
    python_version = sys.version
    print(f"Python Version: {python_version}")

    torch_version = torch.__version__
    print(f"Torch Version: {torch_version}")

    cuda_version = torch.version.cuda
    print(f"CUDA Version: {cuda_version}")

    print(f"Operating System: {platform.system()} {platform.release()}")

    # 檢查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    # 根據CUDA的可用性設置計算設備
    if cuda_available: 
        device = torch.device("cuda")
        cuda_device_name = torch.cuda.get_device_name(0)
        print(f"CUDA Device Name: {cuda_device_name}")
    else:
        device = torch.device("cpu")
        print("No CUDA Devices available.")

    # 讀取設定檔
    with open('config.json', 'r') as f:
        config = json.load(f)

    num_epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    min_learning_rate = config["min_learning_rate"]
    weight_decay_rate = config["weight_decay_rate"]

    # 獲取訓練和測試 DataLoader
    train_loader, test_loader = get_data_loaders(batch_size)
    num_classes = len(train_loader.dataset.classes)

    # 初始化模型、損失函數、優化器和學習率調整器
    model = LeNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_learning_rate)

    # 初始化用於保存結果的列表
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0

    # 訓練循環
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_loss, val_acc = validate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train_Loss: {train_loss:.4f}, Train_Accuracy: {train_acc:.4f} - "
              f"Validation_Loss: {val_loss:.4f}, Validation_Accuracy: {val_acc:.4f}")

        # 如果在驗證集上取得更好的準確度，保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, 'results/lenet_mnist.pth')

    # 繪製訓練曲線
    plot_results(train_losses, train_accuracies, val_losses, val_accuracies, save_path='results/training_curve.png')

if __name__ == "__main__":
    main()
