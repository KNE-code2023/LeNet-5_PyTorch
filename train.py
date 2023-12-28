import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LeNet 
from data import get_data_loaders
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

def train(model, train_loader, criterion, optimizer, device):
    """ 訓練模型
    Args:
        model: 要訓練的模型
        train_loader: 用於訓練的 DataLoader
        criterion: 損失函數
        optimizer: 優化器
        device: 設備（"cuda" 或 "cpu"）

    Returns:
        train_loss: 訓練損失
        train_acc: 訓練準確率
    """
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
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.update(1)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    return train_loss, train_acc

def validate(model, test_loader, criterion, device):
    """ 驗證模型
    Args:
        model: 要驗證的模型
        test_loader: 用於驗證的 DataLoader
        criterion: 損失函數
        device: 設備（"cuda" 或 "cpu"）

    Returns:
        val_loss: 驗證損失
        val_acc: 驗證準確率
    """
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

def plot_results(train_losses, train_accuracies, val_losses, val_accuracies):
    """ 繪製訓練和驗證結果
    Args:
        train_losses: 訓練損失列表
        train_accuracies: 訓練準確率列表
        val_losses: 驗證損失列表
        val_accuracies: 驗證準確率列表
    """
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
    plt.show()

def main():
    # 讀取設定檔
    with open('config.json', 'r') as f:
        config = json.load(f)

    num_epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]

    # 獲取訓練和測試 DataLoader
    train_loader, test_loader = get_data_loaders(batch_size)
    num_classes = len(train_loader.dataset.classes)

    # 檢查 CUDA 是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_device_name = torch.cuda.get_device_name(0)
        print(f"Device Name: {cuda_device_name}")
    else:
        device = torch.device("cpu")
        print("No CUDA Devices available")

    # 初始化模型、損失函數、優化器和學習率調度器
    model = LeNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0

    # 開始訓練
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

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, 'model.pth')

    # 繪製 Loss 和 Acc 曲線
    plot_results(train_losses, train_accuracies, val_losses, val_accuracies)

if __name__ == "__main__":
    main()
