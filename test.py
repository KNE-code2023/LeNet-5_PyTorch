# ======================
# file: test.py
# author: KONI-code2023@github
# date: 2024-01-16
# ======================
import sys
import platform
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from model.LeNet import LeNet
from dataset import get_data_loaders

# 測試函數，用於評估模型性能
def test(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return all_labels, all_predictions

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
    
    # 獲取測試數據加載器
    _, test_loader = get_data_loaders(batch_size=64)

    # 載入預先訓練的模型
    model = torch.load('results/lenet_mnist.pth').to(device)
    model.eval()

    # 進行模型測試
    all_labels, all_predictions = test(model, test_loader, device)

    # 計算混淆矩陣和性能指標
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    model_name = 'LeNet'

    # 繪製混淆矩陣
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=test_loader.dataset.classes, yticklabels=test_loader.dataset.classes)
    plt.text(0.5, 1.03, f'{model_name}  |  Accuracy: {accuracy:.4f}  |  Precision: {np.mean(precision):.4f}'
                 f'  |  Recall: {np.mean(recall):.4f}  |  F1-Score: {np.mean(f1):.4f}',
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('results/confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    main()
