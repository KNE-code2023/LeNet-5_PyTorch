import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from model import LeNet
from data import get_data_loaders
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def test(model, test_loader, device):
    """ 測試模型
    Args:
        model: 要測試的模型
        test_loader: 用於測試的 DataLoader
        device: 設備（"cuda" 或 "cpu"）

    Returns:
        all_labels: 所有真實標籤
        all_predictions: 所有預測標籤
    """
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
    # 獲取測試 DataLoader
    _, test_loader = get_data_loaders(batch_size=64)

    # 檢查 CUDA 是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_device_name = torch.cuda.get_device_name(0)
        print(f"Device Name: {cuda_device_name}")
    else:
        device = torch.device("cpu")
        print("No CUDA Devices available")

    # 載入先前訓練好的模型
    model = torch.load('model.pth')
    model.eval()

    # 進行測試，獲取所有真實標籤和預測標籤
    all_labels, all_predictions = test(model, test_loader, device)

    # 計算混淆矩陣和模型性能指標
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    model_name='LeNet'

    # 繪製混淆矩陣
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=test_loader.dataset.classes, yticklabels=test_loader.dataset.classes)
    plt.text(0.5, 1.03, f'{model_name}  |  Accuracy: {accuracy:.4f}  |  Precision: {np.mean(precision):.4f}'
                 f'  |  Recall: {np.mean(recall):.4f}  |  F1-Score: {np.mean(f1):.4f}',
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    main()
