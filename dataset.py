# ======================
# file: dataset.py
# author: KONI-code2023@github
# date: 2024-01-16
# ======================
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 計算數據集的均值和標準差
def calculate_mean_std(data_loader):
    mean = 0.0
    std = 0.0
    total_images = 0

    # 逐批次遍歷數據集
    for images, _ in data_loader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_size

    mean /= total_images
    std /= total_images

    return mean, std

# 載入MNIST數據集
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])

    # 訓練數據集
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # 計算均值和標準差
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    mean, std = calculate_mean_std(train_loader)

    # 使用均值和標準差進行歸一化
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
    ])

    # 重新載入訓練和測試數據集
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    return train_dataset, test_dataset

# 獲取數據加載器
def get_data_loaders(batch_size):
    train_dataset, test_dataset = load_mnist()

    # 訓練和測試數據加載器
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 主函數
def main():
    print("Loading MNIST dataset...")
    train_dataset, test_dataset = load_mnist()

    print(f"Number of train dataset: {len(train_dataset)}")
    print(f"Number of test datset: {len(test_dataset)}")

    sample_image, _ = train_dataset[0]
    print(f"Sample Image Shape: {sample_image.shape}")


if __name__ == "__main__":
    main()
