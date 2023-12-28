# data.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def calculate_mean_std(data_loader):
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in data_loader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_size

    mean /= total_images
    std /= total_images

    return mean, std

def load_mnist():
    # 載入 MNIST 資料集
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 計算訓練集的平均值和標準差
    train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    mean, std = calculate_mean_std(train_loader)

    # 顯示計算得到的平均值和標準差
    #print(f"Mean: {mean}")
    #print(f"Std: {std}")

    # 新增正規化的轉換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 使用計算得到的平均值和標準差進行正規化
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def get_data_loaders(batch_size):
    train_dataset, test_dataset = load_mnist()

    # 載入 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    print("Loading MNIST dataset...")
    train_dataset, test_dataset = load_mnist()

    # 顯示資料集資訊
    print(f"Number of train dataset: {len(train_dataset)}")
    print(f"Number of test datset: {len(test_dataset)}")

    # 印出訓練集第一張圖片的尺寸
    sample_image, _ = train_dataset[0]
    print(f"Sample Image Shape: {sample_image.shape}")

if __name__ == "__main__":
    main()
