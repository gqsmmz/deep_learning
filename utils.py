import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

def save_model_params(params, file_path):
    np.savez(file_path, **params)

def train_collate_fn(batch):
    easy_transform = transforms.Compose([
        transforms.Resize((224, 224)),  #将图像缩放到 224x224 像素
        transforms.ToTensor(), #将图像从 PIL Image 或 numpy.ndarray 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #用 ImageNet 数据集的均值和标准差对图像进行标准化
    ])
    other_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图像。
        transforms.RandomRotation(15), #随机旋转图像，角度在 (-15, 15) 度之间
        easy_transform
    ])
    images, labels = zip(*[(other_transform(x), y) for x, y in batch]) #对每个批次中的图像 x 应用 other_transform 变换，并保持标签 y 不变。
    return default_collate(images), default_collate(labels)  #使用 PyTorch 的默认 collate 函数将图像列表打包成一个张量。

def val_collate_fn(batch):
    easy_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images, labels = zip(*[(easy_transform(x), y) for x, y in batch])
    return default_collate(images), default_collate(labels)

def plot_loss(loss_list, result_dir):
    loss_train = loss_list['loss_train']
    loss_val = loss_list['loss_val']
    val_accuracy = loss_list['val_accuracy']
    length = len(loss_train) + 1
    epochs = range(1, length)

    # 设置 seaborn 风格
    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 5))
    
    # 绘制训练和验证损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_train, label='Train Loss', color='blue')
    plt.plot(epochs, loss_val, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制验证准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'training_plot.png'))
    plt.close()