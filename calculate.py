import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_and_search import initialize_model,train_model
import os
import pickle

# 计算准确率
def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 加载模型和权重
model_name = "resnet18"
num_classes = 200
batch_size=16
model_path = "./search/pretrained1/bs_16_lr_0.013_epoch_nums_60/model_parameters.pth"  #mid_term/search/random1/bs_16_lr_0.005_epoch_nums_40/model_parameters.pth
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

 ##读取对应batch的dataloader
bs_dir='./dataloaders/dataloader_bs_'+str(batch_size)+'.pkl'
if os.path.exists(bs_dir):
    with open(bs_dir, 'rb') as f:
        dataloaders = pickle.load(f)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

model = initialize_model(model_name, num_classes, use_pretrained=False)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# 计算并输出验证准确率
val_accuracy = calculate_accuracy(model, val_loader, device)
print(f'Validation Accuracy: {val_accuracy:.2f}%')
