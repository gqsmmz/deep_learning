import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils import *  
from model_and_search import initialize_model,train_model, hyperparameter_search
import os
import pickle


def train(task_type="pretrained_model"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('./dataloaders/dataloader_bs_16.pkl', 'rb') as f:  #两者都是16的batch size比较好
        dataloaders = pickle.load(f)  
    pretrained_params = {'lr': 0.013, 'num_epochs': 60}
    random_params = {'lr': 0.005, 'num_epochs': 40}

    if task_type=="pretrained_model":
        ### 读入预训练参数，fc层参数作初始化
        model_pretrained = initialize_model("resnet18", 200, use_pretrained=True)  
        model_pretrained = model_pretrained.to(device)
        nn.init.kaiming_normal_(model_pretrained.fc.weight)
        if model_pretrained.fc.bias is not None:
            nn.init.constant_(model_pretrained.fc.bias, 0)
            

        # fc层参数以lr训练、其余层以较小lr训练。
        optimizer_pretrained = optim.SGD([
            {'params': model_pretrained.fc.parameters(), 'lr': pretrained_params['lr']},  # 最后一层，网络参数从0开始。
            {'params': (p for n, p in model_pretrained.named_parameters() if 'fc' not in n), 'lr': pretrained_params['lr'] / 10}  # 其他所有层
        ], momentum=0.9, weight_decay=1e-3)

        criterion = nn.CrossEntropyLoss()
        pretrained_model = train_model(model_pretrained, dataloaders, criterion, optimizer_pretrained, num_epochs=pretrained_params['num_epochs'],device=device, task_type=task_type)
    
    elif task_type=="random_model":
        # 随机初始化所有参数
        model_random= initialize_model("resnet18", 200, use_pretrained=False)
        model_random = model_random.to(device)
        def initialize_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        model_random.apply(initialize_weights)


        # 设置优化器
        optimizer_random = optim.SGD(model_random.parameters(), lr=random_params['lr'], momentum=0.9, weight_decay=1e-3)  #model参数随机化

        criterion = nn.CrossEntropyLoss()
        random_model = train_model(model_random, dataloaders, criterion, optimizer_random, num_epochs=random_params['num_epochs'],device=device, task_type=task_type)


if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train("pretrained_model")
    train("random_model")



