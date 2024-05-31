import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from model_and_search import initialize_model,train_model, hyperparameter_search
import os
import pickle

def grid_search(task_type="pretrained_model",results_dir='./'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 超参数列表
    lr_list = [1.3e-2,2.2e-2,1.2e-2,2.1e-2,1e-2, 2e-2,3e-2]
    epoch_list = [25,40,50,60]
    batch_size=[64,32,16]


    if task_type=="pretrained_model":
        #初始化参数
        model = initialize_model("resnet18", 200, use_pretrained=True)  #model里储存着每一层的weights
        model = model.to(device)
        nn.init.kaiming_normal_(model.fc.weight)
        if model.fc.bias is not None:
            nn.init.constant_(model.fc.bias, 0)

        params_pretrained_best,best_val_accuracy = hyperparameter_search(model, device, batch_size,epoch_list, lr_list, results_dir,use_pretrained=True,writer_type="pretrained/")  
    
    elif task_type=="random_model":
        #初始化参数
        model= initialize_model("resnet18", 200, use_pretrained=False)  #直接将参数设置为none
        model = model.to(device)
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
        model.apply(initialize_weights)

        params_pretrained_best,best_val_accuracy = hyperparameter_search(model, device, batch_size,epoch_list, lr_list, results_dir,use_pretrained=False,writer_type="random/")

if __name__=='__main__':
    results_dir='./search/' 
    os.makedirs(results_dir,exist_ok=True)

    grid_search("pretrained_model",results_dir)
    grid_search("random_model",results_dir)
