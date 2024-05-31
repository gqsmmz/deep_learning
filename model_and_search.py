import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import pickle
from copy import deepcopy
from torchvision.models import ResNet18_Weights, AlexNet_Weights
from torch.utils.tensorboard import SummaryWriter

def save_model(model, task_type):
    ##pth文件下载路径
    file_name = f"./runs/{task_type}/{task_type}.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")

def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    # 可用的模型字典，模型名映射到相应的类
    model_dict = {
        "alexnet": models.alexnet,
        "resnet18": models.resnet18,
        # 添加其他模型
    }

    # 确保所选模型在字典中
    if model_name not in model_dict:
        raise ValueError(f"Invalid model name. Choose from: {list(model_dict.keys())}")

    # 加载所选模型
    model = model_dict[model_name](pretrained=use_pretrained)

    # 修改模型的输出层大小以适应新的类别数量
    if isinstance(model, models.AlexNet):
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif isinstance(model, models.ResNet):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError("Unsupported model architecture. Only AlexNet and ResNet are supported.")

    # 冻结模型参数（特征提取）或保持所有参数可训练
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device="cuda" if torch.cuda.is_available() else "cpu", task_type="pretrained_model"):
    print("num", num_epochs)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    model.to(device)
    model.train()
    
    if task_type == "pretrained_model":
        writer = SummaryWriter('runs/pretrained_model1')
    elif task_type == "random_model":
        writer = SummaryWriter('runs/random_model1')
    else:
        writer = SummaryWriter(task_type)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                loader = train_loader
                model.train()  # Set model to training mode
            else:
                loader = val_loader
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # TensorBoard logging
            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)  #训练集和验证集上的损失和准确率
                writer.add_scalar('Acc/train', epoch_acc, epoch)
            else:
                writer.add_scalar('Loss/val', epoch_loss, epoch)
                writer.add_scalar('Acc/val', epoch_acc, epoch)
                
    writer.close()
    
    # 根据 task_type 返回不同的结果
    if task_type not in ["pretrained_model", "random_model"]:
        return model, epoch_acc
    else:
        # 保存文件
        save_model(model, task_type)
    
    return model

def hyperparameter_search(model, device, batch_size_list,num_epochs_list, lr_list,results_dir, use_pretrained=True,writer_type="pretrained/"):
    ##变量设置
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_params = {'lr': None, 'num_epochs': None, 'accuracy': 0}
    loss_list={'loss_train':[],'loss_val':[],'val_accuracy':[]}


    #网格搜索
    for bs in batch_size_list:

        ##读取对应batch的dataloader
        bs_dir='./dataloaders/dataloader_bs_'+str(bs)+'.pkl'
        if os.path.exists(bs_dir):
            with open(bs_dir, 'rb') as f:
                dataloaders = pickle.load(f)
            train_loader = dataloaders['train']
            val_loader = dataloaders['val']
    
        for lr in lr_list:
            for num_epochs in num_epochs_list:

                ##构建writer的路径
                dir='bs_'+str(bs)+'_lr_'+str(lr)+'_epoch_nums_'+str(num_epochs)
                writer_dir='search/'+writer_type+dir
                writer_search = SummaryWriter(writer_dir)
                search_dir=writer_dir
                os.makedirs(search_dir,exist_ok=True)
                print(f"Training with batch_size_list={bs}, lr={lr}, num_epochs={num_epochs}")


                model_copy = deepcopy(model).to(device)

                #以某学习率训练新的输出层，并对其余参数使用较小的学习率进行微调
                if use_pretrained:
                    optimizer = optim.SGD([
                        {'params': model_copy.fc.parameters(), 'lr': lr},
                        {'params': (p for n, p in model_copy.named_parameters() if 'fc' not in n), 'lr': lr / 10}
                    ], momentum=0.9, weight_decay=1e-3)
                else:
                    optimizer = optim.SGD(model_copy.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)

                #训练
                trained_model,epoch_acc = train_model(model_copy, dataloaders, criterion,optimizer, num_epochs,device=device,task_type=writer_dir)
                
                #保存每个参数组合下的模型权重
                save_dir=search_dir+'/model_parameters.pth'
                print(save_dir)
                torch.save(trained_model.state_dict(), save_dir)

                #更新最佳权重
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_params = {'bs':bs,'lr': lr, 'num_epochs': num_epochs, 'accuracy': epoch_acc}
                    print(f"最新最佳参数: bs={bs},lr={lr},num_epochs={num_epochs},accuracy={epoch_acc}")
       
    #输出最佳权重，方便检索
    print(f"Best Params: batch_size={best_params['bs']},lr={best_params['lr']}, num_epochs={best_params['num_epochs']}, Accuracy={best_params['accuracy']:.4f}")
    return best_params,best_acc
