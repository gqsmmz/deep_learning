## 数据集

数据集是[CUB-200-2011]( https://data.caltech.edu/records/65de6-vp158)，包含属于鸟类的 200 个子类别的 11,788 张图像，其中 5,994 张用于训练，5,794 张用于测试。datasets.py包含对原始数据的处理，download_and_extract 是数据集压缩包的下载和解压、load_dataset 是数据预处理、create_data_loaders 进一步将数据分为训练集 (5994 张) 和测试集 (5794 张) 并按设定的 batch_size 打包成 train_loader 和 val_loader。最终把两个loader 以字典的形式储存到”./dataloaders/” 文件夹下，以 dataloader_bs_batch_size.pkl 的形式命名。

## 模型：

model.py文件中initialize_model 函数：构建了 AlexNet、ResNet-18 的网格架构，且读出倒数第二层的输出维数，重新构建最后的线性层。

1）pretrained 参数初始化

• 代码位置：train.py 和 grid_seach.py 中初始化模型后对参数做初始化。

• 代码解读：初始化模型时采用 use_pretrained = T rue，初始化后对最后的线性层做随机初始
化，其余层不做处理。

2）随机参数初始化

• 代码位置：train.py 和 gridseach.py 中初始化模型后对参数做初始化。

• 代码解读：初始化模型时采用 use_pretrained = F alse，得到初始化模型后再对所有参数做
随机初始化。

## 代码运行相关

1. utils.py：构建 dataloader、保存文件等用到的额外操作函数
2. train.py：pretrained/random 参数初始化的模型训练和测试，以及其结果可视化和保存
3. model_and_search.py：包含模型初始化、模型训练、网格搜索的基本函数设计
4. grid_search.py 是 pretrained/random 参数初始化的网格搜索，以及结果可视化和保存

运行方法：

1. 运行 train.py: 修改 train.py 的超参数，仅对某超参数组合做训练和测试
2. 运行 grid_search.py：修改超参数组合，在不同超参数组合中选择做训练和测试，并得出最佳超参数组合

## 训练和测试

train.py：此文件仅训练某超参数组合，需要指定task_type是"pretrained_model"还是"random_model"。前者对最后一层的学习率为lr对其余层的学习率为lr/10。后者对所有层都用学习率lr。

  计算损失采用交叉熵形式：criterion = nn.CrossEntropyLoss()

  tensorboard可视化结果、对应模型权重文件输出在：runs/pretrained_model和runs/random_model.


## 网格搜索

grid_search.py：训练测试几组指定的超参数组合，需要指定task_type是"pretrained_model"还是"random_model"。学习率设计同train.py。多组超参数的遍历采用for循环，且不同batch_size下读取对应不同的dataloader的pkl数据文件，且每组超参数训练测试后返回验证准确率，比较得出所有组合中准确率最高的组合。

  tensorboard可视化结果、对应模型权重文件输出在：seach/pretrained1/bs\_\{bs\}\_lr\_\{lr\}\_epoch\_nums\_\{num\_epochs\}、seach/random1/bs\_\{bs\}\_lr\_\{lr\}\_epoch\_nums\_\{num\_epochs\}
