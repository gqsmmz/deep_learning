import os
import requests
import tarfile
import pandas as pd
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader,Subset
from torchvision.datasets import ImageFolder
import pickle
from utils import *

# 下载并解压数据集
def download_and_extract(url, download_path, extract_path):
    if os.path.exists(download_path) and os.path.exists(extract_path):
        print(f"Files already downloaded and extracted.")
        return

    if not os.path.exists(download_path):
        print(f"Downloading {url}")
        response = requests.get(url, stream=True)
        with open(download_path, 'wb') as f:
            f.write(response.raw.read())

    if not os.path.exists(extract_path):
        print(f"Extracting {download_path}")
        with tarfile.open(download_path) as tar:
            tar.extractall(path="./")

# 读取并处理数据集
def load_dataset(data_root):
    image_dir = os.path.join(data_root, 'images')
    split_file = os.path.join(data_root, 'train_test_split.txt')
    images_file = os.path.join(data_root, 'images.txt')

    images_name = pd.read_csv(images_file, delim_whitespace=True, header=None, index_col=0)[1].to_dict()
    image_split = pd.read_csv(split_file, delim_whitespace=True, header=None, index_col=0)[1].to_dict()

    files = {}
    images_path = os.path.join(data_root, 'images')
    for id, path in images_name.items():
        full_path = os.path.normpath(os.path.join(images_path, path))
        files[full_path] = image_split[id]
    return files


# 创建数据加载器
def create_data_loaders(data_df, batch_size=32):

    real_data = ImageFolder(root=os.path.join('./CUB_200_2011', 'images'))

    train_indices = [i for i, (img_path, _) in enumerate(real_data.imgs)
                     if data_df[os.path.normpath(img_path)] == 1]
    val_indices = [i for i, (img_path, _) in enumerate(real_data.imgs)
                   if data_df[os.path.normpath(img_path)] == 0]
    #print('length', len(train_indices),len(val_indices))  #5994 5794

    train_dataset = Subset(real_data, train_indices)
    val_dataset = Subset(real_data, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,collate_fn=train_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,collate_fn=val_collate_fn)
    
    return train_loader, val_loader

def save_dataloaders(train_loader, val_loader, batch_size):
    os.makedirs(os.path.dirname('./dataloaders'), exist_ok=True)
    file_path = f"./dataloaders/dataloader_bs_{batch_size}.pkl"
    dataloaders = {'train': train_loader, 'val': val_loader}
    with open(file_path, 'wb') as f:
        pickle.dump(dataloaders, f)
    print(f"Dataloaders saved as {file_path}")

# 主函数
def load_batchsize_data(batch_size=32):
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    download_path = 'CUB_200_2011.tgz'
    extract_path = 'CUB_200_2011'
    data_root = 'CUB_200_2011'

    # 下载并解压数据集
    download_and_extract(url, download_path, extract_path)

    # 加载数据集
    data_df = load_dataset(data_root)


    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(data_df,batch_size)


    # 保存数据加载器
    save_dataloaders(train_loader, val_loader,batch_size)

    # 这里可以添加训练和验证模型的代码

if __name__ == '__main__':
    load_batchsize_data(16)
    load_batchsize_data(32)
    load_batchsize_data(64)
    # load_batchsize_data(128)
