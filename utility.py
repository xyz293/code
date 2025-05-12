
from cdataset import Cdata
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from cdataset import Cdata

def get_loaders(train_img_dir,train_mask_dir,val_img_dir,val_mask_dir,train_transform,val_transform,batch_size):

    #创建数据集对象
    train_ds=Cdata(train_img_dir,train_mask_dir,train_transform)
    #训练集数据集对象
    val_ds=Cdata(val_img_dir,val_mask_dir,val_transform)
    train_loader=DataLoader(train_ds,batch_size=batch_size)
    #DataLoader 内部会自动调用数据集的 __len__ 和 __getitem__ 方法

    val_loader=DataLoader(val_ds,batch_size=batch_size)

    #创建数据加载器对象
    return train_loader,val_loader
    #返回训练集和验证集的数据加载器对象