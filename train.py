import torch 
import torch.nn as nn
import os
import torch.optim as optim
import albumentations as A
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import tqdm as  tqdm
from tqdm import tqdm
from model import unet
#Unet是一个类，表示一个模型
import numpy as np
import random
from utility import get_loaders
learning_rate=1e-8
batch_size=8
num_epochs=5
num_workers=0
pin_memory=False
load_model=False
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)
train_losses=[]
val_acc=[]
    #设置训练和验证的损失值列表
image_height=160
image_width=240
seed=random.randint(0,100)
torch.manual_seed(seed)     
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
    #设置随机种子，保证每次运行的结果一样   
def train_fn(loader,model,optimizer,loss_fn,scaler,device):
    #tqdm是一个进度条库，leave=True表示在训练结束后不清除进度条
    total_loss=0
    for (inputs,label) in tqdm(loader, desc="Training", leave=False):
        inputs = inputs.to(device)
        label=label.unsqueeze(1).float().to(device)
        with torch.cuda.amp.autocast():

            pred=model(inputs)
            loss=loss_fn(pred,label)
            #pred是模型的输出，targets是标签
            optimizer.zero_grad()
            #将梯度清零
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss+=loss.item()
            #将损失值添加到列表中
    
    return total_loss/len(loader)
    #返回平均损失值  

def check_accuracy(model,loader,device='cuda'):
    num_correct=0
    num_pixels=0
    dice_score=0
    #dice_score是一个评价指标，表示模型的准确率
    model.eval()
    #将模型设置为评估模式
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            y=y.unsqueeze(1).to(device)
            pred=torch.sigmoid(model(x))

            pred=(pred>0.5).float()
            #将预测值大于0.5的值设为1，小于0.5的值设为0
            num_correct+=(pred==y).sum()
            num_pixels+=torch.numel(pred)
            #计算预测值和标签值相等的像素点的数量

            dice_score+=(2*(pred*y).sum())/(2*(pred+y).sum()+((pred*y)<1).sum())
            #计算dice_score的值
    accuracy=round(float(num_correct/num_pixels),4)
    
    print(f'准确率为{accuracy}')
    model.train()
    return accuracy

def main():
    train_transform=A.Compose(
        [
            A.Resize(image_height,image_width),
            A.Rotate(limit=35, p=1.0),#进行旋转操作，最大角度是35度
            A.HorizontalFlip(p=0.5),#水平翻转
            A.VerticalFlip(p=0.5),#竖直翻转
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),#转化成向量
        ])#数据增强
    val_transform=A.Compose([
            A.Resize(image_height,image_width),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),#转化成向量 
    ])
    #目前还缺少数据集的路径

    #训练集和验证集的路径
    train_img_dir=r'annotations\training'  
    train_mask_dir=r'annotations\train__'
    val_img_dir=r'images\val_'
    val_mask_dir=r'images\validation'
    train_loader,val_loader=get_loaders(train_img_dir,train_mask_dir,val_img_dir,val_mask_dir,train_transform,val_transform,batch_size)
    model=unet(in_channels=3,out_channels=1).to(device)
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    #优化器

    loss_fn=nn.BCEWithLogitsLoss()

    #损失函数，二分类交叉熵损失函数，并且使用logits=True，因为输出是未经过sigmoid激活的

    scaler=torch.cuda.amp.GradScaler()
    #混合精度训练
    for i in range(num_epochs):
        print(f'训练第{i+1}次'.format(i+1))
        train_loss=train_fn(train_loader,model,optimizer,loss_fn,scaler,device)
    #训练损失值
        train_losses.append(train_loss)
    #将训练损失值添加到列表中
        accuracy=check_accuracy(model,val_loader,device)
        val_acc.append(accuracy)
    torch.save(model.state_dict(),'model.pth')
    #验证模型的准确率
if __name__=="__main__":
    main()
    #运行主函数
        
    

