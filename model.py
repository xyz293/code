import torch 
import torch.nn as nn
import torchvision.transforms.functional as F
class doublecov2d(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(doublecov2d,self).__init__()
        #会执行 nn.Module 的构造函数，初始化模型的基本属性和功能

        self.conv=nn.Sequential(nn.Conv2d(in_channels,out_channels,3,padding=1),
                                nn.BatchNorm2d(out_channels),
                                # 批归一化

                                nn.ReLU(inplace=True),
                                # 激活函数，防止过拟合

                                nn.Conv2d(out_channels,out_channels,3,padding=1),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(inplace=True),
                                )
        #self.pool=nn.MaxPool2d(2,2)
        #最大池化，步长为2，核大小为2*2
    def forward(self,x):
        x=self.conv(x)
        #将x传入conv中，得到一个新的x
        return x
        #返回新的x
class unet(nn.Module):
    def __init__(self,in_channels=3,out_channels=1,features=[64,128,256,512,1024]):
        #该列表定义了每一层的输出通道数

        super(unet,self).__init__()
        self.ups=nn.ModuleList()
        self.downs=nn.ModuleList()
        self.pool=nn.MaxPool2d(2,2)
        #最大池化，步长为2，核大小为2*2
        for feature in features:
            self.downs.append(doublecov2d(in_channels,feature))
            #将输入的通道数和输出的通道数传入doublecov2d类中，得到一个ublecov2d类的实例
            in_channels=feature
            #将输入的通道数更新为输出的通道数，以便下一次循环
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=3, stride=1,padding=1))
            #输入通道数，通常是当前层的通道数加上跳跃连接的通道数（即 feature * 2）。
            #输出通道数，表示经过反卷积后减少通道数
            self.ups.append(doublecov2d(feature*2,feature))
        self.bottleneck=doublecov2d(features[-1],features[-1]*2)#瓶颈层
        #将最后一层的输出通道数乘以2，作为bottleneck的输入通道数

        self.final_conv=nn.Conv2d(features[0],out_channels,kernel_size=1)#最终输出层

        #将第一个层的输出通道数作为final_conv的输入通道数，输出通道数为out_channels，核大小为1*1，表示输出一个像素点的值
    def forward(self,x):
        skip_connections=[]
        for down in self.downs:
            x=down(x)
            skip_connections.append(x)

            #将每一层的输出保存到skip_connections列表中
            x=self.pool(x)

            #最大池化，步长为2，核大小为2*2
        x=self.bottleneck(x)
            #将x传入bottleneck中，得到一个新的x
        skip_connections=skip_connections[::-1]
        for index in range(0,len(self.ups),2):
            #在 UNet 的解码器部分，通常会交替使用上采样层和卷积层。
            #步长为 2 的目的是跳过某些层（如卷积层），直接处理上采样层。

            x=self.ups[index](x)
            skip_connection=skip_connections[index//2]
            if skip_connection.shape!=x.shape:
                #如果x的形状和skip_connections的形状不一样，就进行裁剪
                x = F.resize(x, size=skip_connection.shape[2:])
                #将x的形状调整为skip_connections的形状
            concat=torch.cat((x,skip_connection),dim=1)
            #将x和skip_connections在通道维度上进行拼接，得到一个新的x
            x=self.ups[index+1](concat)
            #将x传入ups[index+1]中，得到一个新的x
        return self.final_conv(x)
def test():
    x=torch.randn((3,3,160,160))
    model=unet(3,1)
    preds=model(x)
    print(x.shape)
    print(preds.shape)

if __name__=="__main__":
    test()
    #测试模型的输入和输出形状是否正确
    #如果正确，就会输出输入和输出的形状