import torch
import torch.nn as nn
class Bconv(nn.Module):
    def __init__(self,ch_in,ch_out,k,s):

        super(Bconv, self).__init__()
        self.conv=nn.Conv2d(ch_in,ch_out,k,s,padding=k//2)
        self.bn=nn.BatchNorm2d(ch_out)
        self.act=nn.SiLU()
    def forward(self,x):

        return self.act(self.bn(self.conv(x)))
class SppCSPC(nn.Module):
    def __init__(self,ch_in,ch_out):

        super(SppCSPC, self).__init__()

        self.conv1=nn.Sequential(
            Bconv(ch_in,ch_out,1,1),
            Bconv(ch_out,ch_out,3,1),
            Bconv(ch_out,ch_out,1,1)
        )

        self.mp1=nn.MaxPool2d(5,1,5//2) 
        self.mp2=nn.MaxPool2d(9,1,9//2) 
        self.mp3=nn.MaxPool2d(13,1,13//2) 


        self.conv1_2=nn.Sequential(
            Bconv(4*ch_out,ch_out,1,1),
            Bconv(ch_out,ch_out,3,1)
        )



        self.conv3=Bconv(ch_in,ch_out,1,1)

        self.conv4=Bconv(2*ch_out,ch_out,1,1)
    def forward(self,x):

        output1=self.conv1(x)


        mp_output1=self.mp1(output1)
        mp_output2=self.mp2(output1)
        mp_output3=self.mp3(output1)


        result1=self.conv1_2(torch.cat((output1,mp_output1,mp_output2,mp_output3),dim=1))

        result2=self.conv3(x)

        return self.conv4(torch.cat((result1,result2),dim=1))

