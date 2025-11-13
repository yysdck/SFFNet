import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from .MDAF import MDAF
from .FMS import FMS



class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )



class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

class WS(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WS, self).__init__()
        self.pre_conv = Conv(in_channels, in_channels, kernel_size=1)
        self.pre_conv2 = Conv(in_channels, in_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(in_channels, decode_channels, kernel_size=3)

    def forward(self, x, res,ade):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x + fuse_weights[2]*ade
        x = self.post_conv(x)
        return x



class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat




class SFFNet(nn.Module):
    def __init__(self,
                 decode_channels=96,
                 dropout=0.1,
                 # backbone_name="convnextv2_base.fcmae_ft_in22k_in1k_384",
                 backbone_name="convnext_tiny.in12k_ft_in1k_384",
                 pretrained=True,
                 window_size=8,
                 num_classes=6,
                 use_aux_loss = True
                 ):
        super().__init__()
        self.use_aux_loss = use_aux_loss
        self.backbone = timm.create_model(model_name=backbone_name, features_only=True,pretrained=pretrained, output_stride=32, out_indices=(0, 1, 2,3))

        self.conv2 = ConvBN(192,decode_channels,kernel_size=1)
        self.conv3 = ConvBN(384, decode_channels, kernel_size=1)
        self.conv4 = ConvBN(768, decode_channels, kernel_size=1)

        self.MDAF_L = MDAF(decode_channels,num_heads=8,LayerNorm_type = 'WithBias')
        self.MDAF_H = MDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.fuseFeature = FMS(in_ch=3*decode_channels, out_ch=decode_channels,num_heads=8,window_size=window_size)
        self.WF1 = WF(in_channels=decode_channels,decode_channels=decode_channels)
        self.WF2 = WF(in_channels=decode_channels,decode_channels=decode_channels)


        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.down = Conv(in_channels=3*decode_channels,out_channels=decode_channels,kernel_size=1)
    def forward(self, x,imagename=None):
        b = x.size()[0]
        h, w = x.size()[-2:]

        res1,res2,res3,res4 = self.backbone(x)
        res1h,res1w = res1.size()[-2:]

        res2 = self.conv2(res2)
        res3 = self.conv3(res3)
        res4 = self.conv4(res4)
        res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
        middleres =torch.cat([res2,res3,res4],dim=1)

        fusefeature_L,fusefeature_H,glb,local = self.fuseFeature(middleres,imagename)
        glb = self.MDAF_L(fusefeature_L,glb)
        local = self.MDAF_H(fusefeature_H,local)


        res  = self.WF1(glb,local)

        middleres = self.down(middleres)
        res = F.interpolate(res,size=(res1h,res1w), mode='bicubic', align_corners=False)
        res = middleres + res
        res = self.WF2(res,res1)
        res = self.segmentation_head(res)

        if self.training:
            if self.use_aux_loss == True:
                x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
                return x
            else:
                x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
                return x
        else:
            x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
            return x