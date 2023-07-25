from statistics import mode
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# from .resnet_model import *

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual


class CTLFM(nn.Module):
    def __init__(self, inc):
        super(CTLFM, self).__init__()
        self.conv_jc0 = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.bn_jc0 = nn.BatchNorm2d(inc)
        self.conv_jc1_1 = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.bn_jc1_1 = nn.BatchNorm2d(inc)
        self.conv_jc1_2 = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.bn_jc1_2 = nn.BatchNorm2d(inc)
        self.conv_jc2 = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.bn_jc2 = nn.BatchNorm2d(inc)

        self.conv_jd = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.bn_jd = nn.BatchNorm2d(inc)
        self.conv_fusion = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.bn_fusion = nn.BatchNorm2d(inc)

        
    def forward(self, feat1, feat2, pred=None): ##pred相当于mask
        feat_jc = feat1 * feat2
        if pred is not None:
            feat_jc = feat_jc * pred
        feat_jc = F.relu(self.bn_jc0(self.conv_jc0(feat_jc)))
        feat_jc1 = F.relu(self.bn_jc1_1(self.conv_jc1_1(feat1+feat_jc)))
        feat_jc2 = F.relu(self.bn_jc1_2(self.conv_jc1_2(feat2+feat_jc)))
        feat_jc = F.relu(self.bn_jc2(self.conv_jc2(feat_jc1+feat_jc2)))
        
        feat_jd = torch.abs(feat1 - feat2)
        if pred is not None:
            feat_jd = feat_jd * pred
        feat_jd = F.relu(self.bn_jd(self.conv_jd(feat_jd)))
        feat_fusion = F.relu(self.bn_fusion(self.conv_fusion(feat_jd+feat_jc)))
        return feat_fusion





class MSCDNet_v1(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(MSCDNet_v1,self).__init__()

        resnet = models.resnet34(pretrained=True)
        ## -------------Encoder--------------
        # inputs 64*256*256
        self.inconv = nn.Conv2d(n_channels,64,3,padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        self.encoder0 = resnet.conv1 #64*256*256
        #stage 1
        self.encoder1 = resnet.layer1 #64*256*256
        #stage 2
        self.encoder2 = resnet.layer2 #128*128*128
        #stage 3
        self.encoder3 = resnet.layer3 #256*64*64
        #stage 4
        self.encoder4 = resnet.layer4 #512*32*32
        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 5
        self.resb5_1 = BasicBlock(512,512)
        self.resb5_2 = BasicBlock(512,512)
        self.resb5_3 = BasicBlock(512,512) #16

        # -------------Difference Fusion----------------
        self.fusion5 = CTLFM(512)
        self.fusion4 = CTLFM(512)
        self.fusion3 = CTLFM(256)
        self.fusion2 = CTLFM(128)
        self.fusion1 = CTLFM(64 )
        # self.fusion0 = CTLFM(64 )

        # -------------multi-level Fusion--------------
        self.mfusion4 = CTLFM(512)
        self.mfusion3 = CTLFM(256)
        self.mfusion2 = CTLFM(128)
        self.mfusion1 = CTLFM(64 )

        # 这里可以加一个ASPP或者多尺度特征提取模块
        # ---------------Decoder-----------------------
        # decoder stage 5
        self.conv5d_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512, 512, kernel_size=3, padding=1)###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        # decoder stage 4
        self.conv4d_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512, 512, kernel_size=3, padding=1)###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(512)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # decoder stage 3
        self.conv3d_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256, 256, kernel_size=3, padding=1)###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(256)
        self.relu3d_2 = nn.ReLU(inplace=True)

        # decoder stage 2
        self.conv2d_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128, 128, kernel_size=3, padding=1)###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(128)
        self.relu2d_2 = nn.ReLU(inplace=True)

        # decoder stage 1
        self.conv1d_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64, 64, kernel_size=3, padding=1)###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        # multi_level fusion stage4
        self.conv4f_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4f_1 = nn.BatchNorm2d(512)
        self.relu4f_1 = nn.ReLU(inplace=True)

        self.conv4f_m = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4f_m = nn.BatchNorm2d(512)
        self.relu4f_m = nn.ReLU(inplace=True)

        self.conv4f_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn4f_2 = nn.BatchNorm2d(256)
        self.relu4f_2 = nn.ReLU(inplace=True)

        # multi_level fusion stage3
        self.conv3f_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3f_1 = nn.BatchNorm2d(256)
        self.relu3f_1 = nn.ReLU(inplace=True)

        self.conv3f_m = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3f_m = nn.BatchNorm2d(256)
        self.relu3f_m = nn.ReLU(inplace=True)

        self.conv3f_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3f_2 = nn.BatchNorm2d(128)
        self.relu3f_2 = nn.ReLU(inplace=True)

        # multi_level fusion stage2
        self.conv2f_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2f_1 = nn.BatchNorm2d(128)
        self.relu2f_1 = nn.ReLU(inplace=True)

        self.conv2f_m = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2f_m = nn.BatchNorm2d(128)
        self.relu2f_m = nn.ReLU(inplace=True)

        self.conv2f_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2f_2 = nn.BatchNorm2d(64)
        self.relu2f_2 = nn.ReLU(inplace=True)

        # multi_level fusion stage2
        self.conv1f_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1f_1 = nn.BatchNorm2d(64)
        self.relu1f_1 = nn.ReLU(inplace=True)

        self.conv1f_m = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1f_m = nn.BatchNorm2d(64)
        self.relu1f_m = nn.ReLU(inplace=True)

        self.conv1f_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1f_2 = nn.BatchNorm2d(64)
        self.relu1f_2 = nn.ReLU(inplace=True)

        # ---------------------bilinear upsampling
        self.upscore5 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor= 16, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor= 8, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor= 2, mode='bilinear')
        self.upscore1 = nn.Upsample(scale_factor=4, mode='bilinear')

        # ---------------------Side output-----------------------
        self.outconv5 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        self.outconv4 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.outconv2 = nn.Conv2d(64 , 1, kernel_size=3, padding=1)
        self.outconv1 = nn.Conv2d(64 , 1, kernel_size=3, padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1,64)


    def forward(self,x, x2):

        hx = x
        hx2 = x2

        ## -------------Encoder-------------
        # hx = self.inconv(hx)
        # hx = self.inbn(hx)
        # hx = self.inrelu(hx)
        hx = self.encoder0(hx) # 64
        # layer1
        h1 = self.encoder1(hx) # 256
        # layer2
        h2 = self.encoder2(h1) # 128
        # layer3
        h3 = self.encoder3(h2) # 64
        # layer4
        h4 = self.encoder4(h3) # 32

        # layer5
        hx = self.pool4(h4)
        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        # hx2 = self.inconv(hx2)
        # hx2 = self.inbn(hx2)
        # hx2 = self.inrelu(hx2)
        hx2 = self.encoder0(hx2) # 64

        # layer1
        h1_2 = self.encoder1(hx2) # 256
        # layer2
        h2_2 = self.encoder2(h1_2) # 128
        # layer3
        h3_2 = self.encoder3(h2_2) # 64
        # layer4
        h4_2 = self.encoder4(h3_2) # 32

        # layer5
        hx2 = self.pool4(h4_2)
        hx2 = self.resb5_1(hx2)
        hx2 = self.resb5_2(hx2)
        h5_2 = self.resb5_3(hx2)

        # -----------Feature Fusion---------------
        hx = self.fusion5(h5, h5_2)
        feat_s5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))
        out_s5 = self.outconv5(feat_s5)

        hx = self.upscore2(out_s5)
        hx = self.fusion4(h4, h4_2, hx)
        feat_s4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.mfusion4(self.upscore2(feat_s5), feat_s4, self.upscore2(out_s5))
        feat_f4 = self.relu4f_2(self.bn4f_2(self.conv4f_2(hx)))
        out_s4 = self.outconv4(feat_f4)

        hx = self.upscore2(out_s4)
        hx = self.fusion3(h3, h3_2, hx)
        feat_s3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.mfusion3(self.upscore2(feat_f4), feat_s3, self.upscore2(out_s4))
        feat_f3 = self.relu3f_2(self.bn3f_2(self.conv3f_2(hx)))
        out_s3 = self.outconv3(feat_f3)
        
        hx = self.upscore2(out_s3)
        hx = self.fusion2(h2, h2_2, hx)
        feat_s2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.mfusion2(self.upscore2(feat_f3), feat_s2, self.upscore2(out_s3))
        feat_f2 = self.relu2f_2(self.bn2f_2(self.conv2f_2(hx)))
        out_s2 = self.outconv2(feat_f2)

        hx = self.upscore2(out_s2)
        hx = self.fusion1(h1, h1_2, hx)
        feat_s1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        hx = self.mfusion1(self.upscore2(feat_f2), feat_s1, self.upscore2(out_s2))
        feat_f1 = self.relu1f_2(self.bn1f_2(self.conv1f_2(hx)))
        out_s1 = self.outconv2(feat_f1)

        # --------------Bilinear upsampling------------
        S5 = self.upscore5(out_s5)
        S4 = self.upscore4(out_s4)
        # # return S4
        S3 = self.upscore3(out_s3)
        S2 = self.upscore1(out_s2)
        S1 = self.upscore2(out_s1)
        ## -------------Refine Module-------------
        dout = self.refunet(S1) # 256
        return F.sigmoid(dout), F.sigmoid(S1), F.sigmoid(S2), F.sigmoid(S3), F.sigmoid(S4), F.sigmoid(S5)

class MSCDNet_v2(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(MSCDNet_v2,self).__init__()

        resnet = models.resnet34(pretrained=True)
        ## -------------Encoder--------------
        # inputs 64*256*256
        self.inconv = nn.Conv2d(n_channels,64,3,padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)
        #stage 1
        self.encoder1 = resnet.layer1 #64*256*256
        #stage 2
        self.encoder2 = resnet.layer2 #128*128*128
        #stage 3
        self.encoder3 = resnet.layer3 #256*64*64
        #stage 4
        self.encoder4 = resnet.layer4 #512*32*32
        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 5
        self.resb5_1 = BasicBlock(512,512)
        self.resb5_2 = BasicBlock(512,512)
        self.resb5_3 = BasicBlock(512,512) #16

        # -------------Difference Fusion----------------
        self.fusion5 = CTLFM(512)
        self.fusion4 = CTLFM(512)
        self.fusion3 = CTLFM(256)
        self.fusion2 = CTLFM(128)
        self.fusion1 = CTLFM(64 )
        # self.fusion0 = CTLFM(64 )

        # -------------multi-level Fusion--------------
        self.mfusion4 = CTLFM(512)
        self.mfusion3 = CTLFM(256)
        self.mfusion2 = CTLFM(128)
        self.mfusion1 = CTLFM(64 )

        # 这里可以加一个ASPP或者多尺度特征提取模块
        # ---------------Decoder-----------------------
        # decoder stage 5
        self.conv5d_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512, 512, kernel_size=3, padding=1)###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        # decoder stage 4
        self.conv4d_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512, 512, kernel_size=3, padding=1)###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(512)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # decoder stage 3
        self.conv3d_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256, 256, kernel_size=3, padding=1)###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(256)
        self.relu3d_2 = nn.ReLU(inplace=True)

        # decoder stage 2
        self.conv2d_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128, 128, kernel_size=3, padding=1)###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(128)
        self.relu2d_2 = nn.ReLU(inplace=True)

        # decoder stage 1
        self.conv1d_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64, 64, kernel_size=3, padding=1)###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        # multi_level fusion stage4
        self.conv4f_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4f_1 = nn.BatchNorm2d(512)
        self.relu4f_1 = nn.ReLU(inplace=True)

        self.conv4f_m = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4f_m = nn.BatchNorm2d(512)
        self.relu4f_m = nn.ReLU(inplace=True)

        self.conv4f_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn4f_2 = nn.BatchNorm2d(256)
        self.relu4f_2 = nn.ReLU(inplace=True)

        # multi_level fusion stage3
        self.conv3f_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3f_1 = nn.BatchNorm2d(256)
        self.relu3f_1 = nn.ReLU(inplace=True)

        self.conv3f_m = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3f_m = nn.BatchNorm2d(256)
        self.relu3f_m = nn.ReLU(inplace=True)

        self.conv3f_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3f_2 = nn.BatchNorm2d(128)
        self.relu3f_2 = nn.ReLU(inplace=True)

        # multi_level fusion stage2
        self.conv2f_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2f_1 = nn.BatchNorm2d(128)
        self.relu2f_1 = nn.ReLU(inplace=True)

        self.conv2f_m = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2f_m = nn.BatchNorm2d(128)
        self.relu2f_m = nn.ReLU(inplace=True)

        self.conv2f_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2f_2 = nn.BatchNorm2d(64)
        self.relu2f_2 = nn.ReLU(inplace=True)

        # multi_level fusion stage2
        self.conv1f_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1f_1 = nn.BatchNorm2d(64)
        self.relu1f_1 = nn.ReLU(inplace=True)

        self.conv1f_m = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1f_m = nn.BatchNorm2d(64)
        self.relu1f_m = nn.ReLU(inplace=True)

        self.conv1f_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1f_2 = nn.BatchNorm2d(64)
        self.relu1f_2 = nn.ReLU(inplace=True)

        # ---------------------bilinear upsampling
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor= 8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor= 4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor= 2, mode='bilinear')
        # self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')

        # ---------------------Side output-----------------------
        self.outconv5 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        self.outconv4 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.outconv2 = nn.Conv2d(64 , 1, kernel_size=3, padding=1)
        self.outconv1 = nn.Conv2d(64 , 1, kernel_size=3, padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1,64)


    def forward(self,x, x2):

        hx = x
        hx2 = x2

        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        # layer1
        h1 = self.encoder1(hx) # 256
        # layer2
        h2 = self.encoder2(h1) # 128
        # layer3
        h3 = self.encoder3(h2) # 64
        # layer4
        h4 = self.encoder4(h3) # 32

        # layer5
        hx = self.pool4(h4)
        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx2 = self.inconv(hx2)
        hx2 = self.inbn(hx2)
        hx2 = self.inrelu(hx2)

        # layer1
        h1_2 = self.encoder1(hx2) # 256
        # layer2
        h2_2 = self.encoder2(h1_2) # 128
        # layer3
        h3_2 = self.encoder3(h2_2) # 64
        # layer4
        h4_2 = self.encoder4(h3_2) # 32

        # layer5
        hx2 = self.pool4(h4_2)
        hx2 = self.resb5_1(hx2)
        hx2 = self.resb5_2(hx2)
        h5_2 = self.resb5_3(hx2)

        # -----------Feature Fusion---------------
        hx = self.fusion5(h5, h5_2)
        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(hx)))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        feat_s5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))
        out_s5 = self.outconv5(feat_s5)

        hx = self.upscore2(out_s5)
        hx = self.fusion4(h4, h4_2, hx)
        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(hx)))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        feat_s4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.mfusion4(self.upscore2(feat_s5), feat_s4, self.upscore2(out_s5))
        hx = self.relu4f_1(self.bn4f_1(self.conv4f_1(hx)))
        hx = self.relu4f_m(self.bn4f_m(self.conv4f_m(hx)))
        feat_f4 = self.relu4f_2(self.bn4f_2(self.conv4f_2(hx)))
        out_s4 = self.outconv4(feat_f4)

        hx = self.upscore2(out_s4)
        hx = self.fusion3(h3, h3_2, hx)
        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(hx)))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        feat_s3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.mfusion3(self.upscore2(feat_f4), feat_s3, self.upscore2(out_s4))
        hx = self.relu3f_1(self.bn3f_1(self.conv3f_1(hx)))
        hx = self.relu3f_m(self.bn3f_m(self.conv3f_m(hx)))
        feat_f3 = self.relu3f_2(self.bn3f_2(self.conv3f_2(hx)))
        out_s3 = self.outconv3(feat_f3)
        
        hx = self.upscore2(out_s3)
        hx = self.fusion2(h2, h2_2, hx)
        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(hx)))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        feat_s2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.mfusion2(self.upscore2(feat_f3), feat_s2, self.upscore2(out_s3))
        hx = self.relu2f_1(self.bn2f_1(self.conv2f_1(hx)))
        hx = self.relu2f_m(self.bn2f_m(self.conv2f_m(hx)))
        feat_f2 = self.relu2f_2(self.bn2f_2(self.conv2f_2(hx)))
        out_s2 = self.outconv2(feat_f2)

        hx = self.upscore2(out_s2)
        hx = self.fusion1(h1, h1_2, hx)
        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(hx)))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        feat_s1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        hx = self.mfusion1(self.upscore2(feat_f2), feat_s1, self.upscore2(out_s2))
        hx = self.relu1f_1(self.bn1f_1(self.conv1f_1(hx)))
        hx = self.relu1f_m(self.bn1f_m(self.conv1f_m(hx)))
        feat_f1 = self.relu1f_2(self.bn1f_2(self.conv1f_2(hx)))
        out_s1 = self.outconv2(feat_f1)

        # --------------Bilinear upsampling------------
        S5 = self.upscore5(out_s5)
        S4 = self.upscore4(out_s4)
        # # return S4
        S3 = self.upscore3(out_s3)
        S2 = self.upscore2(out_s2)
        S1 = out_s1
        ## -------------Refine Module-------------
        dout = self.refunet(S1) # 256

        return F.sigmoid(dout), F.sigmoid(S1), F.sigmoid(S2), F.sigmoid(S3), F.sigmoid(S4), F.sigmoid(S5)



