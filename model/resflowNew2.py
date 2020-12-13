
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import numpy as np
# from splat import SplAtConv2d
from mod import CMDTop
from mod import OpticalFlowEstimatorNoDenseConnection, OpticalFlowEstimator, FeatureL2Norm, \
    CorrelationVolume, deconv, conv, predict_flow, unnormalise_and_convert_mapping_to_flow, warp
from consensus_network_modules import MutualMatching, NeighConsensus, FeatureCorrelation

import correlation # the custom cost volume layer
from torchsummary import summary
__all__ = ['Res2Net', 'res2net50']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_urls = {
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
}

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

def Conv2(in_planes, places):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=1,stride=1,bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=places,out_channels=2,kernel_size=1,stride=1, bias=False),
    )

def FlowConv(in_planes = 64 ,places=512):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=1,stride=1,bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
    )

def dilateconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, bias=True),
                            nn.BatchNorm2d(out_planes),
                            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=True),
                            nn.LeakyReLU(0.1))

class DeConvBlock(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False):
        super(DeConvBlock,self).__init__()
        self.downsampling = downsampling
        self.deConvNet = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_places,out_channels=in_places,kernel_size=2,stride=stride, bias=False),
            nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_places,out_channels=places,kernel_size=2,stride=stride, bias=False),
                nn.BatchNorm2d(places)
            )
        self.relu = nn.ReLU(inplace=True)
        # self.ca = ChannelAttention(places)
        # self.sa = SpatialAttention()
    def forward(self, x):
        residual = x
        out = self.deConvNet(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        # out = self.ca(out)*out
        # out = self.sa(out)*out
        return out

#CBAM 结构代码
#通道注意力模块

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
        # return out

#空间注意力模块

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, 
        dilation=1,radix=1, cardinality=1,downsample=None,
        rectified_conv=False, rectify_avg=False,
        norm_layer=None, dropblock_prob=0.0, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality.
            planes: output channel dimensionality.
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1.
            baseWidth: basic width of conv3x3.
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        # self.conv2 = SplAtConv2d(
        #         width, width, kernel_size=3,
        #         stride=stride, padding=dilation,
        #         dilation=dilation, groups=cardinality, bias=False,
        #         radix=radix, rectify=rectified_conv,
        #         rectify_avg=rectify_avg,
        #         norm_layer=norm_layer,
        #         dropblock_prob=dropblock_prob)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        # out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.cyclic_consistency=True
        self.consensus_network = False
        self.batch_norm = True
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.convflow1 = FlowConv(in_planes = 4096 ,places=64)
        self.convflow2 = FlowConv(in_planes = 1024 ,places=128)
        self.convflow3 = FlowConv(in_planes = 256 ,places=256)
        self.convflow4 = FlowConv(in_planes = 64 ,places=256)

        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, bias=False)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, 16, layers[0],stride=1)
        self.layer2 = self._make_layer(block, 64, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 64,layers[3], stride=2)
        self.layer5 = self.make_layer_deconv(768, 256, stride=2)
        # self.layer5 = self._make_layer(block, 1024, 128, layers[3], stride=1)
        self.layer6 = self.make_layer_deconv(512, 128, stride=2)
        self.layer7 = self.make_layer_deconv1(block, 256, 64, layers[4], stride=2)
        self.layer8 = self.make_layer_deconv1(block, 128, 32, layers[5], stride=2)
        # self.layer7 = self._make_layer(block, 512, 64, layers[4], stride=1)
        # self.layer9 = self.make_layer_deconv1(block, 128, 64, layers[4], stride=2)
        # self.layer9 = self._make_layer(block, 32, 16, layers[6], stride=1)
        self.conv2=Conv2(in_planes = 32 ,places=256)

        self.l_dc_conv1 = dilateconv(256, 256, kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=True)
        self.l_dc_conv2 = dilateconv(256, 256, kernel_size=3, stride=1, padding=2,  dilation=2, batch_norm=True)
        self.l_dc_conv3 = dilateconv(256, 256, kernel_size=3, stride=1, padding=4,  dilation=4, batch_norm=True)
        # self.l_dc_conv4 = dilateconv(256, 256, kernel_size=3, stride=1, padding=8,  dilation=8, batch_norm=True)

        # self.l_dc_conv1 = dilateconv(128, 128, kernel_size=3, stride=1, padding=1,  dilation=1, batch_norm=True)
        self.l_dc_conv11 = dilateconv(128, 128, kernel_size=3, stride=1, padding=2,  dilation=2, batch_norm=True)
        self.l_dc_conv22 = dilateconv(128, 128, kernel_size=3, stride=1, padding=4,  dilation=4, batch_norm=True)
        self.l_dc_conv33 = dilateconv(128, 128, kernel_size=3, stride=1, padding=8,  dilation=8, batch_norm=True)

        self.ca0 = ChannelAttention(256)
        self.sa0 = SpatialAttention()

        self.ca1 = ChannelAttention(32)
        self.sa1 = SpatialAttention()
        
        self.l2norm = FeatureL2Norm()
        if self.cyclic_consistency:
            self.corr = FeatureCorrelation(shape='4D', normalization=False)
        elif consensus_network:
            ncons_kernel_sizes = [3, 3, 3]
            ncons_channels = [10, 10, 1]
            self.corr = FeatureCorrelation(shape='4D', normalization=False)
            # normalisation is applied in code here
            self.NeighConsensus = NeighConsensus(use_cuda=True,
                                                 kernel_sizes=ncons_kernel_sizes,
                                                 channels=ncons_channels)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def make_layer_deconv(self, in_places, places, stride):
        layers = []
        layers.append(DeConvBlock(in_places, places,stride, downsampling =True))
        # for i in range(1, block):
        #     layers.append(Bottleneck(places, places//4))

        return nn.Sequential(*layers)

    def make_layer_deconv1(self, block, in_places, places, blocks, stride):
        layers = []
        layers.append(DeConvBlock(in_places, places,stride, downsampling =True))
        planes = places//4
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def coarsest_resolution_flow(self, c14, c24):
        # ratio_x = 16.0 / float(w_256)
        # ratio_y = 16.0 / float(h_256)
        # print(c14.shape,c24.shape)
        b = c24.shape[0]
        if self.cyclic_consistency:
            corr4d = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            # run match processing model
            corr4d = MutualMatching(corr4d)
            corr4 = corr4d.squeeze(1).view(b, c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])
        elif self.consensus_network:
            corr4d = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            # run match processing model
            corr4d = MutualMatching(corr4d)
            corr4d = self.NeighConsensus(corr4d)
            corr4d = MutualMatching(corr4d)  # size is [b, 1, hsource, wsource, htarget, wtarget]
            corr4 = corr4d.squeeze(1).view(c24.shape[0], c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])
        else:
            corr4 = self.corr(self.l2norm(c24), self.l2norm(c14))
        corr4 = self.l2norm(F.relu(corr4))

        return corr4

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x

    def forward(self, input1,input2):
        o11=self.forward_once(input1)
        o21=self.forward_once(input2)
        # print(o11.shape)
        # exit()
        o12=self.layer2(o11)
        o22=self.layer2(o21)
        # print(o12.shape)
        o13=self.layer3(o12)
        o23=self.layer3(o22)

        o14=self.layer4(o13)
        o24=self.layer4(o23)
        # print(o23.shape)
        

        flow1 = self.coarsest_resolution_flow(o11, o21)
        flow2 = self.coarsest_resolution_flow(o12, o22)
        flow3 = self.coarsest_resolution_flow(o13, o23)
        flow4 = self.coarsest_resolution_flow(o14, o24)
        
        flow1 = self.convflow1(flow1)
        flow2 = self.convflow2(flow2)
        flow3 = self.convflow3(flow3)
        flow4 = self.convflow4(flow4)
        # print(flow1.shape)
        # print(flow2.shape)
        # print(flow3.shape)
        # print(flow4.shape)

        out = torch.cat((o14,o24,flow4),dim=1)
        out = self.layer5(out)      
        out = self.l_dc_conv1(out)
        out = self.l_dc_conv2(out)
        out = self.l_dc_conv3(out)
        # out = self.l_dc_conv4(out)

        out = torch.cat((out,flow3),dim=1)
        # print(out.shape)
        out = self.layer6(out)
        out = self.l_dc_conv11(out)
        out = self.l_dc_conv22(out)
        out = self.l_dc_conv33(out)


        out = torch.cat((out,flow2),dim=1)
        # print(out.shape)
        out = self.ca0(out)*out
        out = self.sa0(out)*out
        out = self.layer7(out)

        out = torch.cat((out,flow1),dim=1)
        out = self.layer8(out)
        out = self.ca1(out)*out
        out = self.sa1(out)*out

        out = self.conv2(out)
        # print(out.shape)
        # exit()
        # out = self.layer4(out)
        # print(out.shape,flow3.shape)
        # out = torch.cat((out,flow3),dim=1)
        # print(out.shape)

        return out


def res2net50(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 4, 3,2], baseWidth = 26, scale = 4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model



if __name__ == '__main__':
    images = torch.rand(1, 1, 128, 128)
    source = torch.rand(1, 1, 128, 128)
    model = res2net50()
    modelR=model.to(device)
    print(summary(modelR,[(1,128,128),(1,128,128)]))
    # model = model.cuda(0)
    # print(model(images,source).size())
