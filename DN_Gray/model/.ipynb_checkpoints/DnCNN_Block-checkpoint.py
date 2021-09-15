import math
import torch.nn as nn
import torch


def convnxn(in_planes, out_planes, kernelsize, stride=1, bias=False):
    padding = kernelsize//2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding, bias=bias)

def dncnn_batchnorm_init(m, kernelsize=3, b_min=0.025):
    r"""
    Reproduces batchnorm initialization from DnCNN
    https://github.com/cszn/DnCNN/blob/master/TrainingCodes/DnCNN_TrainingCodes_v1.1/DnCNN_init_model_64_25_Res_Bnorm_Adam.m
    """
    n = kernelsize**2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.weight.data[(m.weight.data > 0) & (m.weight.data <= b_min)] = b_min
    m.weight.data[(m.weight.data < 0) & (m.weight.data >= -b_min)] = -b_min
    m.weight.data = m.weight.data.abs()
    m.bias.data.zero_()
    m.momentum = 0.001
class DnCNN(nn.Module):
    def __init__(self, nplanes_in, nplanes_out, features, kernel, depth, bn=True):
        super(DnCNN,self).__init__()
        layers = []
        for idx in range(depth):
            if idx ==0:
                layers.append(nn.Conv2d(in_channels=nplanes_in, out_channels=features, kernel_size=kernel, padding=1,stride=1, bias=False))
                if bn == True:
                    layers.append(nn.BatchNorm2d(features))
                layers.append(nn.ReLU(inplace=True))
            elif idx == depth-1:
                layers.append(nn.Conv2d(in_channels=features, out_channels=nplanes_out, kernel_size=kernel, padding=1,stride=1, bias=False))
            else:
                layers.append(
                    nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel, padding=1, stride=1,
                              bias=False))
                if bn == True:
                    layers.append(nn.BatchNorm2d(features))
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        # self.residual = residual
    def forward(self, x):
        out = self.layers(x)
        # if self.residual ==True:
        #     out = out+x
        return out

if __name__ == "__main__":
    net = DnCNN(nplanes_in=64,nplanes_out=64,features=64,kernel=3,depth=4,bn=True).cuda()
    input_im = torch.zeros((1,64,8,8)).cuda()
    output_im = net(input_im)
    print(output_im.shape)
