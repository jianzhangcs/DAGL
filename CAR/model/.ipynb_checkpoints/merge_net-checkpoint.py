from model.merge_unit import merge_block
from model.DnCNN_Block import DnCNN
import torch.nn as nn
import torch

class MergeNet(nn.Module):
    def __init__(self,in_channels,intermediate_channels,vector_length,use_multiple_size,dncnn_depth,num_merge_block):
        super(MergeNet,self).__init__()
        layers = []
        for i in range(num_merge_block):
            if i == 0:
                layers.append(
                    DnCNN(nplanes_in=in_channels,nplanes_out=intermediate_channels,features=intermediate_channels,
                          kernel=3,depth=dncnn_depth,residual=True,bn=True)
                )
                layers.append(
                    merge_block(in_channels=intermediate_channels, out_channels=intermediate_channels,
                                vector_length=vector_length, use_multiple_size=use_multiple_size)
                )
            else:
                layers.append(
                    DnCNN(nplanes_in=intermediate_channels,nplanes_out=intermediate_channels,features=intermediate_channels,
                          kernel=3,depth=dncnn_depth,residual=True,bn=True)
                )
                layers.append(
                    merge_block(in_channels=intermediate_channels, out_channels=intermediate_channels,
                                vector_length=vector_length, use_multiple_size=use_multiple_size)
                )
        layers.append(
            DnCNN(nplanes_in=intermediate_channels, nplanes_out=in_channels, features=intermediate_channels,
                  kernel=3, depth=dncnn_depth, residual=False, bn=True)
        )
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        out = self.model(x)
        return x+out

