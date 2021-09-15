from model.CA_model import ContextualAttention_Enhance
from model.SK_model import SKUnit
import torch
import torch.nn as nn
import numpy as np

class merge_block(nn.Module):
    def __init__(self,in_channels,out_channels,vector_length=32,use_multiple_size=False,use_topk=False):
        super(merge_block,self).__init__()
        self.SKUnit = SKUnit(in_features=in_channels,out_features=out_channels,M=2,G=8,r=2)
        self.CAUnit = ContextualAttention_Enhance(in_channels=in_channels,use_multiple_size=use_multiple_size)
        self.fc1 = nn.Linear(in_features=in_channels,out_features=vector_length)
        self.att_CA = nn.Linear(in_features=vector_length,out_features=out_channels)
        self.att_SK = nn.Linear(in_features=vector_length,out_features=out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out1 = self.SKUnit(x).unsqueeze_(dim=1)
        out2 = self.CAUnit(x).unsqueeze_(dim=1)
        out = torch.cat((out2,out1),dim=1)
        U = torch.sum(out,dim=1)
        attention_vector = U.mean(-1).mean(-1)
        attention_vector = self.fc1(attention_vector)
        attention_vector_CA = self.att_CA(attention_vector).unsqueeze_(dim=1)
        attention_vector_SK = self.att_SK(attention_vector).unsqueeze_(dim=1)
        vector = torch.cat((attention_vector_CA,attention_vector_SK),dim=1)
        vector = self.softmax(vector).unsqueeze(-1).unsqueeze(-1)
        out = (out*vector).sum(dim=1)
        return out
