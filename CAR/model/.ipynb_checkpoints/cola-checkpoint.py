import torch.nn as nn
from model.merge_net import MergeNet
import math
from model.GreccRcaa import RR
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif type(m) == nn.BatchNorm2d or type(m) == nn.BatchNorm1d:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)
def make_model(args, parent=False):
    if args.mode == 'E':
        print('COLA-E')
        return RR(args)
    elif args.mode == 'B':
        print('COLA-B')
        net =  MergeNet(in_channels=1,intermediate_channels=64,vector_length=32,use_multiple_size=True,dncnn_depth=6,num_merge_block=4)
        net.apply(weights_init_kaiming)
        return net
    else:
        raise ValueError('Wrong mode.')