import model.common as common
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from model.merge_unit import merge_block

class RR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RR, self).__init__()

        n_resblocks = 16  # args.n_resblocks
        n_feats = 64  # args.n_feats
        kernel_size = 3

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        msa = CES(in_channels=n_feats,num=args.stages,blocks=args.blocks)
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, nn.PReLU(), res_scale=args.res_scale
            ) for _ in range(n_resblocks // 2)
        ]
        m_body.append(msa)
        for i in range(n_resblocks // 2):
            m_body.append(common.ResBlock(conv, n_feats, kernel_size, nn.PReLU(), res_scale=args.res_scale))

        m_body.append(conv(n_feats, n_feats, kernel_size))
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        res = self.head(x)

        res = self.body(res)

        res = self.tail(res)

        return x+res

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
class CES(nn.Module):
    def __init__(self,in_channels,num=4,blocks=2):
        super(CES,self).__init__()
        print('num_RB:',num)
        print('num_blocks',blocks)
        self.blocks = blocks
        if blocks == 2:
            RBS = [
                common.ResBlock(
                    common.default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1
                ) for _ in range(num)
            ]
            self.RBS = nn.Sequential(
                *RBS
            )
            self.c1 = merge_block(in_channels = in_channels,out_channels=in_channels)#CE(in_channels=in_channels)
            self.c2 = merge_block(in_channels = in_channels,out_channels=in_channels)#CE(in_channels=in_channels)
        else:
            RBS1 = [
                common.ResBlock(
                    common.default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1
                ) for _ in range(num//2)
            ]
            RBS2 = [
                common.ResBlock(
                    common.default_conv, n_feats=in_channels, kernel_size=3, act=nn.PReLU(), res_scale=1
                ) for _ in range(num//2)
            ]
            self.RBS1 = nn.Sequential(
                *RBS1
            )
            self.RBS2 = nn.Sequential(
                *RBS2
            )
            self.c1 = merge_block(in_channels = in_channels,out_channels=in_channels)#CE(in_channels=in_channels)
            self.c2 = merge_block(in_channels = in_channels,out_channels=in_channels)#CE(in_channels=in_channels)
            self.c3 = merge_block(in_channels = in_channels,out_channels=in_channels)
    def forward(self, x):
        if self.blocks == 2:
            out = self.c1(x)
            out = self.RBS(out)
            out = self.c2(out)
        else:
            out = self.c1(x)
            out = self.RBS1(out)
            out = self.c2(out)
            out = self.RBS2(out)
            out = self.c3(out)
        return out
"""
fundamental functions
"""
def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches, paddings

"""
CE network
"""
class CE(nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=1, softmax_scale=10,shape=64 ,p_len=64,in_channels=64
                 , inter_channels=16,use_multiple_size=False,use_topk=False,add_SE=False):
        super(CE, self).__init__()
        self.ksize = ksize
        self.shape=shape
        self.p_len=p_len
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size=use_multiple_size
        self.use_topk=use_topk
        self.add_SE=add_SE
        # self.SE=SE_net(in_channels=in_channels)
        self.conv33=nn.Conv2d(in_channels=2*in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0)
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)
    def forward(self, b):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes

        kernel = self.ksize

        b1 = self.g(b)
        b2 = self.theta(b)
        b3 = self.phi(b)

        raw_int_bs = list(b1.size())  # b*c*h*w

        patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
                                                      strides=[self.stride_1, self.stride_1],
                                                      rates=[1, 1],
                                                      padding='same')
        patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        patch_28 = patch_28.permute(0, 4, 1, 2, 3)
        patch_28_group = torch.split(patch_28, 1, dim=0)

        patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],
                                                        padding='same')

        patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        patch_112 = patch_112.permute(0, 4, 1, 2, 3)
        patch_112_group = torch.split(patch_112, 1, dim=0)

        patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],
                                                        padding='same')

        patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)
        patch_112_group_2 = torch.split(patch_112_2, 1, dim=0)
        f_groups = torch.split(b3, 1, dim=0)
        y = []
        for xii,xi, wi,pi in zip(f_groups,patch_112_group_2, patch_28_group, patch_112_group):
            w,h = xii.shape[2], xii.shape[3]
            _, paddings = same_padding(xii, [self.ksize, self.ksize], [1, 1], [1, 1])
            # wi = wi[0]  # [L, C, k, k]
            c_s = pi.shape[2]
            k_s = wi[0].shape[2]
            wi = wi.view(wi.shape[0],wi.shape[1],-1)
            xi = xi.permute(0, 2, 3, 4, 1)
            xi = xi.view(xi.shape[0],-1,xi.shape[4])
            score_map = torch.matmul(wi,xi)
            score_map = score_map.view(score_map.shape[0],score_map.shape[1],w,h)
            b_s, l_s, h_s, w_s = score_map.shape

            if self.use_topk:
                yi = score_map.view(l_s, -1)
                top_k = min(500, yi.shape[1])
                _, pred = torch.topk(yi, top_k, dim=1)
                mask = torch.zeros_like(yi)
                for idx in range(mask.shape[0]):
                    mask[idx].index_fill_(0, pred[idx], 1)
                yi = yi * mask
                yi = F.softmax(yi*self.softmax_scale, dim=1)
                yi = yi * mask
            else:
                yi = score_map.view(b_s, l_s, -1)
                yi = F.softmax(yi*self.softmax_scale, dim=2).view(l_s, -1)
            pi = pi.view(h_s * w_s, -1)
            # print(pi.shape,yi.shape)
            # exit(0)
            yi = torch.mm(yi, pi)
            # print(yi.shape,b_s, l_s, c_s, k_s, k_s)
            # exit(0)
            yi=yi.view(b_s, l_s, c_s, k_s, k_s)[0]
            zi = yi.view(1, l_s, -1).permute(0, 2, 1)
            zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            zi = zi / out_mask
            y.append(zi)
        y = torch.cat(y, dim=0)
        y = self.W(y)
        y = b + y
        if self.add_SE:
            y_SE=self.SE(y)
            y=self.conv33(torch.cat((y_SE*y,y),dim=1))
        return y
    def GSmap(self,a,b):
        return torch.matmul(a,b)
class SE_net(nn.Module):
    def __init__(self,in_channels,reduction=16):
        super(SE_net,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Conv2d(in_channels=in_channels,out_channels=in_channels//reduction,kernel_size=1,stride=1,padding=0)
        self.fc2=nn.Conv2d(in_channels=in_channels//reduction,out_channels=in_channels,kernel_size=1,stride=1,padding=0)
    def forward(self, x):
        o1=self.pool(x)
        o1=F.relu(self.fc1(o1))
        o1=self.fc2(o1)
        return o1

class size_selector(nn.Module):
    def __init__(self,in_channels,intermediate_channels,out_channels):
        super(size_selector,self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_features=in_channels,out_features=intermediate_channels),
            nn.BatchNorm1d(intermediate_channels),
            nn.ReLU(inplace=True)
        )
        self.selector_a = nn.Linear(in_features=intermediate_channels,out_features=out_channels)
        self.selector_b = nn.Linear(in_features=intermediate_channels, out_features=out_channels)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        vector = x.mean(-1).mean(-1)
        o1 = self.embedding(vector)
        a = self.selector_a(o1)
        b = self.selector_b(o1)
        v = torch.cat((a,b),dim=1)
        v = self.softmax(v)
        a = v[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        b = v[:,1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return a,b
