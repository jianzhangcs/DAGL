import numpy as np
import scipy.io as sio
import os
import h5py
import torch
import argparse
import cv2
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model.GReccRcaa import RR2
from option import args
import glob
import tqdm
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image

parse = argparse.ArgumentParser('DND')
parse.add_argument('--logdir',type=str,default='checkpoints/cola_v2/model/model_best.pt')
parse.add_argument('--data_path',type=str,default='testsets/patches_DND')
parse.add_argument('--save_path',type=str,default='save/res_cola_v2')
opt = parse.parse_args()

def test_pad(model, L, modulo=16):
    h, w = L.size()[-2:]
    paddingBottom = int(np.ceil(h / modulo) * modulo - h)
    paddingRight = int(np.ceil(w / modulo) * modulo - w)
    L = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(L)
    E = model(L)
    E = E[..., :h, :w]
    return E


def augment_img_tensor(img, mode=0):
    img_size = img.size()
    img_np = img.data.cpu().numpy()
    if len(img_size) == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    elif len(img_size) == 4:
        img_np = np.transpose(img_np, (2, 3, 1, 0))
    img_np = augment_img(img_np, mode=mode)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_np))
    if len(img_size) == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    elif len(img_size) == 4:
        img_tensor = img_tensor.permute(3, 2, 0, 1)

    return img_tensor.type_as(img)


def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def test_x8(model, L):
    E_list = [model(augment_img_tensor(L, mode=i)) for i in range(8)]
    for i in range(len(E_list)):
        if i == 3 or i == 5:
            E_list[i] = augment_img_tensor(E_list[i], mode=8 - i)
        else:
            E_list[i] = augment_img_tensor(E_list[i], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E
def forward_chop(x, nn_model, n_GPUs= 1, shave=10, min_size=10000,flg=1):
    scale = 1
    n_GPUs = min(n_GPUs, 4)
    b, c, h, w = x.size()
    #############################################
    # adaptive shave
    # corresponding to scaling factor of the downscaling and upscaling modules in the network
    shave_scale = 4
    # max shave size
    shave_size_max = 24
    # get half size of the hight and width
    h_half, w_half = h // 2, w // 2
    # mod
    mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
    # ditermine midsize along height and width directions
    h_size = mod_h * shave_scale + shave_size_max
    w_size = mod_w * shave_scale + shave_size_max
    # h_size, w_size = h_half + shave, w_half + shave
    ###############################################
    # h_size, w_size = adaptive_shave(h, w)
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            #if flg == 1:
                #sr_batch = nn_model(lr_batch)
            #else:
            sr_batch = test_x8(nn_model, lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(patch, nn_model, shave=shave, min_size=min_size,flg=flg) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = Variable(x.data.new(b, c, h, w), volatile=True)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

class denoiser():
    def __init__(self):
        self.model=RR2(args).cuda()
        self.model.load_state_dict(torch.load(opt.logdir, map_location = 'cuda'))
        self.model.eval()
    def denoise(self,Img,flg):
        with torch.no_grad():
            Out = torch.clamp(forward_chop(x=Img, nn_model=self.model,flg=flg), 0., 1.)
        return Out

def _np2Tensor(img,rgb_range):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor
if __name__ == '__main__':
    torch.cuda.set_device(0)
    my_denoiser = denoiser()
    images = glob.glob(os.path.join(opt.data_path,'*.png'))
    print(images)
    count = 1
    if os.path.isdir(opt.save_path)==False:
        os.mkdir(opt.save_path)
    for im_path in tqdm.tqdm(images):
        Img = cv2.cvtColor(cv2.imread(im_path),cv2.COLOR_BGR2RGB)
        Img = transforms.ToTensor()(Image.fromarray(Img)).cuda().unsqueeze(0)
        Out = my_denoiser.denoise(Img=Img,flg=0)
        vutils.save_image(Out, os.path.join(opt.save_path , str(count)+'.png'), padding=0, normalize=False)
        count += 1
