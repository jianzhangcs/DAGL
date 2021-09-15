import cv2
import os
import argparse
import glob
from torch.autograd import Variable
from model.merge_net import MergeNet
from utils import *
import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image
import tqdm

parser = argparse.ArgumentParser(description="COLA-Net Test")
parser.add_argument("--logdir", type=str, default="checkpoints/cola_v1/model/model_best.pt", help='path of log files')
parser.add_argument("--test_data", type=str, default='testsets/patches_DND', help='test on Set12 or Set68')
parser.add_argument("--save_path",type=str,default='save/res_cola_v1')
parser.add_argument('--gpu_idx',type=int,default=0)
opt = parser.parse_args()

def normalize(data):
    return data/255.

def forward_chop(x, nn_model, n_GPUs= 1, shave=10, min_size=10000):
    scale = 1
    n_GPUs = min(n_GPUs, 4)
    b, c, h, w = x.size()
    # print(h,w)
    # exit(0)
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
            # print(len(lr_list),lr_list[0].shape)
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            # print(lr_batch.shape)
            # exit(0)
            sr_batch = nn_model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(patch, nn_model, shave=shave, min_size=min_size) \
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

def main():
    torch.cuda.set_device(opt.gpu_idx)
    # Build model
    print('Loading model ...\n')
    net = MergeNet(in_channels=3,intermediate_channels=64,vector_length=32,use_multiple_size=True,dncnn_depth=6,num_merge_block=4)
    net.load_state_dict(torch.load(opt.logdir, map_location='cuda'))
    model = net.cuda()
    model.eval()
    # load data info
    images = glob.glob(os.path.join(opt.test_data,'*.png'))
    count = 1
    if os.path.isdir(opt.save_path)==False:
        os.mkdir(opt.save_path)
    for im_path in tqdm.tqdm(images):
        Img = cv2.cvtColor(cv2.imread(im_path),cv2.COLOR_BGR2RGB)
        Img = transforms.ToTensor()(Image.fromarray(Img)).cuda().unsqueeze(0)
        with torch.no_grad():  # this can save much memory
            Out = torch.clamp(forward_chop(x=Img, nn_model=model), 0., 1.)
        vutils.save_image(Out, os.path.join(opt.save_path , str(count)+'.png'), padding=0, normalize=False)
        count += 1
if __name__ == "__main__":
    main()