import torch
import utility
import data
from option import args
import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *
import model as my_model
import time
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
from pytorch_ssim import ssim

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description="Test_DAGL")
parser.add_argument("--logdir", type=str, default='checkpoints/dmic/model/model_best.pt')
parser.add_argument("--test_data", type=str, default='testsets/Kodak24', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=30, help='noise level used on test set')
parser.add_argument('--rgb_range',type=int,default=1.)
parser.add_argument('--results_path',type=str,default='res_vis/res_adp_thd3_32b_urban100')
parser.add_argument('--save_image',action='store_true')
parser.add_argument("--ensemble",action='store_true')
opt = parser.parse_args()

def normalize(data,rgb_range):
    return data/(255./rgb_range)

def main():
    # Build model
    print('Loading model ...\n')
    net = my_model.Model(args, checkpoint)
    net.model.load_state_dict(torch.load(opt.logdir,map_location='cuda'))
#     net.load_state_dict({('model.'+k):v for k,v in torch.load(opt.logdir,map_location={'cuda:1':'cuda:0'}).items()})
    model = net.cuda()#nn.DataParallel(net, device_ids=device_ids).cuda()
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join(opt.test_data, '*.png'))+glob.glob(os.path.join(opt.test_data, '*.tif'))
    files_lr = glob.glob(os.path.join(opt.test_data,'lr', '*.png'))+glob.glob(os.path.join(opt.test_data,'lr', '*.tif'))
    files_source.sort()
    files_lr.sort()
    psnr_test = 0
    ssim_test = 0
    for f,f_lr in zip(files_source,files_lr):
        Img = cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2RGB)
        Img = normalize(np.float32(Img),opt.rgb_range)
        ISource = torch.Tensor(Img).permute(2,0,1).unsqueeze(0)

        Img_lr = cv2.cvtColor(cv2.imread(f_lr),cv2.COLOR_BGR2RGB)
        Img_lr = normalize(np.float32(Img_lr),opt.rgb_range)
        ISource_lr = torch.Tensor(Img_lr).permute(2,0,1).unsqueeze(0)

        ISource, INoisy = Variable(ISource.cuda()), Variable(ISource_lr.cuda())
        with torch.no_grad():  # this can save much memory
            Out = torch.clamp(model(INoisy ,0,ensemble=opt.ensemble), 0., opt.rgb_range)
            if opt.save_image == True:
                if os.path.isdir(opt.results_path) == False:
                    os.mkdir(opt.results_path)
                path = os.path.join(opt.results_path,f.split('/')[-1])
                image = cv2.cvtColor(cv2.merge((Out*(255./opt.rgb_range)).squeeze(0).cpu().data.numpy().astype(np.uint8)),cv2.COLOR_BGR2RGB)
                cv2.imwrite(path,image)
        err = torch.mean((Out - ISource) * (Out - ISource))
        ssim_curr = ssim(Out / opt.rgb_range, ISource / opt.rgb_range)
        noi_psnr = batch_PSNR(INoisy, ISource, opt.rgb_range)
        psnr = utility.calc_psnr(
                        Out, ISource, 1, args.rgb_range
                    )#batch_PSNR(Out, ISource, opt.rgb_range)
        psnr_test += psnr
        ssim_test += ssim_curr
        print("%s input %.2f, PSNR %.2f SSIM %f" % (f, noi_psnr, psnr, ssim_curr))
    psnr_test /= len(files_source)
    ssim_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
    print('SSIM on test data %f' % ssim_test)


if __name__ == "__main__":
    main()
