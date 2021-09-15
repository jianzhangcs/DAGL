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
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description="GATIR_Test")
parser.add_argument("--logdir", type=str, default="checkpoints/mq/model/model_best.pt", help='path of log files')
parser.add_argument("--test_data", type=str, default='classic5', help='test on Classic5 or LIVE1')
parser.add_argument('--quality',type=int,default=10,help='Quality factor')
parser.add_argument("--lr_data",type=str,default="testsets/Classic5/lr_10")
parser.add_argument("--save_path",type=str,default='res')
parser.add_argument("--rgb_range",type=int,default=1.)
parser.add_argument('--save_res',action='store_true')
parser.add_argument("--ensemble",action='store_true')
opt = parser.parse_args()

def normalize(data,rgb_range):
    return data/(255./rgb_range)

def main():
    # Build model
    torch.cuda.set_device(0)
    print('Loading model ...\n')
    net = my_model.Model(args, checkpoint)
    net.model.load_state_dict(torch.load(opt.logdir,map_location='cuda'))
    model = net.cuda()#nn.DataParallel(net, device_ids=device_ids).cuda()
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join(opt.test_data, '*.bmp'))
    files_lr = glob.glob(os.path.join(opt.test_data,'lr_'+str(opt.quality), '*.jpeg'))
    files_source.sort()
    files_lr.sort()
    # process data
    psnr_test = 0
    cou=0
    print(files_lr)
    if os.path.isdir(opt.save_path)==False:
        os.mkdir(opt.save_path)
    for f,f_lr in zip(files_source,files_lr):
        # image
        path = os.path.join(opt.save_path,f.split('/')[-1])
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:, :, 0]),opt.rgb_range)
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)

        Img_lr = cv2.imread(f_lr)
        Img_lr = normalize(np.float32(Img_lr[:, :, 0]),opt.rgb_range)
        Img_lr = np.expand_dims(Img_lr, 0)
        Img_lr = np.expand_dims(Img_lr, 1)
        ISource_lr = torch.Tensor(Img_lr)
        ISource, INoisy = Variable(ISource.cuda()), Variable(ISource_lr.cuda())
        with torch.no_grad():  # this can save much memory
            Out = torch.clamp(model(INoisy ,0,ensemble=opt.ensemble), 0., opt.rgb_range)
            if opt.save_res == True:
                image_clean = (Out[0,0].cpu().data.numpy()*(255./opt.rgb_range)).astype(np.uint8)
                cv2.imwrite(path,image_clean)
        err = torch.mean((Out - ISource) * (Out - ISource))
        noi_psnr = batch_PSNR(INoisy, ISource, opt.rgb_range)
        psnr = batch_PSNR(Out, ISource, opt.rgb_range)
        psnr_test += psnr
        print("%s Input %.2f, Output PSNR %.2f MSE %f" % (f, noi_psnr, psnr, err))
        cou+=1
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
    print('Finish!')

if __name__ == "__main__":
    main()
