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
parser = argparse.ArgumentParser(description="GATIR")
parser.add_argument('--logdir',type=str,default='checkpoints/15/model/model_best.pt',help="Path to pretrained model")
parser.add_argument("--test_data", type=str, default='testsets/BSD68', help='test on Set12, BSD68 and Urban100')
parser.add_argument("--test_noiseL", type=float, default=15., help='noise level used on test set')
parser.add_argument("--rgb_range",type=int,default=1.)
parser.add_argument("--save_path",type=str,default='res_vis/GATIR_15',help='Save restoration results')
parser.add_argument("--save",action='store_true')
parser.add_argument("--ensemble",action='store_true')
opt = parser.parse_args()
def normalize(data,rgb_range):
    return data/(255./opt.rgb_range)

def main():
    # Build model
    torch.cuda.set_device(0)
    print('Loading model ...\n')
    net = my_model.Model(args, checkpoint)
    net.model.load_state_dict(torch.load(opt.logdir,map_location='cuda')) # this model is trained at cuda1.
    model = net.cuda()
    model.eval()
    
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join(opt.test_data, '*.png'))
    files_source.sort()
    print(os.path.join(opt.test_data, '*.png'),files_source)
    # process data
    psnr_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:, :, 0]),opt.rgb_range)
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        torch.manual_seed(1)    # fixed seed
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / (255./opt.rgb_range))
        # noisy image
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad():  # this can save much memory
            Out = torch.clamp(model(INoisy ,0,ensemble=opt.ensemble), 0., opt.rgb_range)
            image = (Out[0,0].cpu().data.numpy()*(255./opt.rgb_range)).astype(np.uint8)
            # save results
            if opt.save == True:
                if os.path.isdir(opt.save_path) == False:
                    os.mkdir(opt.save_path)
                cv2.imwrite(os.path.join(opt.save_path,f.split('/')[-1]),image)
        noi_psnr = batch_PSNR(INoisy, ISource, opt.rgb_range)
        psnr = batch_PSNR(Out, ISource, opt.rgb_range)
        psnr_test += psnr
        print("%s Input PSNR %.2f, Output PSNR %.2f" % (f, noi_psnr, psnr))
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
    print('Finish!')
if __name__ == "__main__":
    main()
