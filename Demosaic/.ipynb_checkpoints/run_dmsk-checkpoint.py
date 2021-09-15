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
from pytorch_ssim import ssim
import time
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
#parser.add_argument("--logdir", type=str, default="/home/umin/Downloads/G2/RNAN_dmsk/DN_Gray/code/modelzoo/RNAN/Demosaic/experiment/model/RNAN_Demosaic_RGB_F64G10P48L2N1.pt", help='path of log files')
# parser.add_argument("--logdir", type=str, default='res_cola_v2_dmsk/model/model_best.pt')
parser.add_argument("--logdir", type=str, default='res_greccr2bv2_adp_thd3_800_32b/model/model_best.pt')
parser.add_argument("--test_data", type=str, default='testsets/McMaster18', help='test on Set12 or Set68')
parser.add_argument("--lr_data",type=str,default="testsets/McMaster18/lr")
# parser.add_argument("--test_data", type=str, default='DIV2K_dmsk/Val/DIV2K_dmsk_HQ', help='test on Set12 or Set68')
# parser.add_argument("--lr_data",type=str,default="DIV2K_dmsk/Val/DIV2K_dmsk_LQ/25")
# parser.add_argument("--test_data", type=str, default='Set18')#'DIV2K_dmsk/Val/DIV2K_dmsk_HQ', help='test on Set12 or Set68')
# parser.add_argument("--lr_data",type=str,default='Set18/lr')#"DIV2K_dmsk/Val/DIV2K_dmsk_LQ/25")
# parser.add_argument("--test_data", type=str, default='BSD68')#'DIV2K_dmsk/Val/DIV2K_dmsk_HQ', help='test on Set12 or Set68')
# parser.add_argument("--lr_data",type=str,default='BSD68/lr')#"DIV2K_dmsk/Val/DIV2K_dmsk_LQ/25")
parser.add_argument("--test_noiseL", type=float, default=30, help='noise level used on test set')
parser.add_argument('--rgb_range',type=int,default=1.)
parser.add_argument('--save_path',type=str,default='res_vis/res_adp_thd3_32b_urban100')
parser.add_argument('--if_save',type=bool,default=False)
opt = parser.parse_args()

def normalize(data,rgb_range):
    return data/(255./rgb_range)

def forward_chop(x, nn_model, n_GPUs= 1, shave=10, min_size=10000):
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
            # print(lr_batch.shape)
            # exit(0)
            # image_noise = (lr_batch[0,0].cpu().data.numpy()*255.).astype(np.uint8)
            sr_batch = nn_model(lr_batch,0)
            # image_clean = (sr_batch[0,0].cpu().data.numpy()*255.).astype(np.uint8)
            # image = np.concatenate((image_noise,image_clean),axis=1)
            # cv2.imshow('image',image)
            # cv2.waitKey(0)
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
    # Build model
    print('Loading model ...\n')
    net = my_model.Model(args, checkpoint)
    # net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    # device_ids = [0]
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # net.load_state_dict(torch.load(opt.logdir))
    net.load_state_dict({('model.'+k):v for k,v in torch.load(opt.logdir,map_location={'cuda:1':'cuda:0'}).items()})
    # net.load_state_dict({k.replace('model.', ''): v for k, v in
    #                     torch.load(os.path.join(opt.logdir, 'net_40.pth')).items()})
    model = net.cuda()#nn.DataParallel(net, device_ids=device_ids).cuda()
    # model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_49.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join(opt.test_data, '*.png'))
    files_lr = glob.glob(os.path.join(opt.lr_data, '*.png'))
    files_source.sort()
    files_lr.sort()
    # print(len(files_lr))
    # exit(0)
    # process data
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
            Out = torch.clamp(model(INoisy ,0), 0., opt.rgb_range)
            if opt.if_save == True:
                if os.path.isdir(opt.save_path) == False:
                    os.mkdir(opt.save_path)
                # cv2.imwrite(os.path.join(opt.save_path, f.split('/')[-1]), image)
                path = os.path.join(opt.save_path,f.split('/')[-1])
                # print(path)
                image = cv2.cvtColor(cv2.merge((Out*(255./opt.rgb_range)).squeeze(0).cpu().data.numpy().astype(np.uint8)),cv2.COLOR_BGR2RGB)
                cv2.imwrite(path,image)
            # cv2.imshow('image',image)
            # cv2.waitKey(1)

        # vutils.save_image(Out, os.path.join('D:\pc\Pyramid-Attention-Networks\CAR\experiment\Test_RNAN\\results\Classic5\Q10',
        #                                     f.split('\\')[-1].replace('HQ','LQ')), padding=0, normalize=True)
        # exit(0)
        # vutils.save_image(INoisy, f + '_noi.tif', padding=0, normalize=True)
        # vutils.save_image(torch.abs(Out - ISource), f + '_l1.tif', padding=0, normalize=True)
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        err = torch.mean((Out - ISource) * (Out - ISource))
        ssim_curr = ssim(Out / opt.rgb_range, ISource / opt.rgb_range)
        noi_psnr = batch_PSNR(INoisy, ISource, opt.rgb_range)
        psnr = batch_PSNR(Out, ISource, opt.rgb_range)
        psnr_test += psnr
        ssim_test += ssim_curr
        print("%s input %.2f, PSNR %.2f SSIM %f" % (f, noi_psnr, psnr, ssim_curr))
    psnr_test /= len(files_source)
    ssim_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)
    print('SSIM on test data %f' % ssim_test)


if __name__ == "__main__":
    main()
