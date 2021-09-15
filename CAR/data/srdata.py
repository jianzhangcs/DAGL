import os
import random
import pickle
from data import common
import numpy as np
import imageio
import torch.utils.data as data
import cv2

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.list_q = ['/10/','/20/','/30/','/40/']
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        self.images_hr, self.images_lr = list_hr, list_lr
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)
    # Below functions as used to prepare images
    def _scan(self):
        name_hr = []
        name_lr = []
        list_hr = os.listdir(self.dir_hr)
        list_lr = os.listdir(self.dir_lr)

        for i in list_hr:
            if i.endswith('.png') or i.endswith('jpeg'):
                name_hr.append(os.path.join(self.dir_hr,i))
        for j in list_lr:
            if j.endswith('.png') or j.endswith('jpeg'):
                name_lr.append(os.path.join(self.dir_lr,j.replace('.jpeg','.png')))
        name_hr.sort()
        name_lr.sort()
        return name_hr, name_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        self.ext = '.png'

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr = lr[:,:,0]
        lr, hr = self.get_patch(lr, hr)
        lr = np.expand_dims(lr,2)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, filename

    def __len__(self):
        if self.train:                                 # method of repeat
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:                                 # method of repeat
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx].replace('.png','.jpeg')
        if self.train==True:
            id_q=np.random.randint(low=0, high=4)
            f_lr=f_lr.replace('/10/',self.list_q[id_q])

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)#[:,:,0]
            if len(hr.shape)==3:
                hr = hr[:,:,0]
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            lr = np.load(f_lr)
            hr = np.load(f_hr)

        return lr, hr, filename

    def get_patch(self, lr, hr):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr, hr, patch_size, scale, multi_scale=multi_scale
            )
            lr, hr = common.augment([lr, hr])
            lr = common.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)
