import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data


class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
            # print(data_range)
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
            # print(data_range)
            # exit(0)

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        self.train = train
        # self.noise_level = args.noise_level
        # print(self.noise_level)
        # exit(0)

    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        self.apath = dir_data
        if self.train == False:
            self.apath = os.path.join(self.apath,'Val')
        self.dir_hr = os.path.join(self.apath, 'DIV2K_HQ')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_LQ')
        if self.input_large: self.dir_lr += 'L'

