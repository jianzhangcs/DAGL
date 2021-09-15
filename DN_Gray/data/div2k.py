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
        self.noiseL=args.noiseL
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        self.train = train

    def _scan(self):
        names_hr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        return names_hr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        self.apath = dir_data
        if self.train == False:
            self.apath = os.path.join(self.apath,'Val')
        self.dir_hr = os.path.join(self.apath, 'DIV2K_HQ')