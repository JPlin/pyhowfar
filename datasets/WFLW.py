from __future__ import print_function

import os
import numpy as np
import random
import math
from skimage import io

import torch
import torch.utils.data as data

# from utils.utils import *
from utils.imutils import *
from utils.transforms import *
from utils.osutils import *


class WFLW(data.Dataset):
    def __init__(self,
                 args,
                 split,
                 img_folder='/mnt/d1p8/ming/FaceData/WFLW/WFLW_align',
                 resize_size=256):
        self.nParts = 98
        self.pointType = args.pointType
        # self.anno = anno
        self.img_folder = img_folder
        self.split = split
        self.is_train = True if self.split == 'train' else False
        self.anno = self._getDataFaces(self.is_train)
        self.total = len(self.anno)
        self.scale_factor = args.scale_factor
        self.rot_factor = args.rot_factor
        self.resize_size = resize_size
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def _getDataFaces(self, is_train):
        files = os.listdir(self.img_folder)
        lines = []
        for d in files:
            if d.endswith('.jpg') or d.endswith('.png'):
                lines.append(os.path.join(self.img_folder, d))

        split_point = int(len(lines) * 0.95)
        if is_train:
            print('=> loaded wflwtrain set, {} images were found'.format(
                split_point))
            return lines[:split_point]
        else:
            print('=> loaded wflw validation set, {} images were found'.format(
                len(lines) - split_point))
            return lines[split_point:]

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        inp, out, pts, c, s = self.generateSampleFace(index)
        self.pts, self.c, self.s = pts, c, s
        if self.is_train:
            return inp, out
        else:
            meta = {
                'index': index,
                'center': c,
                'scale': s,
                'pts': pts,
            }
            return inp, out, meta

    def generateSampleFace(self, idx):
        sf = self.scale_factor
        rf = self.rot_factor

        pts = read_npy(
            os.path.join(
                self.img_folder, self.anno[idx].replace('.jpg',
                                                        '.npy').replace(
                                                            '.png', '.npy')))
        c = torch.Tensor((256 / 2, 256 / 2))
        s = 1.0

        img = load_image(os.path.join(self.img_folder, self.anno[idx]))

        # rescale image
        img = resize(img, self.resize_size, self.resize_size)
        pts = torch.Tensor(pts)

        r = 0
        if self.is_train:
            # s = s * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            r = torch.randn(1).mul_(rf).clamp(
                -2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='WFLW')
                c[0] = img.size(2) - c[0]

            # add random color disturb
            img[0, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)

        inp = crop(img, c, s, [256, 256], rot=r)
        # inp = img
        inp = color_normalize(inp, self.mean, self.std)

        tpts = pts.clone()
        out = torch.zeros(self.nParts, 64, 64)
        for i in range(self.nParts):
            if tpts[i, 0] > 0:
                if r != 0:
                    tpts[i, 0:2] = to_torch(
                        sk_transform(
                            tpts[i, 0:2] + 1,
                            c,
                            s, [256, 256],
                            invert=0,
                            rot=-r))
                else:
                    tpts[i, 0:2] = tpts[i, 0:2] + 1
                tpts[i] = tpts[i] / 4.0
                out[i] = draw_labelmap(out[i], tpts[i] - 1, sigma=1)

        return inp, out, tpts * 4.0, c, s


if __name__ == "__main__":
    import opts, demo
    args = opts.argparser()
    dataset = WFLW(args, 'test')
    crop_win = None
    for i in range(dataset.__len__()):
        input, target, meta = dataset.__getitem__(i)
        input = input.numpy().transpose(1, 2, 0) * 255.
        target = target.numpy()
        if crop_win is None:
            crop_win = plt.imshow(input)
        else:
            crop_win.set_data(input)
        plt.pause(0.5)
        plt.draw
