from __future__ import print_function

import math
import os
import random

import numpy as np
import torch
from skimage import io

import opts
from datasets.cari_align import CARI_ALIGN
from datasets.WFLW import WFLW
from utils.imutils import *
from utils.osutils import *
from utils.transforms import *

args = opts.argparser()


def test_dataset():
    dataset = CARI_ALIGN(args, 'train')
    plt.figure()
    for i in range(dataset.__len__()):
        input, target = dataset.__getitem__(i)
        input = color_denormalize(
            input, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # input_img = input.numpy().transpose(1, 2, 0) * 255.
        # input_img = input_img.astype(np.uint8)
        # plt.subplot(121)
        # plt.imshow(input_img)
        # plt.subplot(122)
        # plt.imshow(target[30])
        # plt.show()
        # pts = meta['pts']
        # show_joints(input, pts, show_N=True)
        # plt.show()
        preds = get_preds_fromhm(torch.unsqueeze(target, 0))
        show_joints(input, preds[0] * 4.0, show_N=True)
        plt.show()
        # print(pts - preds[0] * 4.0)


def test_gaussian():
    out = torch.zeros(1, 10, 10)
    out[0] = draw_labelmap(out[0], torch.Tensor([3.4, 5.5]), sigma=1)
    preds = get_preds_fromhm(torch.unsqueeze(out, 0))
    print(out)
    print(preds)
    show_joints(torch.ones(3, 10, 10), preds[0])
    plt.show()


def test_dataset_WFLW():
    dataset = WFLW(args, 'val')
    plt.figure()
    for i in range(dataset.__len__()):
        input, target, meta = dataset.__getitem__(i)
        input = color_denormalize(
            input, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # input = input.numpy().transpose(1, 2, 0) * 255.
        # input = input.astype(np.uint8)
        pts = meta['pts']
        show_joints(input, pts, show_N=True)
        plt.show()
        preds = get_preds_fromhm(torch.unsqueeze(target, 0))
        show_joints(input, preds[0] * 4.0, show_N=True)
        plt.show()
        print(pts - preds[0] * 4.0)


if __name__ == '__main__':
    test_dataset()
    # test_gaussian()
    # test_dataset_WFLW()
