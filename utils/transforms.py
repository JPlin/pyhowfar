from __future__ import absolute_import

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import torch
from skimage import transform as T

from .imutils import *
from .misc import *


def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, x.size(1), x.size(2))

    for t, m, s in zip(x, mean, std):
        t.sub_(m).div_(s)
    return x

def color_denormalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, x.size(1), x.size(2))

    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return x


def flip_back(flip_output, dataset='mpii'):
    """
    flip output map
    """
    if dataset == 'mpii':
        matchedParts = ([0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13])
    elif dataset == 'cari_align':
        matchedParts = ([27 , 29],[0 , 5] , [1 , 4] , [2 , 3] , [7 , 12], [6 , 13], [10 ,15] , [16 , 17] , [9 , 14] , [8 ,11] , [18 , 19] , [20 , 22] , [23 , 25] , [62 , 30] , [61 , 31] , [60 , 32], [59 , 33], [58 , 34], [57 , 35], [56 , 36], [55 , 37] , [54 , 38], [53, 39] , [52 , 40] , [51 , 41] , [50 , 42], [49 , 43] , [48 , 44] , [47 , 45])
    elif dataset == 'WFLW':
        matchedParts = ([0 , 32], [1 , 31] , [2 , 30] , [3 , 29] , [4, 28] , [5 , 27] , [6 , 26] , [7 , 25] , [8 , 24] , [9 , 23] , [10 , 22] , [11 , 21], [12 , 20] , [13 , 19] , [14 , 18] , [15 , 17] , [33 , 46] , [34 , 45] , [35 , 44], [36 , 43],[37 , 42], [38 , 50] , [39 , 49] , [40 , 48] , [41 , 47] , [60 , 72] , [61 , 71] , [62 , 70] , [63 , 69] , [64 , 68] , [65 , 75] , [66 , 74] , [67 , 73] , [56 , 58] , [55 , 59] , [76 , 82] , [88 , 92] , [77 , 81] , [78 , 80] , [87 , 83] , [86 , 84], [89 , 91], [95 , 93] , [96 , 97])
    else:
        print('Not supported dataset: ' + dataset)

    # flip output horizontally
    flip_output = fliplr(flip_output.numpy())

    # Change left-right parts
    for pair in matchedParts:
        tmp = np.copy(flip_output[:, pair[0], :, :])
        flip_output[:, pair[0], :, :] = flip_output[:, pair[1], :, :]
        flip_output[:, pair[1], :, :] = tmp

    return torch.from_numpy(flip_output).float()


def shufflelr(x, width, dataset='mpii'):
    """
    flip coords
    """
    if dataset == 'mpii':
        matchedParts = ([0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13])
    elif dataset in ['w300lp', 'vw300', 'w300', 'menpo']:
        matchedParts = ([0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
                        [17, 26], [18, 25], [19, 26], [20, 23], [21, 22], [36, 45], [37, 44],
                        [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34], [50, 52],
                        [49, 53], [48, 54], [61, 63], [62, 64], [67, 65], [59, 55], [58, 56])
    elif dataset == 'WFLW':
        matchedParts = ([0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [
            6, 26
        ], [7, 25], [8, 24], [9, 23], [10, 22], [11, 21], [12, 20], [13, 19], [
            14, 18
        ], [15, 17], [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [
            38, 50
        ], [39, 49], [40, 48], [41, 47], [60, 72], [61, 71], [62, 70], [
            63, 69
        ], [64, 68], [65, 75], [66, 74], [67, 73], [56, 58], [55, 59],
                        [76, 82], [88, 92], [77, 81], [78, 80], [87, 83],
                        [86, 84], [89, 91], [95, 93], [96, 97])
    elif dataset == 'cari_align':
        matchedParts = ([27 , 29],[0 , 5] , [1 , 4] , [2 , 3] , [7 , 12], [6 , 13], [10 ,15] , [16 , 17] , [9 , 14] , [8 ,11] , [18 , 19] , [20 , 22] , [23 , 25] , [62 , 30] , [61 , 31] , [60 , 32], [59 , 33], [58 , 34], [57 , 35], [56 , 36], [55 , 37] , [54 , 38], [53, 39] , [52 , 40] , [51 , 41] , [50 , 42], [49 , 43] , [48 , 44] , [47 , 45])
    else:
        print('Not supported dataset: ' + dataset)

    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    # Change left-right parts
    for pair in matchedParts:
        tmp = x[pair[0], :].clone()
        x[pair[0], :] = x[pair[1], :]
        x[pair[1], :] = tmp

    return x


def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)


def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-center[0] / h + .5)
    t[1, 2] = res[0] * (-center[1] / h + .5)
    # t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    # t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1

def sk_transform(pt, center, scale, res, invert=0, rot=0):
    rot = rot * np.pi / 180
    x0, y0 = res[0] / 2, res[0] / 2
    ret_pt = np.zeros_like(pt)
    ret_pt[0] = ((pt[0] - x0) * np.cos(rot)) - ((pt[1] - y0) * np.sin(rot)) + x0
    ret_pt[1] = ((pt[0] - x0) * np.sin(rot)) + ((pt[1] - y0) * np.cos(rot)) + y0
    return ret_pt


def transform_preds(coords, center, scale, res):
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())
    for p in range(coords.size(0)):
        coords[p, 0:2] = to_torch(transform(coords[p, 0:2], center, scale, res, 1, 0))
    return coords


def crop(img, center, scale, res, rot=0):
    img = im_to_numpy(img)

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(res[0], res[1], img.shape[2]) \
                if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else:
            img = scipy.misc.imresize(img, [new_ht, new_wd])
            center = center * 1. / sf
            scale = scale / sf

    # Upper left point
    # ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    # br = np.array(transform(res, center, scale, res, invert=1))
    ul = np.array([0, 0])
    br = np.array([255, 255])

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = im_to_torch(scipy.misc.imresize(new_img, res))
    return new_img
