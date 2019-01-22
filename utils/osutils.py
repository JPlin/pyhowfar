from __future__ import absolute_import

import os
import errno
import numpy as np


def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def isfile(fname):
    return os.path.isfile(fname)


def isdir(dirname):
    return os.path.isdir(dirname)


def join(path, *paths):
    return os.path.join(path, *paths)


def read_txt(file_path):
    lines = []
    with open(file_path, 'r') as f:
        flines = f.readlines()
        for line in flines:
            lines.append([float(x) for x in line.strip().split(' ')])
    return np.array(lines)


def read_npy(file_path):
    return np.load(file_path)