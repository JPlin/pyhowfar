import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io

import models
import opts
from datasets import *
from utils.evaluation import final_preds
from utils.imutils import *
from utils.transforms import *

args = opts.argparser()
datatype = "TEST"

# Load model
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

model = models.__dict__[args.netType](
    num_modules=args.nStacks, pointNumber=args.pointNumber)
model = torch.nn.DataParallel(model).cuda()
if args.checkpoint:
    checkpoint = torch.load(
        os.path.join(args.checkpoint, 'checkpoints.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
else:
    raise 'checkpoint are not specified'

if datatype == "TEST":
    # Setup data loader
    def get_loader(data):
        return {
            '300W_LP': W300LP,
            'LS3D-W/300VW-3D': VW300,
            'AFLW2000': AFLW2000,
            'LS3D-W': LS3DW,
            'CARI': CARI_ALIGN
        }[data[5:]]

    Loader = get_loader(args.data)
    val_loader = Loader(args, 'train')

    # Start inference
    for i in range(len(val_loader)):
        input, target = val_loader[i]
        batch_input = torch.unsqueeze(input, 0).cuda()
        batch_output = model(batch_input)[-1].detach()
        print(batch_output.size())
        batch_score_map = batch_output.data.cpu()
        pts_1 = get_preds_fromhm(batch_score_map)[0] * 4.0

        demold_input = color_denormalize(
            input, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # gt_pts = meta['pts']
        # plt.subplot(121)
        # show_joints(demold_input, gt_pts)
        plt.subplot(122)
        show_joints(demold_input, pts_1)
        plt.show()
else:
    example_dir = 'example/input'
    example_list = os.listdir(example_dir)
    example_list = [
        x for x in example_list if x.endswith('.png') or x.endswith('.jpg')
    ]
    for example in example_list:
        im_path = os.path.join(example_dir, example)
        im = io.imread(im_path)
        t_im = resize(im, 256, 256)
        t_im = color_normalize(t_im, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        batch_input = torch.unsqueeze(t_im, 0).cuda()
        print(batch_input.size())
        batch_output = model(batch_input)[-1].detach()
        batch_score_map = batch_output.data.cpu()
        pts_1 = get_preds_fromhm(batch_score_map)[0] * 4.0
        demold_input = color_denormalize(
            t_im, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        show_joints(demold_input, pts_1)
        plt.show()