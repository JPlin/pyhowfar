import os
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torchvision.datasets as datasets
from progress.bar import Bar
import copy

import models
import opts
from datasets import AFLW2000, CARI_ALIGN, LS3DW, VW300, W300LP, WFLW
from utils.evaluation import (AverageMeter, accuracy, calc_dists, calc_metrics,
                              final_preds)
from utils.imutils import batch_with_heatmap
from utils.logger import Logger, savefig
from utils.misc import adjust_learning_rate, save_checkpoint, save_pred, adjust_map_weight

matplotlib.use('Agg')

args = opts.argparser()
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
# torch.setdefaulttensortype('torch.FloatTensor')

best_acc = 0.
best_auc = 0.
idx = range(1, args.pointNumber, 1)


def get_loader(data):
    return {
        '300W_LP': W300LP,
        'LS3D-W/300VW-3D': VW300,
        'AFLW2000': AFLW2000,
        'LS3D-W': LS3DW,
        'CARI': CARI_ALIGN,
        'WFLW': WFLW
    }[data[5:]]


def weighted_mse_loss(input, target, weight):
    weight = weight.unsqueeze(-1).unsqueeze(-1)
    weight = weight.expand_as(target[0])
    return torch.mean(weight * (input - target)**2)


def main(args):
    global best_acc
    global best_auc
    global map_weight
    map_weight = torch.Tensor([1.] * 63).cuda()

    print("==> Creating model '{}-{}', stacks={}, blocks={}, feats={}".format(
        args.netType, args.pointType, args.nStacks, args.nModules,
        args.nFeats))

    print("=> Models will be saved at dir: {}".format(args.checkpoint))

    # model = models.__dict__[args.netType](
    #     num_stacks=args.nStacks,
    #     num_blocks=args.nModules,
    #     num_feats=args.nFeats,
    #     use_se=args.use_se,
    #     use_attention=args.use_attention,
    #     num_classes=args.pointNumber)
    model = models.__dict__[args.netType](
        num_modules=args.nStacks, pointNumber=args.pointNumber)
    # model.load_state_dict(models.__dict__[args.netType + '_weights'](model))
    model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    title = args.checkpoint.split('/')[-1] + ' on ' + args.data.split('/')[-1]

    Loader = get_loader(args.data)

    val_loader = torch.utils.data.DataLoader(
        Loader(args, 'A'),
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> Loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
            logger = Logger(
                os.path.join(args.checkpoint, 'log.txt'),
                title=title,
                resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names([
            'Epoch', 'LR', 'Train Loss', 'Valid Loss', 'Train Acc', 'Val Acc',
            'AUC'
        ])

    cudnn.benchmark = True
    print('=> Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters()) / (1024. * 1024)))

    if args.evaluation:
        print('=> Evaluation only')
        D = args.data.split('/')[-1]
        save_dir = os.path.join(args.checkpoint, D)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        loss, acc, predictions, auc = validate(
            val_loader, model, criterion, args.netType, args.debug, args.flip)
        save_pred(predictions, checkpoint=save_dir)
        return

    train_loader = torch.utils.data.DataLoader(
        Loader(args, 'train'),
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    # get train_loader of WFLW
    new_args = copy.deepcopy(args)
    new_args.data = 'data/WFLW'
    new_args.train_batch = 32
    Loader = get_loader(new_args.data)
    train_loader2 = torch.utils.data.DataLoader(
        Loader(new_args, 'train'),
        batch_size=new_args.train_batch,
        shuffle=True,
        num_workers=new_args.workers,
        pin_memory=True)
    data_iter = iter(train_loader2)
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule,
                                  args.gamma)
        map_weight = adjust_map_weight(epoch, map_weight, args.weight_schedule)
        print('=> Epoch: %d | LR %.8f | Map_Weight %.8f' % (epoch + 1, lr,
                                                            map_weight[27]))

        train_loss, train_acc = train(
            train_loader,
            model,
            criterion,
            optimizer,
            args.netType,
            args.debug,
            args.flip,
            train_loader2=train_loader2,
            data_iter2=data_iter)
        # do not save predictions in model file
        valid_loss, valid_acc, predictions, valid_auc = validate(
            val_loader, model, criterion, args.netType, args.debug, args.flip)

        logger.append([
            int(epoch + 1), lr, train_loss, valid_loss, train_acc, valid_acc,
            valid_auc
        ])

        is_best = valid_auc >= best_auc
        best_auc = max(valid_auc, best_auc)
        save_checkpoint({
            'epoch': epoch + 1,
            'netType': args.netType,
            'state_dict': model.state_dict(),
            'best_acc': best_auc,
            'optimizer': optimizer.state_dict(),
        },
                        is_best,
                        predictions,
                        checkpoint=args.checkpoint)

    logger.close()
    logger.plot(['AUC'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(loader,
          model,
          criterion,
          optimizer,
          netType,
          debug=False,
          flip=False,
          train_loader2=None,
          data_iter2=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    model.train()
    end = time.time()

    # rnn = torch.nn.LSTM(10, 20, 2)
    # hidden = torch.autograd.Variable(torch.zeros((args.train_batch)))

    gt_win, pred_win = None, None
    bar = Bar('Training', max=len(loader))
    for i, (inputs, target) in enumerate(loader):
        data_time.update(time.time() - end)

        def train_one_iter(inputs, target, ext=False):
            input_var = inputs.cuda()
            target_var = target.cuda(async=True)

            if debug:
                gt_batch_img = batch_with_heatmap(inputs, target)
                # pred_batch_img = batch_with_heatmap(inputs, score_map)
                if not gt_win or not pred_win:
                    plt.subplot(121)
                    gt_win = plt.imshow(gt_batch_img)
                    # plt.subplot(122)
                    # pred_win = plt.imshow(pred_batch_img)
                else:
                    gt_win.set_data(gt_batch_img)
                    # pred_win.set_data(pred_batch_img)
                plt.pause(.05)
                plt.draw()

            output = model(input_var, branch='ext' if ext else 'main')
            score_map = output[-1].data.cpu()

            # intermediate supervision
            loss = 0
            for o in output:
                # loss += criterion(o, target_var)
                if ext == False:
                    loss += weighted_mse_loss(o, target_var, map_weight)
                    # loss += criterion(o, target_var)
                else:
                    loss += criterion(o, target_var) * .1
            acc, _ = accuracy(score_map, target.cpu(), idx, thr=0.07)

            # no update for extra branch
            if ext == False:
                losses.update(loss.data[0], inputs.size(0))
                acces.update(acc[0], inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_one_iter(inputs, target)
        if data_iter2:
            try:
                inputs, target = next(data_iter2)
            except StopIteration:
                data_iter2 = iter(train_loader2)
                inputs, target = next(data_iter2)
            # train_one_iter(inputs, target, ext=True)

        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=i + 1,
            size=len(loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg)
        bar.next()

    bar.finish()

    return losses.avg, acces.avg


def validate(loader, model, criterion, netType, debug, flip):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    end = time.time()

    # predictions
    predictions = torch.Tensor(loader.dataset.__len__(), 63, 2)

    model.eval()
    gt_win, pred_win = None, None
    bar = Bar('Validating', max=len(loader))
    all_dists = torch.zeros((63, loader.dataset.__len__()))
    for i, (inputs, target, meta) in enumerate(loader):
        data_time.update(time.time() - end)

        input_var = inputs.cuda()
        target_var = target.cuda(async=True)

        output = model(input_var)
        score_map = output[-1].data.cpu()

        # intermediate supervision
        loss = 0
        for o in output:
            loss += criterion(o, target_var)
        acc, batch_dists = accuracy(score_map, target.cpu(), idx, thr=0.07)
        all_dists[:, i * args.val_batch:(i + 1) * args.val_batch] = batch_dists

        preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
        for n in range(score_map.size(0)):
            predictions[meta['index'][n], :, :] = preds[n, :, :]

        if debug:
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, score_map)
            if not gt_win or not pred_win:
                plt.subplot(121)
                gt_win = plt.imshow(gt_batch_img)
                plt.subplot(122)
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        losses.update(loss.data[0], inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=i + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg)
        bar.next()

    bar.finish()
    mean_error = torch.mean(all_dists)
    auc = calc_metrics(all_dists)  # this is auc of predicted maps and target.
    print("=> Mean Error: {:.2f}, AUC@0.07: {} based on maps".format(
        mean_error * 100., auc))
    return losses.avg, acces.avg, predictions, auc


if __name__ == '__main__':
    main(args)
