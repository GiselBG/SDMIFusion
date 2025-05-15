

from PIL import Image
import numpy as np
from torch.autograd import Variable
from taskfusiondataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from FusionNet import FusionNet
from customdataset import CustomDataset
from lossFunction import OhemCELoss, Fusionloss
from optimizer import Optimizer
import torch
torch.cuda.empty_cache()
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

import random 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def seed_everything(seed) -> int:

    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED", _select_seed_randomly(min_seed_value, max_seed_value))
        seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if (seed > max_seed_value) or (seed < min_seed_value):
        log.warning(
            f"{seed} is not in bounds, \
            numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda(args.gpu)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda(args.gpu)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda(args.gpu)
    temp = (im_flat + bias).mm(mat).cuda(args.gpu)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def train_seg(i=0, logger=None,nc=9, n=4):
    model_name = 'segmentation_model.pth'
    modelpth = os.path.join(args.model_path, 'Models')
    load_path = f'{modelpth}/{model_name}'
  
    os.makedirs(modelpth, mode=0o777, exist_ok=True)

    # dataset
    n_img_per_gpu = args.batch_size
    n_workers = 4
    cropsize = [args.cs_width, args.cs_height]
    dataset= args.dataset_path
    ds = CustomDataset(dataset, cropsize=cropsize, mode='train')#, Method=Method)
    dl = DataLoader(
        ds,
        batch_size=n_img_per_gpu,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )

    # model
    ignore_idx = args.ignore_idx #255#0
   
    net = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=False, num_classes=9)
    if i>0:
        net.load_state_dict(torch.load(load_path))
    net.cuda(args.gpu)
    net.train()
    print('Load Pre-trained Segmentation Model:{}!'.format(load_path))
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // n_img_per_gpu
    criteria_p = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(
        thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    # optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    it_start = i*20000
    iter_nums=20000
    max_iter = iter_nums*n

    optim = Optimizer(
        model=net,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power,
        it=it_start,
    )

    # train loop
    msg_iter = 10
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(iter_nums):
        try:
            im, lb, _ = next(diter)
            if not im.size()[0] == n_img_per_gpu:
                raise StopIteration
        except StopIteration:
            epoch += 1
            # sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb, _ = next(diter)
        im = im.cuda(args.gpu)
        lb = lb.cuda(args.gpu)
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        out = net(im)
        out = out["out"]

       
        loss = criteria_p(out, lb)
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        # print training log message
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)

            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int(( max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join(
                [
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]
            ).format(
                it=it_start+it + 1, max_it= max_iter, lr=lr, loss=loss_avg, time=t_intv, eta=eta
            )
            logger.info(msg)
            loss_avg = []
            st = ed
    # dump the final model
    save_pth = osp.join(modelpth, model_name)
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info(
        'Segmentation Model Training done~, The Model is saved to: {}'.format(
            save_pth)
    )
    logger.info('\n')

def train_fusion(num=0, logger=None, nc=9 ):
    # num: control the segmodel 
    lr_start = 0.001
    modelpth = args.model_path
    dataset = args.dataset_path
    #Method = 'Fusion'
    modelpth = os.path.join(modelpth, 'Models')
    fusionmodel = eval('FusionNet')(output=1)
    fusionmodel.cuda(args.gpu)
    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    if num>0:
        #Segmentation Model
        segmodel = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=False, num_classes=nc)
        save_pth = osp.join(modelpth, 'segmentation_model.pth')
        if logger == None:
            logger = logging.getLogger()
            setup_logger(modelpth)
        segmodel.load_state_dict(torch.load(save_pth))
        segmodel.cuda(args.gpu)
        segmodel.eval()
        for p in segmodel.parameters():
            p.requires_grad = False
        print('Load Segmentation Model {} Sucessfully~'.format(save_pth))
    
    train_dataset = Fusion_dataset(split='train', ir_path=dataset+'/Infrared/', vi_path=dataset+'/Visible/', lb_path=dataset+'/Label/')
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    # 
    if num>0:
        score_thres = 0.7
        ignore_idx = args.ignore_idx
        n_min = 8 * args.cs_width * args.cs_height // 8
        criteria_p = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
        criteria_16 = OhemCELoss(
            thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_fusion = Fusionloss()

    epoch = 10
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    for epo in range(0, epoch):
        # print('\n| epo #%s beg in...' % epo)
        lr_start = 0.001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
            fusionmodel.train()
            image_vis = Variable(image_vis).cuda(args.gpu)
            image_vis_ycrcb = RGB2YCrCb(image_vis).cuda(args.gpu)
            image_ir = Variable(image_ir).cuda(args.gpu)
            label = Variable(label).cuda(args.gpu)
            logits = fusionmodel(image_vis_ycrcb, image_ir).cuda(args.gpu)
            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :],
                 image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            lb = torch.squeeze(label, 1)
            optimizer.zero_grad()
            # seg loss
            if num>0:
                out = segmodel(fusion_image)
                out = out["out"]
                seg_loss = criteria_p(out, lb)
            # fusion loss
            loss_fusion, loss_in, loss_grad, loss_color ,loss_ssim= criteria_fusion(
                image_vis_ycrcb, image_ir, label, logits,num,fusion_ycrcb
            )
            if num>0:
                loss_total = loss_fusion + (num) * seg_loss
            else:
                loss_total = loss_fusion
            loss_total.backward()
            optimizer.step()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                if num>0:
                    loss_seg=seg_loss.item()
                else:
                    loss_seg=0
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_color: {loss_color:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',
                        'loss_seg: {loss_seg:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_color=loss_color.item(),
                    loss_ssim=loss_ssim.item(),
                    loss_seg=loss_seg,
                    time=t_intv,
                    eta=eta,
                )
                logger.info(msg)
                st = ed
    fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
    os.makedirs(modelpth, exist_ok=True)
    torch.save(fusionmodel.state_dict(), fusion_model_file)
    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')

def run_fusion(type='train'):
    fusion_model_path = f'{args.model_path}/Models/fusion_model.pth'
    dataset= args.dataset_path
    fused_dir = os.path.join(dataset,'Fusion')
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    fusionmodel = eval('FusionNet')(output=1)
    fusionmodel.eval()
    if args.gpu >= 0:
        fusionmodel.cuda(args.gpu)
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('done!')
    vi_path=os.path.join(dataset,'Visible')
    ir_path=os.path.join(dataset,'Infrared')
    lb_path=os.path.join(dataset,'Label')
    test_dataset = Fusion_dataset(split=type, vi_path=vi_path, ir_path=ir_path, lb_path=lb_path)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, labels, name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            labels = Variable(labels)
            if args.gpu >= 0:
                images_vis = images_vis.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)
                labels = labels.cuda(args.gpu)
            images_vis_ycrcb = RGB2YCrCb(images_vis)
            logits = fusionmodel(images_vis_ycrcb, images_ir)
            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :,
                 :], images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(
                fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Sucessfully!'.format(save_path))


if __name__ == "__main__":
    seed_everything(23)
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_path', '-MP', type=str, default='./model')
    parser.add_argument('--dataset_path', '-DP', type=str, default='./dataset')
    parser.add_argument('--batch_size', '-B', type=int, default=10)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    parser.add_argument('--seg', '-s', type=int, default=1)
    parser.add_argument('--nclasses', '-nc', type=int, default=9)
    parser.add_argument('--ignore_idx', '-ii', type=int, default=255)
    parser.add_argument('--cs_height', '-csh', type=int, default=480)
    parser.add_argument('--cs_width', '-csw', type=int, default=640)
    parser.add_argument('--n', '-n', type=int, default=4)
    parser.add_argument('--c', '-c', type=int, default=0)
    args = parser.parse_args()
  
    logpath=os.path.join(args.model_path, 'TrainLog')
    logger = logging.getLogger()
    setup_logger(logpath)

    start = time.time()

    for i in range(args.c,args.n):
        print(f"loop: {i}")
        if args.seg == 0:
            train_fusion(args.seg, logger,args.nclasses)
            if i == (args.n - 1):
                run_fusion('train')  
                print("|{0} Fusion Image Sucessfully~!".format(i + 1))
                for j in range(args.n):
                    train_seg(j, logger,args.nclasses,args.n)
                    print("|{0} Train Segmentation Model Sucessfully~!".format(i + 1))
        else:
            train_fusion(i, logger,args.nclasses)
            print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
            run_fusion('train')  
            print("|{0} Fusion Image Sucessfully~!".format(i + 1))
            train_seg(i, logger,args.nclasses,args.n)
            print("|{0} Train Segmentation Model Sucessfully~!".format(i + 1))
    end = time.time()
    execution_time = (end - start)/60
    print(f"Execution time {execution_time} minutes")
  