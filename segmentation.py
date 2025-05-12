
from logger import setup_logger
from datasetSegmentation import dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import math
from PIL import Image
from utils import SegmentationMetric
	
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


class MscEval(object):
    def __init__(
        self,
        model,
        dataloader,
        scales=[0.75, 0.9, 1, 1.1, 1.2, 1.25],
        n_classes=9,
        lb_ignore=255,
        cropsize=480,
        flip=True,
        *args,
        **kwargs
    ):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        self.cropsize = cropsize
        ## dataloader
        self.dl = dataloader
        self.net = model

    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0] - H, size[1] - W
        hst, hed = margin_h // 2, margin_h // 2 + H
        wst, wed = margin_w // 2, margin_w // 2 + W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]

    def get_palette(self):
        unlabelled = [0, 0, 0]
        car = [64, 0, 128] #morado
        person = [64, 64, 0] #olive
        bike = [0, 128, 192] #celeste
        curve = [0, 0, 192] #azul
        carstop = [128, 128, 0] #olive claro
        guardrail = [64, 64, 128] #morado dark
        colorcone = [192, 128, 128] #rosado
        bump = [192, 64, 0] #rojo
    
        palette = np.array(
            [
                unlabelled,
                car,
                person,	
                bike,	
                curve,	
                carstop,
                guardrail,
                colorcone,
                bump
            ]
        )
        return palette

    def visualize(self, file_path, predictions):
        palette = self.get_palette()
        pred = predictions
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(0, int(predictions.max()+1)):
            img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))    
        img.save(file_path)

    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)["out"]
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop)["out"]
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
            prob = torch.exp(prob)#
        return prob

    def crop_eval(self, im):
        cropsize = self.cropsize
        stride_rate = 5 / 6.0
        N, C, H, W = im.size()
        long_size, short_size = (H, W) if H > W else (W, H)
        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0] : indices[1], indices[2] : indices[3]]
          
        else:
            stride = math.ceil(cropsize * stride_rate)
            if short_size < cropsize:
                if H < W:
                    im, indices = self.pad_tensor(im, (cropsize, W))
                else:
                    im, indices = self.pad_tensor(im, (H, cropsize))
            N, C, H, W = im.size()
            n_x = math.ceil((W - cropsize) / stride) + 1
            n_y = math.ceil((H - cropsize) / stride) + 1
            prob = torch.zeros(N, self.n_classes, H, W).cuda()
            prob.requires_grad = False
            for iy in range(n_y):
                for ix in range(n_x):
                    hed, wed = (
                        min(H, stride * iy + cropsize),
                        min(W, stride * ix + cropsize),
                    )
                    hst, wst = hed - cropsize, wed - cropsize
                    chip = im[:, :, hst:hed, wst:wed]
                    prob_chip = self.eval_chip(chip)
                    prob[:, :, hst:hed, wst:wed] += prob_chip
            if short_size < cropsize:
                prob = prob[:, :, indices[0] : indices[1], indices[2] : indices[3]]
        return prob

    def scale_crop_eval(self, im, scale):
        N, C, H, W = im.size()
        new_hw = [int(H * scale), int(W * scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob

    def compute_hist(self, pred, lb, n_classes):
        ignore_idx = self.lb_ignore
        keep = np.logical_not(lb == ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes ** 2)
        hist = hist.reshape((n_classes, n_classes))
        return hist

    def evaluate(self, folder='NewResult'):
        
        n_classes = self.n_classes
        print(n_classes)
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
        lb_ignore = [255] 
        seg_metric = SegmentationMetric(n_classes, device=device)          
        dloader = tqdm(self.dl)
        if dist.is_initialized() and not dist.get_rank() == 0:
            dloader = self.dl
        for i, (imgs, label, fn) in enumerate(dloader):
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.n_classes, H, W))
            
            probs.requires_grad = False
            imgs = imgs.cuda()
            probs_torch = torch.zeros((N, self.n_classes, H, W))
            probs_torch = probs_torch.to(device)
            probs_torch.requires_grad = False
            for sc in self.scales:
                prob = self.scale_crop_eval(imgs, sc)
                probs_torch += prob
                probs += prob.detach().cpu()            
            seg_results = torch.argmax(probs_torch, dim=1, keepdim=True)
            seg_metric.addBatch(seg_results, label.to(device), lb_ignore)
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)
            for i in range(1):
                outpreds = preds[i]
                name = fn[i]
                folder_path = folder 
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_path = os.path.join(folder_path, name)
                
                self.visualize(file_path, outpreds)
            label=label.data.numpy().squeeze(1)
            hist_once = self.compute_hist(preds, label,n_classes)
            hist = hist + hist_once
        div1 = (np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist)) 
        div1[div1==0]=1
        print(div1)
    
        IOUs = np.diag(hist) /div1  
        print(lb_ignore[0])
        if lb_ignore[0] == 0:
            IOUs=IOUs[1:]
        print(IOUs.shape)
        mIOU = np.mean(IOUs)
        mIOU = mIOU

        IoU_list = IOUs.tolist()
        IoU_list = [round(100 * i, 2) for i in IoU_list]
        return mIOU, IoU_list


def evaluate(model_path='./Models', fusion_save_dir ='./FusedImage', lb_dir='./Label', segmentation_model=None, folder=None ,nc=9):
    ## logger
    logger = logging.getLogger()
    ## model
    logger.info('\n')
    logger.info('====' * 4)
    logger.info('evaluating the model ...\n')
    n_classes = nc 
    net = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=False, num_classes=n_classes)
   
    if segmentation_model==None:
        segmentation_model = osp.join(model_path, 'segmentation_model.pth')
    net.load_state_dict(torch.load(segmentation_model))
    net.cuda()
    net.eval()

    ## dataset
    batchsize = 1
    n_workers = 2
    dsval = dataset(fusion_save_dir , lb_dir, mode='test')
    dl = DataLoader(
        dsval,
        batch_size=batchsize,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
    )

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(n_classes=nc,model=net, dataloader=dl)

    ## eval
    mIOU, IoU_list = evaluator.evaluate(folder=folder)
    logger.info(' IoU:{}, mIoU:{:.4f}'.format( IoU_list, mIOU))
    return mIOU


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SDMIFusion with pytorch')
    parser.add_argument('--model_path', '-rp', type=str, default='./Models')
    parser.add_argument('--fusion_save_dir', '-fd', type=str, default='./Result')
    parser.add_argument('--lb_dir', '-lp', type=str, default='./Label')
    parser.add_argument('--segmentation_save_dir', '-sr', type=str, default='/SegmentationResult')    
    parser.add_argument('--n_classes', '-nc', type=int, default=9)    
    args = parser.parse_args()
    log_dir = os.path.join(args.model_path, 'SegmentationTestLog')
    os.makedirs(log_dir, exist_ok=True)

    setup_logger(log_dir)
    print(f"args: {args}")
    evaluate(model_path=args.model_path, 
             fusion_save_dir =args.fusion_save_dir ,   
             lb_dir=args.lb_dir,
             folder = args.segmentation_save_dir, 
             nc=args.n_classes)


