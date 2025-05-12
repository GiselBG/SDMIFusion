#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from datasetSegmentation import dataset
# import cv2
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

import fusion
import segmentation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SDMIFusion with pytorch')
    parser.add_argument('--model_path', '-rp', type=str, default='./Models')
    #fusion
    parser.add_argument('--fusion_model_path', '-M', type=str, default='./Models/Fusion/fusion_model.pth')
     ## dataset
    parser.add_argument('--ir_dir', '-id', type=str, default='./test_imgs/ir')
    parser.add_argument('--vi_dir', '-vd', type=str, default='./test_imgs/vi')
    parser.add_argument('--lb_dir', '-ld', type=str, default='./test_imgs/Label')
    parser.add_argument('--fusion_save_dir', '-fd', type=str, default='./SDMIFusion')
    parser.add_argument('--segmentation_save_dir', '-sd', type=str, default='/NewResult')    
    parser.add_argument('--n_classes', '-nc', type=int, default=9)    
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    local_args = parser.parse_args()

    #run fusion
    os.makedirs(local_args.fusion_save_dir, exist_ok=True)
    print('| Fusion testing %s on GPU #%d with pytorch' % ('SDMIFusion', local_args.gpu))
    fusion.run(args=local_args)

    #run and evaluate Segmentation
    log_dir = os.path.join(local_args.model_path, 'SegmentationTestLog')
    os.makedirs(log_dir, exist_ok=True)
    setup_logger(log_dir)
    segmentation.evaluate(model_path=local_args.model_path, fusion_save_dir =local_args.fusion_save_dir ,   lb_dir =local_args.lb_dir,  folder = local_args.segmentation_save_dir, nc=local_args.n_classes)