# coding:utf-8
import os
import argparse
from utils import *
import torch
from torch.utils.data import DataLoader
from datasets import Fusion_dataset
from FusionNet import FusionNet
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def run(args=None):
    ir_dir=args.ir_dir
    vi_dir=args.vi_dir
    save_dir=args.fusion_save_dir
    fusion_model_path=args.fusion_model_path
    fusionmodel = FusionNet(output=1)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    fusionmodel = fusionmodel.to(device)
    print('fusionmodel load done!')
    test_dataset = Fusion_dataset('val', ir_path=ir_dir, vi_path=vi_dir)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (img_vis, img_ir, name) in enumerate(test_bar):
            img_vis = img_vis.to(device)
            img_ir = img_ir.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            fused_img = fusionmodel(vi_Y, img_ir)
            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            for k in range(len(name)):
                img_name = name[k]
                save_path = os.path.join(save_dir, img_name)
                save_img_single(fused_img[k, ::], save_path)
                test_bar.set_description('Fusion {0} Sucessfully!'.format(name[k]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SeAFusiuon with pytorch')
    ## model
    parser.add_argument('--fusion_model_path', '-M', type=str, default='./Models/Fusion/fusion_model.pth')
    ## dataset
    parser.add_argument('--ir_dir', '-ir_dir', type=str, default='./test_imgs/ir')
    parser.add_argument('--vi_dir', '-vi_dir', type=str, default='./test_imgs/vi')
    parser.add_argument('--fusion_save_dir', '-fusion_dir', type=str, default='./SDMIFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    local_args = parser.parse_args()
    os.makedirs(local_args.fusion_save_dir, exist_ok=True)
    print('| Fusion testing %s on GPU #%d with pytorch' % ('SDMIFusion', local_args.gpu))
    run(args=local_args)