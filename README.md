<div align="center">

<h1> SDMIFusion </h1>
</div>
<div align="center">
 <em>
  SDMIFusion: A Deep Learning Framework for Segmentation-Driven Multimodal Image Fusion 
 </em>
</div>
<img src="sdmifusion.png" alt="SDMIFusion Architecture" style="display:block; margin-left:auto; margin-right:auto;">
 <br>
 
### Overview

SDMIFusion is a task-driven multimodal image fusion framework designed to improve semantic segmentation by effectively combining visible and infrared image modalities. The framework utilizes DeepLabV3+ as the segmentation backbone, and its modular design allows easy adaptation to other high-performance segmentation models depending on the specific application or computational requirements.

---

### Train the Model

To train the SDMIFusion model, run the following command:

```bash
python ./train.py --model_path './Model' --dataset_path './dataset' --batch_size 10 --gpu 0 --num_workers 4 --seg 1 --nclasses 9

```
#### Parameters

The training script supports the following arguments:

| Argument        | Alias | Type | Default    | Description                                    |
|-----------------|-------|------|------------|------------------------------------------------|
| `--model_path`  | `-MP` | str  | `./model`  | Path to save or load model weights             |
| `--dataset_path`| `-DP` | str  | `./dataset`| Path to dataset directory                       |
| `--batch_size`  | `-B`  | int  | 10         | batch size                                |
| `--gpu`         | `-G`  | int  | 0          | GPU device ID                                  |
| `--num_workers` | `-j`  | int  | 4          | Number of workers for data loading             |
| `--seg`         | `-s`  | int  | 1          | Segmentation loop (1 = yes, 0 = no)    |
| `--nclasses`    | `-nc` | int  | 9          | Number of segmentation classes                 |
| `--ignore_idx`  | `-ii` | int  | 255        | Label index to ignore during training          |
| `--cs_height`   | `-csh`| int  | 480        | Input image crop height                         |
| `--cs_width`    | `-csw`| int  | 640        | Input image crop width                          |

### Test the Model

The test script performs the multimodal fusion process by combining the infrared and visible images using the trained fusion model. After fusion, it applies the segmentation model to produce the final segmentation results.

To run inference with the trained SDMIFusion model, use the following command:

```bash
python ./test.py --fusion_model_path './fusion_model.pth' --ir_dir './Infrared' --vi_dir './Visible' --save_dir './SegmentationResults'
```

### Acknowledgments

We thank the developers of DeepLabV3+ for providing a powerful and flexible segmentation backbone that forms the foundation of our framework. We also acknowledge the authors of SeaFusion for their pioneering work in task-driven multimodal image fusion, which inspired aspects of our fusion approach. Their contributions have been invaluable to the development of SDMIFusion.
