# gRefCOCO - Dataset for [CVPR2023 Highlight] GRES: Generalized Referring Expression Segmentation

**[🏠[Project page]](https://henghuiding.github.io/GRES/)** &emsp; **[📄[Arxiv]](https://arxiv.org/abs/2306.00968)**

This repository contains information and tools for the [gRefCOCO](https://henghuiding.github.io/GRES/) dataset.


## Download

⬇️ Get the dataset from: 
 - ☁️ [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/liuc0058_e_ntu_edu_sg/EaEz86LZwtNBmUdD4oMo9TkBBJ5Kft-ctoxyJ4cFhsNlHQ?e=ibiOi4)
 - ☁️ [Google Drive](https://drive.google.com/file/d/1WXifNjJ8gKQAcPQWcCkCdNYvRrkS6nbQ/view?usp=sharing)


## Usage

Like RefCOCO, gRefCOCO also should be used together with images from the `train2014` of [MS COCO](https://cocodataset.org/#download). 

We will update this repository with full API package and documentation soon. Please follow the usage in the [baseline code](https://github.com/henghuiding/ReLA) for now.

## GREC: Generalized Referring Expression Comprehension Task 
We provide code based on  [MDETR](https://github.com/ashkamath/mdetr)


### Training (Finetuning)
1. Process grefcoco to coco format.
```
python scripts/fine-tuning/grefexp_coco_format.py --data_path xxx --out_path mdetr_annotations/ --coco_path xxx
```
2. Training and download `pretrained_resnet101_checkpoint.pth` from [MDETR](https://github.com/ashkamath/mdetr)
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/grefcoco.json --batch_size 4  --load pretrained_resnet101_checkpoint.pth  --ema --text_encoder_lr 1e-5 --lr 5e-5 --output-dir grefcoco
```
### Inference

1. Obtain `checkpoint.pth` after training or download trained  model [ here ☁️ Google Drive](https://drive.google.com/file/d/1djNwwNAyAIEJMZIQQHV_NYnlc8TeA4wU/view?usp=drive_link)
2. For test results, pass --test and --test_type test or testA or testB according to the dataset.
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/grefcoco.json --batch_size 4  --resume grefcoco/checkpoint.pth --ema --eval
```
## GRES: Generalized Referring Expression Segmentation Task 
Please refer to [ReLA](https://github.com/henghuiding/ReLA) for more detail.


## Acknowledgement

Our project is built upon [refer](https://github.com/lichengunc/refer) and [cocoapi](https://github.com/cocodataset/cocoapi). Many thanks to the authors for their great works!


## BibTeX
Please consider to cite GRES if it helps your research.

```latex
@inproceedings{GRES,
  title={{GRES}: Generalized Referring Expression Segmentation},
  author={Liu, Chang and Ding, Henghui and Jiang, Xudong},
  booktitle={CVPR},
  year={2023}
}
```
