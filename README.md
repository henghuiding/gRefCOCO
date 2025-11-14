# gRefCOCO - Dataset for [CVPR2023 Highlight] GRES: Generalized Referring Expression Segmentation
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.7%20|%203.8%20|%203.9-blue.svg?style=&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gres-generalized-referring-expression-1/generalized-referring-expression-segmentation)](https://paperswithcode.com/sota/generalized-referring-expression-segmentation?p=gres-generalized-referring-expression-1)

**[🏠[Project page]](https://henghuiding.github.io/GRES/)** &emsp; **[📄[GRES Arxiv]](https://arxiv.org/abs/2306.00968)** &emsp; **[📄[GREC Arxiv]](https://arxiv.org/abs/2308.16182)**

This repository contains information and tools for the [gRefCOCO](https://henghuiding.github.io/GRES/) dataset, proposed by the **CVPR2023 Highlight** paper:
> [GRES: Generalized Referring Expression Segmentation](https://arxiv.org/abs/2306.00968)  
> Chang Liu, Henghui Ding, Xudong Jiang  
> CVPR 2023 Highlight, Acceptance Rate 2.5%

<div align="center">
  <img src="https://github.com/henghuiding/ReLA/blob/main/imgs/fig1.png?raw=true" width="100%" height="100%"/>
</div><br/>

## gRefCOCO Dataset Download

⬇️ Get the gRefCOCO dataset from: 
 - ☁️ [Google Drive](https://drive.google.com/drive/folders/1My2U6SuTAZG9yGBKe_PjsUJJgjdxOiiN?usp=sharing)

## Usage

 - Like RefCOCO, gRefCOCO also should be used together with images from the `train2014` of [MS COCO](https://cocodataset.org/#download). 
 - An example of dataloader [grefer.py](https://github.com/henghuiding/gRefCOCO/blob/main/grefer.py) is provided. 
 - We will update this repository with full API package and documentation soon. Please follow the usage in the [baseline code](https://github.com/henghuiding/ReLA) for now.

## Task 1 - GREC: Generalized Referring Expression Comprehension 
- The GREC evaluation metric code is [here](https://github.com/henghuiding/gRefCOCO/blob/main/mdetr/datasets/refexp.py).


- We provide code based on  [MDETR](https://github.com/ashkamath/mdetr), its training and inference are as follows:


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

1. Obtain `checkpoint.pth` after training or download trained  model [ here ☁️ Google Drive](https://drive.google.com/file/d/14OrM3n_Oap7xCT6nxj9QEnkJOUMpBGjB/view?usp=drive_link)
2. For test results, pass --test and --test_type test or testA or testB according to the dataset.
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_config configs/grefcoco.json --batch_size 4  --resume grefcoco/checkpoint.pth --ema --eval
```
## Task 2 - GRES: Generalized Referring Expression Segmentation 
Please refer to [ReLA](https://github.com/henghuiding/ReLA) for more details.


## Acknowledgement

Our project is built upon [refer](https://github.com/lichengunc/refer) and [cocoapi](https://github.com/cocodataset/cocoapi). Many thanks to the authors for their great works!


## BibTeX
Please consider to cite GRES/GREC if it helps your research.

```bibtex
@inproceedings{GRES,
  title={{GRES}: Generalized Referring Expression Segmentation},
  author={Liu, Chang and Ding, Henghui and Jiang, Xudong},
  booktitle={CVPR},
  year={2023}
}
@article{GREC,
  title={{GREC}: Generalized Referring Expression Comprehension},
  author={He, Shuting and Ding, Henghui and Liu, Chang and Jiang, Xudong},
  journal={arXiv preprint arXiv:2308.16182},
  year={2023}
}
```
We also recommend other highly related works:
```bibtex
@article{VLT,
  title={{VLT}: Vision-language transformer and query generation for referring segmentation},
  author={Ding, Henghui and Liu, Chang and Wang, Suchen and Jiang, Xudong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  volume={45},
  number={6},
  publisher={IEEE}
}
@inproceedings{MeViS,
  title={{MeViS}: A Large-scale Benchmark for Video Segmentation with Motion Expressions},
  author={Ding, Henghui and Liu, Chang and He, Shuting and Jiang, Xudong and Loy, Chen Change},
  booktitle={ICCV},
  year={2023}
}
```
