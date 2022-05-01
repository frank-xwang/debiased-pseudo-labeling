# Debiased Pseudo-Labeling

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/debiased-learning-from-naturally-imbalanced/few-shot-image-classification-on-imagenet-0)](https://paperswithcode.com/sota/few-shot-image-classification-on-imagenet-0?p=debiased-learning-from-naturally-imbalanced)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/debiased-learning-from-naturally-imbalanced/semi-supervised-image-classification-on-16)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-16?p=debiased-learning-from-naturally-imbalanced)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/debiased-learning-from-naturally-imbalanced/semi-supervised-image-classification-on-1)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-1?p=debiased-learning-from-naturally-imbalanced)

This repository contains the code (in PyTorch) for the model introduced in the following paper:

**Debiased Learning from Naturally Imbalanced Pseudo-Labels**<br>
[Xudong Wang](http://people.eecs.berkeley.edu/~xdwang/), [Zhirong Wu](https://www.microsoft.com/en-us/research/people/wuzhiron/), [Long Lian](https://github.com/TonyLianLong/), and [Stella X. Yu](http://www1.icsi.berkeley.edu/~stellayu/)<br>
UC Berkeley and Microsoft Research<br>
[CVPR 2022](https://cvpr2022.thecvf.com)

[Project Page](https://people.eecs.berkeley.edu/~xdwang/projects/DebiasPL/) | [Preprint](https://arxiv.org/abs/2201.01490) | [Citation](#citation)

<p align="center">
  <img src="https://github.com/frank-xwang/debiased-pseudo-labeling/blob/main/DebiasPL.gif" width=70%>
</p>

<p align="center">
  <img align="center" src="https://github.com/frank-xwang/debiased-pseudo-labeling/blob/main/result.png" width=57%>
  <img align="center" src="https://github.com/frank-xwang/debiased-pseudo-labeling/blob/main/ZSL-DomainShift.png" width=40%>
</p>

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@inproceedings{wang2022debiased,
  title={Debiased Learning from Naturally Imbalanced Pseudo-Labels},
  author={Wang, Xudong and Wu, Zhirong and Lian, Long and Yu, Stella X},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={},
  year={2022}
}

```

## Updates
[04/2022] Initial Commit.

## Requirements
### Packages
* Python >= 3.7, < 3.9
* PyTorch >= 1.6
* torchaudio==0.7.2
* tensorboard >= 1.14 (for visualization)
* tqdm
* faiss-gpu
* pandas
* apex (optional, unless using mixed precision training)

### Hardware requirements
8 GPUs with >= 11G GPU RAM or 4 GPUs with >= 16G GPU RAM are recommended.

## Dataset and Pre-trained Model Preparation
Please download pre-trained [MoCo-EMAN model](https://eman-cvpr.s3.amazonaws.com/models/res50_moco_eman_800ep.pth.tar), make a new folder called pretrained and place checkpoints under it. Please download the ImageNet dataset from [this link](http://www.image-net.org/). Then, move and extract the training and validation images to labeled subfolders, using the following [shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh). 
The indexes for semi-supervised learning experiments can be found at [here](https://drive.google.com/drive/folders/18fi_lxZQK_J_E9Uam0c9G1VCfnKYyhmF?usp=sharing). The setting with 1% labeled data is the same as FixMatch. A new list of indexes is made for the setting with 0.2% labeled data by randomly selecting 0.2% of instances from each class. Please put all CSV files in the same location as below:

```
dataset
└── imagenet
    ├── indexes
    │   ├── train_1p_index.csv
    │   ├── train_99p_index.csv
    |   └── ....
    ├── train
    │   ├── n01440764
    │   │   └── *.jpeg
    |   └── ....
    └── val
        ├── n01440764
        │   └── *.jpeg
        └── ....
```

## Training and Evaluation Instructions
### Semi-supervised learning on ImageNet-1k
0.2% labeled data (50 epochs):
```
bash scripts/0.2perc-ssl/train_DebiasPL.sh
```
1% labeled data (50 epochs):
```
bash scripts/1perc-ssl/train_DebiasPL.sh
```

| Method                | epochs            | 0.2% labels       | 1% labels 
| --------------        | ----------------  | ----------------  | ----------------
| FixMatch w/ EMAN      | 50                | 43.6%             | 60.9% 
| DebiasPL (reported)   | 50                | 51.6%             | 65.3% 
| DebiasPL (reproduced) | 50                | 52.0% [[checkpoint](https://drive.google.com/file/d/1_mCbwMokj8WFE5H0LN77TbuWxKLBdGRV/view?usp=sharing) \| [log](https://drive.google.com/file/d/1Wy6hJgvcuWc2sgwMq_fwq9OseTVVmgU9/view?usp=sharing)] | 65.6% [[checkpoint](https://drive.google.com/file/d/1QXO7icAonToHhjmBMNn3rwk7Qf1c0LG9/view?usp=sharing) \| [log](https://drive.google.com/file/d/1FmBJqb5YP1PZaQFW5AIFm4JO9c4xGcx8/view?usp=sharing)]

The results reproduced by this codebase are often slightly higher than what was reported in the paper (52.0 vs 51.6; 65.6 vs. 65.3). We find it beneficial to apply cross-level instance-group discrimination loss [CLD](https://arxiv.org/pdf/2008.03813.pdf) to unlabeled instances to leverage their information fully. 

### Zero-shot learning
Please [download](https://drive.google.com/drive/folders/1mAB49eceMmu0hHfEofcNOCzi1nTxsEon?usp=sharing) zero-shot predictions with a pre-trained CLIP (backbone: RN50) model and put them under imagenet/indexes/. Then run experiments on ImageNet-1k with:
```
bash scripts/zsl/train_DebiasPL.sh
```

| Method                | epochs            | top-1 acc
| --------------        | ----------------  | ---------------- 
| CLIP                  | -                 | 59.6%         
| DebiasPL (reported)   | 100               | 68.3%
| DebiasPL (reproduced) | 50                | 68.7% [[checkpoint](https://drive.google.com/file/d/1fFpLi8WlYy-ByFIV0wPg4r9F-X-W0hhL/view?usp=sharing) \| [log](https://drive.google.com/file/d/19LErjfTtkyOtZb4gae8ODR-lWnWF95Kn/view?usp=sharing)]

## How to get support from us?
If you have any general questions, feel free to email us at `xdwang at eecs.berkeley.edu`. If you have code or implementation-related questions, please feel free to send emails to us or open an issue in this codebase (We recommend that you open an issue in this codebase, because your questions may help others). 

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details. The parts described below follow their original license.

## Acknowledgements
Part of the code is based on [EMAN](https://github.com/amazon-research/exponential-moving-average-normalization), [FixMatch](https://github.com/kekmodel/FixMatch-pytorch), [CLIP](https://github.com/openai/CLIP), [CLD](https://github.com/frank-xwang/CLD-UnsupervisedLearning), and [LA](https://github.com/google-research/google-research/tree/master/logit_adjustment).
