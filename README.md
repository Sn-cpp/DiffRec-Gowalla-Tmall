# Diffusion Recommender Model
This is a fork of YiyanXu/DiffRec repository which contains the pytorch implementation of their paper at SIGIR 2023:
> [Diffusion Recommender Model](https://arxiv.org/abs/2304.04971)
> 
> Wenjie Wang, Yiyan Xu, Fuli Feng, Xinyu Lin, Xiangnan He, Tat-Seng Chua

The point of this fork is to test the behavior of __DiffRec__ (base version only) on the Gowalla and Tmall datasets, in comparsion with _LightGCN_, _SimGCL_ and _LightGCL_. The datasets is provided from the [LightGCL repository](https://github.com/HKUDS/LightGCL?tab=readme-ov-file).

This repository contains adjustments for the base DiffRec only, made to work on:
+ python 3.12.12
+ torch 2.9.1+cu128
+ numpy 2.4.3

# Usage

__Notes__: If you are looking for the detailed arguments list, please refer to the [DiffRec repository](https://github.com/YiyanXu/DiffRec) here.

To perform training on the Gowalla dataset: run the `train_gowalla.bat`

To perform inference on the Gowalla dataset: run the `infer_gowalla.bat`

To perform training on the Tmall dataset: run the `train_tmall.bat` 

To perform inference on the Tmall dataset: run the `infer_tmall.bat`

And adjust the arguments inside the `.bat` file if needed.

# Citation

We value the work of __Wenjie Wang, Yiyan Xu, Fuli Feng, Xinyu Lin, Xiangnan He and Tat-Seng Chua__, please kindly cite their paper:
```
@inproceedings{wang2023diffrec,
title = {Diffusion Recommender Model},
author = {Wang, Wenjie and Xu, Yiyan and Feng, Fuli and Lin, Xinyu and He, Xiangnan and Chua, Tat-Seng},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {832–841},
publisher = {ACM},
year = {2023}
}
```

