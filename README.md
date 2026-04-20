# Diffusion Recommender Model
This is a fork of YiyanXu/DiffRec repository which contains the pytorch implementation of their paper at SIGIR 2023:
> [Diffusion Recommender Model](https://arxiv.org/abs/2304.04971)
> 
> Wenjie Wang, Yiyan Xu, Fuli Feng, Xinyu Lin, Xiangnan He, Tat-Seng Chua

The goal of this fork is to test the behavior of __DiffRec__ (base version only) on four standard datasets: Gowalla, Movielens 1M, Yelp, Amazon Books datasets, along with our collected Steam dataset. 

This repository contains adjustments for the base DiffRec only, made to work on:
+ python 3.12.12
+ torch 2.9.1+cu128
+ numpy 2.4.3
+ scipy 1.17.1

The code has been modernized to work with newer version of Scipy by replacing ```scipy.sparse.csr_matrix.A``` (which is deprecated) with ```scipy.sparse.csr_matrix.toarray()```.

Since all five datasets are implicit type, all train/validation/test sparse matrices read from files have been adjusted to ```int8``` data-type for more efficient RAM usage.

Including additional performance optimizations, code refinements, and other improvements.
# Usage

__Notes__: Please install and activate the correct environment from `requirements.txt`.

## To perform training, for example:
```
python -u main.py ^
    --cuda ^
    --topN="[20]" ^
    --dataset="gowalla" ^
    --tst_w_val ^
    --lr=1e-4 ^
    --weight_decay=1e-5 ^
    --batch_size=512 ^
    --infer_batch_size=1024 ^
    --epochs=1000 ^
    --dims="[1000, 1000]" ^
    --emb_size=12 ^
    --steps=60 ^
    --noise_scale=0.125 ^
    --noise_min=1e-5 ^
    --noise_max=0.02 ^
    --sampling_steps=0
```
Or use
```
python -u main.py --help
```
for the list of training arguments.


## To perform inference, for example:
```
python -u inference.py ^
    --cuda ^
    --dataset="gowalla" ^
    --topN="[20, 40]" ^
    --model_path="./saved_models/" ^
    --model_name="your_model_name.pth"    
```
Or use
```
python -u inference.py --help
```
for the list of inference arguments.

# Checkpoint files
We release the checkpoint files (.pth) of trained models [here](https://drive.google.com/drive/folders/1EdTHEF2wfCaKONbfLfpkKh4A9dmTGbyf?usp=sharing).

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

