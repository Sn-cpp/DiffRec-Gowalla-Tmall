echo off

python -u inference.py ^
    --cuda ^
    --dataset="gowalla" ^
    --topN="[20, 40]" ^
    --model_path="./saved_models/" ^
    --model_name="gowalla_lr0.0001_wd1e-05_bs512_ibs1024_dims[1000, 1000]_emb12_x0_steps60_scale0.125_min1e-05_max0.02_sample0_reweightTrue_log.pth"