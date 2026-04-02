echo off

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
