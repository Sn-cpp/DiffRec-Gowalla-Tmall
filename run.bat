echo off

call conda activate AutoEncoder

python -u main.py ^
    --cuda ^
    --topN="[20]" ^
    --dataset="gowalla" ^
    --lr=3e-5 ^
    --batch_size=512 ^
    --infer_batch_size=1536 ^
    --epochs=1000 ^
    --dims="[2048, 2048]" ^
    --emb_size=32 ^
    --steps=40 ^
    --noise_scale=0.15 ^
    --sampling_steps=0