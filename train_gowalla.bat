echo off

python main.py ^
--topN="[20]" ^
--cuda ^
--dataset=gowalla ^
--data_path=./datasets/gowalla/ ^
--lr=0.001 ^
--weight_decay=1e-5 ^
--batch_size=256 ^
--epochs=100 ^
--dims="[512, 1024]" ^
--emb_size=32 ^
--mean_type=x0 ^
--steps=40 ^
--noise_scale=0.1 ^
--noise_min=0.0001 ^
--noise_max=0.02 ^
--sampling_steps=5 ^
--reweight=1 ^
--log_name=fast ^
--round=1 ^
--gpu=0
