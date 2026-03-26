echo off

python inference.py ^
--dataset=tmall ^
--data_path=./datasets/ ^
--model_name="tmall.pth" ^
--batch_size=256 ^
--topN="[20]" ^
--cuda ^
--gpu=0 ^
--mean_type=x0 ^
--steps=40 ^
--noise_scale=0.0001 ^
--noise_min=0.0005 ^
--noise_max=0.005 