python3 main.py --data ETTh1 \
    --in_len 24 --out_len 24  --forget_num 10000 --monte_carlo_num 32  --std_factor 0.3 --itr 3 --save_pred 

python3 main.py --data ETTh1 \
    --in_len 48 --out_len 48  --forget_num 10000 --monte_carlo_num 16  --std_factor 0.4 --itr 3 --save_pred 

python3 main.py --data ETTh1 \
    --in_len 64 --out_len 168  --forget_num 10000 --monte_carlo_num 16  --std_factor 0.6 --itr 3 --save_pred 

python3 main.py --data ETTh1 \
    --in_len 64 --out_len 336  --forget_num 10000 --monte_carlo_num 16  --std_factor 0.6 --itr 3 --save_pred 

python3 main.py --data ETTh1 \
    --in_len 64 --out_len 720  --forget_num 10000 --monte_carlo_num 16  --std_factor 0.6 --itr 3 --save_pred 

