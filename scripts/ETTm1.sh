
python3 main.py --data ETTm1 \
    --in_len 3 --out_len 24  --forget_num 10000 --monte_carlo_num 32  --std_factor 0.1 --itr 3 --save_pred 

python3 main.py --data ETTm1 \
    --in_len 48 --out_len 48  --forget_num 10000 --monte_carlo_num 16  --std_factor 0.4 --itr 3 --save_pred 

python3 main.py --data ETTm1 \
    --in_len 64 --out_len 96  --forget_num 10000 --monte_carlo_num 16  --std_factor 0.6 --itr 3 --save_pred 

python3 main.py --data ETTm1 \
    --in_len 64 --out_len 288  --forget_num 10000 --monte_carlo_num 16  --std_factor 0.6 --itr 3 --save_pred 

python3 main.py --data ETTm1 \
    --in_len 64 --out_len 672  --forget_num 10000 --monte_carlo_num 16  --std_factor 0.6 --itr 3 --save_pred 

