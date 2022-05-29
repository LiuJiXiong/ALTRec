# Train
# CUDA_VISIBLE_DEVICES=0 python3 train.py --model ALTRec --adv_coeff 100. --num_activeu 300 --p_dims 350 --lam_g 1e-2 --lam_d 1. --dataset ML100K --is_valid True --save_ckpt True  
# CUDA_VISIBLE_DEVICES=0 python3 train.py --model ALTRec --adv_coeff 50. --num_activeu 500 --p_dims 350 --lam_g 1e-2 --lam_d 100. --dataset ML1M --is_valid True --save_ckpt True  
# CUDA_VISIBLE_DEVICES=0 python3 train.py --model ALTRec --adv_coeff 100. --num_activeu 100 --p_dims 350 --lam_g 1e-4 --lam_d 100. --dataset Anime --is_valid True --save_ckpt True  


# Test
# CUDA_VISIBLE_DEVICES=0 python3 train.py --model ALTRec --adv_coeff 100. --num_activeu 300 --p_dims 350 --lam_g 1e-2 --lam_d 1. --dataset ML100K --is_valid False 
CUDA_VISIBLE_DEVICES=0 python3 train.py --model ALTRec --adv_coeff 50. --num_activeu 500 --p_dims 350 --lam_g 1e-2 --lam_d 100. --dataset ML1M --is_valid False 
# CUDA_VISIBLE_DEVICES=0 python3 train.py --model ALTRec --adv_coeff 100. --num_activeu 100 --p_dims 350 --lam_g 1e-4 --lam_d 100. --dataset Anime --is_valid False 
