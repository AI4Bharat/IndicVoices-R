#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8

dataset=ivr
mkdir -p ./logs/${dataset}

exp_root="logs"
exp_name=e830M
dataset_dir="datasets/ivr/"
encodec_codes_folder_name="encodec_16khz_4codebooks"
load_model_from="checkpoints/e830M/best_bundle.pth"


torchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:41980 --nproc_per_node=${WORLD_SIZE} \
./main.py \
--load_model_from ${load_model_from} \
--reduced_eog 1 \
--drop_long 1 \
--eos 2051 \
--n_special 4 \
--pad_x 0 \
--codebook_weight "[3,1,1,1]" \
--encodec_sr 50 \
--num_steps 1000000 \
--lr 0.00001 \
--warmup_fraction 0.1 \
--optimizer_name "AdamW" \
--d_model 2048 \
--audio_embedding_dim 2048 \
--nhead 16 \
--num_decoder_layers 16 \
--max_num_tokens 50000 \
--gradient_accumulation_steps 12 \
--val_max_num_tokens 6000 \
--num_buckets 6 \
--audio_max_length 30 \
--audio_min_length 0.2 \
--text_max_length 400 \
--text_min_length 10 \
--mask_len_min 1 \
--mask_len_max 600 \
--tb_write_every_n_steps 10 \
--print_every_n_steps 400 \
--val_every_n_steps 400 \
--text_vocab_size 1536 \
--text_pad_token 1536 \
--phn_folder_name "phonemes" \
--manifest_name "manifest" \
--encodec_folder_name ${encodec_codes_folder_name} \
--audio_vocab_size 2048 \
--empty_token 2048 \
--eog 2049 \
--audio_pad_token 2050 \
--n_codebooks 4 \
--max_n_spans 3 \
--shuffle_mask_embedding 0 \
--mask_sample_dist poisson1 \
--max_mask_portion 0.9 \
--min_gap 5 \
--num_workers 128 \
--dynamic_batching 1 \
--dataset $dataset \
--exp_dir "${exp_root}/${dataset}/${exp_name}" \
--dataset_dir ${dataset_dir} \
--use_wandb \
--wandb_project ivr \
--resume

