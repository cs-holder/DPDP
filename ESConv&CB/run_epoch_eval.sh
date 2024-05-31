#!/bin/bash
#SBATCH --job-name="sft"
#SBATCH --time=24:00:00     # walltime
#SBATCH -o debug.out
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1

source ~/.bashrc
conda activate ppdpp

export PYTHONPATH=`pwd`
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=2 python -m ppdpp.run \
                                --mode train \
                                --do_epoch_eval \
                                --epochs 50000 \
                                --gamma 0.999 \
                                --lmbda 0.95 \
                                --eps 0.2 \
                                --learning_rate 1e-6 \
                                --data_name esc \
                                --system chatgpt \
                                --user chatgpt \
                                --critic chatgpt \
                                --planner chatgpt \
                                --rl_dir esc-roberta_5.0_400_6e-06_0.001_1e-08_1.0_10.0_0.1_td-chatgpt-chatgpt-chatgpt-0.0-5-5-10-0.25-0.2-False \
                                --max_turn 8 \
                                --max_seq_length 512 \
                                --model_name roberta \
                                --model_name_or_path roberta-large \
                                --model_path lmsys/vicuna-7b-v1.5 \
                                --start_step 1 \
                                --max_steps 2 \
                                --sample_times 100 \
                                --eval_start_index 0 \
                                --eval_sample_times 130 \
                                --eval_num 1 \
                                --save_num 1 \
                                --output_dir ppdpp \
                                --zero_shot \
                                --device_id 0 \
                                --num_gpus 1 \
                                --use_mcts_sys_resp \
                                --use_mcts_usr_resp \
                                --dropout 0.25 \
                                --mcts_applied_ratio 0.75 \
                                --num_mcts_sims 10 \
                                --max_realizations 1 \
                                --max_conv_turns 9 \
                                --max_hist_num_turns 8 \
                                --resp_max_new_tokens 64 \
                                --reward_max_new_tokens 16 \
                                --action_temperature 1.0 \
                                --resp_temperature 0.7 \
                                --reward_temperature 1.1 \
                                --action_num_return_sequences 15 \
                                --reward_num_return_sequences 10 \
                                --train_batch_size 4 \
                                --target_update_count 5 \
                                --critic_loss_w 1.0 \
                                --load_rl_epoch 2 \
                                --remark None