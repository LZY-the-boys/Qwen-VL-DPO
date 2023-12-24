source activate lla

# CUDA_VISIBLE_DEVICES=2 \
# accelerate launch --main_process_port $(shuf -i25000-30000 -n1) \

WANDB_MODE=disabled \
deepspeed --master_port $(shuf -i25000-30000 -n1)  --include localhost:1,2,3 \
run_dpo.py \
--model_name_or_path Qwen/Qwen-VL-Chat \
--output_dir /data/outs/qwen-dpo \
--bf16 True \
--fix_vit True \
--num_train_epochs 3 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 8 \
--evaluation_strategy steps \
--eval_steps 500 \
--save_strategy steps \
--save_steps 100 \
--save_total_limit 100 \
--learning_rate 1e-5 \
--weight_decay 0.05 \
--adam_beta2 0.98 \
--warmup_ratio 0.1 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--report_to wandb \
--run_name silkie-paperconfig \
--model_max_length 2048 \
--gradient_checkpointing True \
--use_lora True \
--tf32 True \
--logging_first_step True \
--remove_unused_columns False \
--output_dir $OUT_ROOT/qwen-vl-dpo