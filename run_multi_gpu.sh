#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2  # 根据您的3个GPU调整
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_COMPILE_DISABLE=1
export TORCH_DYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1  # 帮助调试CUDA错误

# 检查GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $NUM_GPUS 个GPU"

# 如果只有一个GPU，使用单GPU训练
if [ $NUM_GPUS -eq 1 ]; then
    echo "使用单GPU训练"
    python train_multi_gpu.py \
        --model_name "./models/LLM-Research/Llama3-8B-Chinese-Chat" \
        --dataset_path "./data/chinese_qa_dataset.json" \
        --output_dir "./medical_qa_model" \
        --max_length 512 \
        --num_epochs 3 \
        --batch_size 2 \
        --gradient_accumulation_steps 4 \
        --warmup_steps 100 \
        --use_lora
else
    echo "使用多GPU训练 ($NUM_GPUS GPUs)"
    # 使用torchrun代替torch.distributed.launch（推荐的新方法）
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_multi_gpu.py \
        --model_name "./models/LLM-Research/Llama3-8B-Chinese-Chat" \
        --dataset_path "./data/chinese_qa_dataset.json" \
        --output_dir "./medical_qa_model" \
        --max_length 512 \
        --num_epochs 3 \
        --batch_size 2 \
        --gradient_accumulation_steps 4 \
        --warmup_steps 100 \
        --use_lora
fi

echo "训练完成！"