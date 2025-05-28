@echo off

REM 设置环境变量
set CUDA_VISIBLE_DEVICES=0,1,2
set TORCH_DISTRIBUTED_DEBUG=INFO
set TORCH_COMPILE_DISABLE=1
set TORCH_DYNAMO_DISABLE=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set CUDA_LAUNCH_BLOCKING=1

REM 检查是否有多个GPU
nvidia-smi --list-gpus > gpu_count.tmp
for /f %%i in ('type gpu_count.tmp ^| find /c "GPU"') do set NUM_GPUS=%%i
del gpu_count.tmp

echo 检测到 %NUM_GPUS% 个GPU

if %NUM_GPUS% EQU 1 (
    echo 使用单GPU训练
    python train_multi_gpu.py --model_name "./models/LLM-Research/Llama3-8B-Chinese-Chat" --dataset_path "./data/chinese_qa_dataset.json" --output_dir "./medical_qa_model" --max_length 512 --num_epochs 3 --batch_size 2 --gradient_accumulation_steps 4 --warmup_steps 100 --use_lora
) else (
    echo 使用多GPU训练 (%NUM_GPUS% GPUs)
    torchrun --nproc_per_node=%NUM_GPUS% --master_port=29500 train_multi_gpu.py --model_name "./models/LLM-Research/Llama3-8B-Chinese-Chat" --dataset_path "./data/chinese_qa_dataset.json" --output_dir "./medical_qa_model" --max_length 512 --num_epochs 3 --batch_size 2 --gradient_accumulation_steps 4 --warmup_steps 100 --use_lora
)

echo 训练完成！
pause