import os
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score
import argparse
import logging
from datetime import datetime

# 设置环境变量以避免DTensor相关问题
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaMultiGPUTrainer:
    def __init__(self, args):
        self.args = args
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        
        # 设置设备
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device('cpu')
            
        logger.info(f"进程 {self.rank}/{self.world_size} 使用设备: {self.device}")
        
    def setup_distributed(self):
        """初始化分布式训练"""
        if self.world_size > 1:
            try:
                # 使用NCCL后端进行GPU通信
                if torch.cuda.is_available():
                    backend = 'nccl'
                else:
                    backend = 'gloo'
                    
                dist.init_process_group(
                    backend=backend,
                    init_method='env://',
                    world_size=self.world_size,
                    rank=self.rank
                )
                logger.info(f"分布式训练初始化成功，使用后端: {backend}")
            except Exception as e:
                logger.error(f"分布式初始化失败: {e}")
                raise
        else:
            logger.info("单GPU训练模式")
            
    def cleanup_distributed(self):
        """清理分布式进程组"""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("分布式进程组已清理")
            
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        logger.info("正在加载模型和分词器...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=None  # 手动管理设备映射
        )
        
        # 移动模型到指定设备
        self.model = self.model.to(self.device)
        
        logger.info(f"模型参数量: {self.model.num_parameters():,}")
        
    def setup_lora(self):
        """设置LoRA配置"""
        logger.info("配置LoRA...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        if self.rank == 0:
            self.model.print_trainable_parameters()
            
    def wrap_model_for_distributed(self):
        """为分布式训练包装模型"""
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            logger.info("模型已包装为DDP")
            
    def load_dataset(self):
        """加载数据集"""
        logger.info(f"正在加载数据集: {self.args.dataset_path}")
        
        with open(self.args.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 格式化数据
        formatted_data = []
        for item in data:
            text = f"### 指令:\n{item['instruction']}\n\n### 输入:\n{item['input']}\n\n### 回答:\n{item['output']}"
            formatted_data.append({"text": text})
            
        # 分割训练和验证集
        split_idx = int(len(formatted_data) * 0.9)
        train_data = formatted_data[:split_idx]
        eval_data = formatted_data[split_idx:]
        
        if self.rank == 0:
            logger.info(f"训练集大小: {len(train_data)}")
            logger.info(f"验证集大小: {len(eval_data)}")
            
        return Dataset.from_list(train_data), Dataset.from_list(eval_data)
        
    def tokenize_function(self, examples):
        """数据预处理函数"""
        model_inputs = self.tokenizer(
            examples["text"],
            max_length=self.args.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
        
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        # 计算准确率
        mask = labels != -100
        accuracy = accuracy_score(
            labels[mask].flatten(), 
            predictions[mask].flatten()
        )
        
        return {"accuracy": accuracy}
        
    def train(self):
        """训练模型"""
        # 加载数据集
        train_dataset, eval_dataset = self.load_dataset()
        
        # 预处理数据
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_dataset = eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_steps=self.args.warmup_steps,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            local_rank=self.local_rank,
            ddp_find_unused_parameters=False,
            dataloader_num_workers=0,  # 避免多进程数据加载问题
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # 开始训练
        if self.rank == 0:
            logger.info("开始训练...")
            
        trainer.train()
        
        # 保存模型（只在主进程保存）
        if self.rank == 0:
            logger.info("保存模型...")
            trainer.save_model(self.args.output_dir)
            self.tokenizer.save_pretrained(self.args.output_dir)
            logger.info("模型保存完成")
            
def parse_args():
    parser = argparse.ArgumentParser(description="Llama3 多GPU微调训练")
    # 修改默认模型路径为本地路径
    parser.add_argument("--model_name", type=str, default="./models/LLM-Research/Llama3-8B-Chinese-Chat", help="模型名称")
    parser.add_argument("--dataset_path", type=str, default="./data/chinese_qa_dataset.json", help="数据集路径")
    parser.add_argument("--output_dir", type=str, default="./medical_qa_model", help="输出目录")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    
    return parser.parse_args()
    
def main():
    args = parse_args()
    
    # 创建训练器
    trainer = LlamaMultiGPUTrainer(args)
    
    try:
        # 初始化分布式训练
        trainer.setup_distributed()
        
        # 加载模型和分词器
        trainer.load_model_and_tokenizer()
        
        # 设置LoRA（如果启用）
        if args.use_lora:
            trainer.setup_lora()
            
        # 为分布式训练包装模型
        trainer.wrap_model_for_distributed()
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise
    finally:
        # 清理分布式进程组
        trainer.cleanup_distributed()
        
if __name__ == "__main__":
    main()