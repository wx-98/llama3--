import os
import json
import torch
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
import matplotlib.pyplot as plt
from modelscope import snapshot_download

class MedicalQATrainer:
    def __init__(self, model_name="LLM-Research/Llama3-8B-Chinese-Chat", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        # 设置默认CUDA设备，确保所有操作都在同一设备上
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # 明确指定使用cuda:0
            torch.cuda.set_device(0)  # 设置当前活跃设备为cuda:0
        else:
            self.device = torch.device("cpu")
        print(f"使用设备: {self.device}")
        
    def download_model(self):
        """从ModelScope下载模型"""
        print("正在从ModelScope下载Llama3模型...")
        try:
            model_dir = snapshot_download(self.model_name, cache_dir="./models")
            print(f"模型下载完成: {model_dir}")
            return model_dir
        except Exception as e:
            print(f"ModelScope下载失败: {e}")
            print("尝试直接使用Hugging Face模型...")
            return self.model_name
    
    def load_model_and_tokenizer(self, model_path):
        """加载模型和分词器"""
        print("正在加载模型和分词器...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型，明确指定device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"":0},  # 明确指定所有模块都在cuda:0上
            trust_remote_code=True
        )
        
        print(f"模型参数量: {self.model.num_parameters():,}")
        
    def setup_lora(self):
        """设置LoRA配置"""
        print("配置LoRA...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def load_dataset(self, dataset_path):
        """加载数据集"""
        print(f"正在加载数据集: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
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
        
        print(f"训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(eval_data)}")
        
        return Dataset.from_list(train_data), Dataset.from_list(eval_data)
    
    def tokenize_function(self, examples):
        """数据预处理函数"""
        model_inputs = self.tokenizer(
            examples["text"],
            max_length=self.max_length,
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
        
        # 计算准确率（简化版本）
        mask = labels != -100
        accuracy = accuracy_score(
            labels[mask].flatten(), 
            predictions[mask].flatten()
        )
        
        return {"accuracy": accuracy}
    
    def train(self, train_dataset, eval_dataset, output_dir="./fine_tuned_llama3"):
        """训练模型"""
        print("开始训练...")
        
        # 数据预处理
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
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            dataloader_pin_memory=False,
            # 添加设备相关参数
            no_cuda=False,
            local_rank=-1,  # 单GPU训练
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # 开始训练
        print("\n=== 开始训练 ===")
        train_result = trainer.train()
        
        # 打印训练结果
        print("\n=== 训练完成 ===")
        print(f"训练损失: {train_result.training_loss:.4f}")
        print(f"训练步数: {train_result.global_step}")
        
        # 评估模型
        print("\n=== 模型评估 ===")
        eval_result = trainer.evaluate()
        print(f"验证损失: {eval_result['eval_loss']:.4f}")
        if 'eval_accuracy' in eval_result:
            print(f"验证准确率: {eval_result['eval_accuracy']:.4f}")
        
        # 保存模型
        print(f"\n=== 保存模型到 {output_dir} ===")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存训练日志
        with open(f"{output_dir}/training_log.json", 'w', encoding='utf-8') as f:
            json.dump({
                "training_loss": train_result.training_loss,
                "global_step": train_result.global_step,
                "eval_loss": eval_result['eval_loss'],
                "eval_accuracy": eval_result.get('eval_accuracy', 0)
            }, f, indent=2)
        
        print("训练完成！")
        return trainer

def main():
    # 设置环境变量，强制使用单个GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 初始化训练器
    trainer = MedicalQATrainer()
    
    # 下载模型
    model_path = trainer.download_model()
    
    # 加载模型和分词器
    trainer.load_model_and_tokenizer(model_path)
    
    # 设置LoRA
    trainer.setup_lora()
    
    # 加载数据集
    train_dataset, eval_dataset = trainer.load_dataset("data/medical_qa_dataset.json")
    
    # 开始训练
    trained_model = trainer.train(train_dataset, eval_dataset)
    
    print("\n所有训练步骤完成！")

if __name__ == "__main__":
    main()