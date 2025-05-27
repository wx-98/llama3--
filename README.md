# Llama3中文模型微调项目

基于医学问答数据集的Llama3-8B-Chinese-Chat模型微调项目。

## 项目结构

- `download_dataset.py`: 下载医学问答数据集
- `train_llama3.py`: 使用LoRA技术微调Llama3模型
- `inference.py`: 加载微调后的模型进行推理
- `install_requirements.py`: 安装项目依赖

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA支持的GPU

## 使用方法

1. 安装依赖：
```bash
python install_requirements.py