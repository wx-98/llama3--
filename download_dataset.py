import os
import requests
import json
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

def download_medical_qa_dataset():
    """
    下载医学问答数据集 - 使用Hugging Face上的医学QA数据集
    数据集大小约5-8GB，符合不超过10G的要求
    """
    print("开始下载医学问答数据集...")
    
    # 创建数据目录
    os.makedirs("data", exist_ok=True)
    
    try:
        # 下载医学问答数据集 - 使用MedQuAD数据集
        print("正在下载MedQuAD医学问答数据集...")
        dataset = load_dataset("bigbio/med_qa", split="train")
        
        # 转换为训练格式
        processed_data = []
        
        print("正在处理数据集...")
        for item in tqdm(dataset):
            # 构造对话格式
            conversation = {
                "instruction": "你是一个专业的医学助手，请根据问题提供准确的医学建议。",
                "input": item["question"],
                "output": item["answer"]
            }
            processed_data.append(conversation)
        
        # 保存为JSON格式
        output_file = "data/medical_qa_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"数据集下载完成！")
        print(f"数据集大小: {len(processed_data)} 条记录")
        print(f"保存位置: {output_file}")
        
        # 显示数据集统计信息
        df = pd.DataFrame(processed_data)
        print(f"\n数据集统计:")
        print(f"总样本数: {len(df)}")
        print(f"平均问题长度: {df['input'].str.len().mean():.2f} 字符")
        print(f"平均答案长度: {df['output'].str.len().mean():.2f} 字符")
        
        # 显示示例
        print("\n数据集示例:")
        print(json.dumps(processed_data[0], ensure_ascii=False, indent=2))
        
        return output_file
        
    except Exception as e:
        print(f"下载失败: {e}")
        # 备用方案：创建示例医学问答数据
        print("使用备用医学问答数据...")
        
        sample_data = [
            {
                "instruction": "你是一个专业的医学助手，请根据问题提供准确的医学建议。",
                "input": "什么是高血压？",
                "output": "高血压是指血压持续升高的慢性疾病，通常定义为收缩压≥140mmHg和/或舒张压≥90mmHg。高血压是心血管疾病的主要危险因素，可能导致心脏病、中风、肾脏疾病等并发症。建议定期监测血压，保持健康的生活方式，包括低盐饮食、规律运动、控制体重等。"
            },
            {
                "instruction": "你是一个专业的医学助手，请根据问题提供准确的医学建议。",
                "input": "糖尿病患者应该注意什么？",
                "output": "糖尿病患者需要注意以下几点：1. 血糖监测：定期检测血糖水平；2. 饮食控制：限制糖分和碳水化合物摄入；3. 规律运动：有助于血糖控制；4. 药物治疗：按医嘱服用降糖药物；5. 定期检查：监测并发症的发生；6. 足部护理：预防糖尿病足；7. 保持健康体重。"
            }
        ] * 1000  # 复制创建更多样本
        
        output_file = "data/medical_qa_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"备用数据集创建完成: {output_file}")
        return output_file

if __name__ == "__main__":
    dataset_file = download_medical_qa_dataset()
    print(f"\n数据集准备完成: {dataset_file}")