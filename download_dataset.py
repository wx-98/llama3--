import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def download_chinese_qa_dataset():
    """
    下载中文问答数据集 - 使用Hugging Face上的常见中文问答数据集
    数据集大小控制在10G以内
    """
    print("开始下载中文问答数据集...")
    
    # 创建数据目录
    os.makedirs("data", exist_ok=True)
    
    try:
        # 尝试下载方案1：使用BELLE数据集（中文指令微调数据集）
        print("正在下载BELLE中文指令数据集...")
        dataset = load_dataset("BelleGroup/train_1M_CN", split="train[:50000]")  # 只取前50000条数据，控制大小
        
        # 转换为训练格式
        processed_data = []
        
        print("正在处理数据集...")
        for item in tqdm(dataset):
            # 构造对话格式
            conversation = {
                "instruction": "你是一个有帮助的AI助手，请根据问题提供准确的回答。",
                "input": item["instruction"],
                "output": item["output"]
            }
            processed_data.append(conversation)
        
        # 保存为JSON格式
        output_file = "data/chinese_qa_dataset.json"
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
        print(f"BELLE数据集下载失败: {e}")
        print("尝试备用数据集...")
        
        try:
            # 尝试下载方案2：使用中文维基百科问答数据集
            print("正在下载中文维基百科问答数据集...")
            dataset = load_dataset("wangrui6/Zhihu-KOL", split="train[:50000]")  # 只取部分数据
            
            # 转换为训练格式
            processed_data = []
            
            print("正在处理数据集...")
            for item in tqdm(dataset):
                if 'question' in item and 'answer' in item and item['question'] and item['answer']:
                    conversation = {
                        "instruction": "你是一个有帮助的AI助手，请根据问题提供准确的回答。",
                        "input": item["question"],
                        "output": item["answer"]
                    }
                    processed_data.append(conversation)
            
            # 保存为JSON格式
            output_file = "data/chinese_qa_dataset.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            print(f"备用数据集下载完成！")
            print(f"数据集大小: {len(processed_data)} 条记录")
            print(f"保存位置: {output_file}")
            
            return output_file
            
        except Exception as e2:
            print(f"备用数据集下载失败: {e2}")
            # 最终备用方案：创建示例中文问答数据
            print("使用本地生成的示例中文问答数据...")
            
            sample_data = [
                {
                    "instruction": "你是一个有帮助的AI助手，请根据问题提供准确的回答。",
                    "input": "什么是人工智能？",
                    "output": "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于研发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。人工智能可以感知环境、获取知识并使用知识达成特定目标。它包括机器学习、深度学习、自然语言处理、计算机视觉等多个领域，已广泛应用于医疗、金融、教育、交通等各个行业。"
                },
                {
                    "instruction": "你是一个有帮助的AI助手，请根据问题提供准确的回答。",
                    "input": "如何提高学习效率？",
                    "output": "提高学习效率的方法包括：1. 制定明确的学习计划和目标；2. 采用番茄工作法等时间管理技巧；3. 创造专注的学习环境，减少干扰；4. 使用主动学习方法，如费曼技巧（教授他人）；5. 保持规律作息和适当运动；6. 合理安排休息时间；7. 使用思维导图等工具整理知识；8. 定期复习和自测；9. 寻找适合自己的学习方式；10. 保持积极的学习态度和动力。"
                },
                {
                    "instruction": "你是一个有帮助的AI助手，请根据问题提供准确的回答。",
                    "input": "中国的四大发明是什么？",
                    "output": "中国古代的四大发明是指造纸术、印刷术、火药和指南针。这些发明对世界文明的发展产生了深远的影响。造纸术改变了人类记录信息的方式；活字印刷术大大提高了知识传播的效率；火药最初用于烟花爆竹，后来改变了战争形态；指南针则促进了航海技术的发展，为大航海时代奠定了基础。"
                },
                {
                    "instruction": "你是一个有帮助的AI助手，请根据问题提供准确的回答。",
                    "input": "如何做一道红烧肉？",
                    "output": "红烧肉的做法：\n1. 准备食材：五花肉500克，姜片、葱段、八角、桂皮、干辣椒适量。\n2. 将五花肉切成2厘米见方的块。\n3. 冷水下锅，焯水去除血水和杂质，捞出备用。\n4. 锅中放少量油，加入白糖小火熬制成焦糖色。\n5. 放入肉块翻炒均匀上色。\n6. 加入料酒、生抽、老抽、姜片、葱段和各种香料。\n7. 加入没过肉的热水，大火烧开后转小火。\n8. 慢炖40-60分钟，直到肉烂且汤汁浓稠。\n9. 最后大火收汁，使肉块色泽红亮。\n10. 出锅前撒上葱花点缀即可。"
                },
                {
                    "instruction": "你是一个有帮助的AI助手，请根据问题提供准确的回答。",
                    "input": "Python如何读取JSON文件？",
                    "output": "在Python中读取JSON文件的步骤：\n1. 首先导入json模块：`import json`\n2. 使用with语句打开文件并读取内容：\n```python\nwith open('文件路径.json', 'r', encoding='utf-8') as file:\n    data = json.load(file)\n```\n3. 现在data变量中包含了JSON文件的内容，可以像操作Python字典或列表一样使用它。\n\n注意事项：\n- encoding='utf-8'参数确保正确处理中文等Unicode字符\n- 如果JSON格式不正确，json.load()会抛出JSONDecodeError异常\n- 对于大型JSON文件，可以考虑使用ijson库进行流式解析"
                }
            ] * 1000  # 复制创建更多样本
            
            output_file = "data/chinese_qa_dataset.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
            print(f"示例数据集创建完成: {output_file}")
            return output_file

if __name__ == "__main__":
    dataset_file = download_chinese_qa_dataset()
    print(f"\n数据集准备完成: {dataset_file}")