import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

class MedicalQAInference:
    def __init__(self, base_model_path="LLM-Research/Llama3-8B-Chinese-Chat", 
                 fine_tuned_path="./fine_tuned_llama3"):
        self.base_model_path = base_model_path
        self.fine_tuned_path = fine_tuned_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
    def load_model(self):
        """加载微调后的模型"""
        print("正在加载模型...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.fine_tuned_path,
            trust_remote_code=True
        )
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA权重
        self.model = PeftModel.from_pretrained(base_model, self.fine_tuned_path)
        self.model.eval()
        
        print("模型加载完成！")
    
    def generate_response(self, instruction, input_text, max_length=512, temperature=0.7):
        """生成回答"""
        # 构造输入格式
        prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n"
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回答部分
        if "### 回答:" in response:
            answer = response.split("### 回答:")[-1].strip()
        else:
            answer = response[len(prompt):].strip()
        
        return answer
    
    def interactive_chat(self):
        """交互式对话"""
        print("\n=== 医学问答助手 ===")
        print("输入 'quit' 退出程序")
        print("输入 'clear' 清屏")
        print("-" * 50)
        
        instruction = "你是一个专业的医学助手，请根据问题提供准确的医学建议。"
        
        while True:
            try:
                user_input = input("\n请输入您的医学问题: ").strip()
                
                if user_input.lower() == 'quit':
                    print("再见！")
                    break
                elif user_input.lower() == 'clear':
                    import os
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                elif not user_input:
                    continue
                
                print("\n正在生成回答...")
                response = self.generate_response(instruction, user_input)
                
                print(f"\n医学助手: {response}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"\n发生错误: {e}")
    
    def batch_inference(self, test_questions):
        """批量推理测试"""
        print("\n=== 批量推理测试 ===")
        instruction = "你是一个专业的医学助手，请根据问题提供准确的医学建议。"
        
        results = []
        for i, question in enumerate(test_questions, 1):
            print(f"\n问题 {i}: {question}")
            response = self.generate_response(instruction, question)
            print(f"回答: {response}")
            
            results.append({
                "question": question,
                "answer": response
            })
        
        # 保存结果
        with open("inference_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n批量推理完成，结果已保存到 inference_results.json")
        return results

def main():
    # 初始化推理器
    inferencer = MedicalQAInference()
    
    try:
        # 加载模型
        inferencer.load_model()
        
        # 测试问题
        test_questions = [
            "什么是高血压？如何预防？",
            "糖尿病患者的饮食注意事项有哪些？",
            "感冒和流感有什么区别？",
            "如何保持心脏健康？",
            "失眠的常见原因和解决方法是什么？"
        ]
        
        # 批量测试
        inferencer.batch_inference(test_questions)
        
        # 交互式对话
        inferencer.interactive_chat()
        
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        print("请确保已完成模型训练并保存在正确路径")

if __name__ == "__main__":
    main()