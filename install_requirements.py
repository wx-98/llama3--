import subprocess
import sys

def install_requirements():
    """安装所需依赖"""
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "peft>=0.4.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.39.0",
        "modelscope>=1.8.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0"
    ]
    
    print("正在安装依赖包...")
    for package in requirements:
        print(f"安装 {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"✗ {package} 安装失败: {e}")
    
    print("\n依赖安装完成！")

if __name__ == "__main__":
    install_requirements()