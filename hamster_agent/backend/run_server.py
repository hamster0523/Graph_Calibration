#!/usr/bin/env python3
"""
OpenManus Backend 独立启动器
解决模块导入路径问题
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加backend目录到Python路径
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# 现在导入应用
try:
    from main import app, main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("💡 请确保在项目根目录或backend目录中运行此脚本")
    sys.exit(1)