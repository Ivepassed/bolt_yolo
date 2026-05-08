import torch
print(f"1. PyTorch版本: {torch.__version__}")
print(f"2. CUDA 是否可用: {torch.cuda.is_available()}")
print(f"3. 显卡型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}")

import numpy
print(f"4. 当前 NumPy 版本: {numpy.__version__}")