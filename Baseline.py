import torch
import numpy as np
import random
from ultralytics import YOLO
import os
# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)

# 加载预训练的YOLOv8模型
model = YOLO('yolov8s.pt')  # 使用YOLOv8s模型

# 开始训练
results = model.train(
    data='config/fruit_18.yaml',  
    epochs=80,        
    batch=64,          
    imgsz=640,         
    device='0',        # 使用GPU（0表示第一个GPU）
    project='runs/train',  # 保存训练结果的目录
    name='baseline2',   # 实验名称
    optimizer='SGD',   # 优化器类型
    lr0=0.01,          # 初始学习率
    lrf=0.1,           # 最终学习率（lr0 * lrf）
    augment=False,      # 禁用数据增强
    cache=True,        # 使用缓存加速训练
    verbose=True       # 打印详细输出
)

# 训练完成后，保存最佳模型
best_model_path = os.path.join(results.save_dir, 'weights/best.pt')
print(f"训练完成，最佳模型保存在: {best_model_path}")