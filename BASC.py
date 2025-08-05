import os
import torch
from ultralytics import YOLO

# -------------------- 主训练流程（优化版） --------------------
if __name__ == '__main__':
    # 配置文件路径
    model_yaml = 'BASC.yaml'
    data_yaml = os.path.join('config', 'fruit_18.yaml')
    pretrained = 'yolov8s.pt'

    # 初始化模型
    model = YOLO(model_yaml, task='detect').load(pretrained)

    # 显存优化配置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        
    # 优化后的训练参数
    train_args= {
        "task": "detect",
        "mode": "train",
        "model": "allin.yaml",
        "data": "config/fruit_18.yaml",
        "epochs": 200,
        "time": None,
        "patience": 30,
        "batch": 64,
        "imgsz": 640,
        "save": True,
        "save_period": -1,
        "cache": True,
        "device": 0,
        "workers": 8,
        "project": "runs/train",
        "name": "yolov8_allin3",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
        "rect": False,
        "cos_lr": False,
        "close_mosaic": 15,
        "resume": False,
        "amp": True,
        "fraction": 1.0,
        "profile": False,
        "freeze": [0, 1, 2],
        "multi_scale": False,
        "overlap_mask": True,
        "mask_ratio": 4,
        "dropout": 0.0,
        "val": True,
        "split": "val",
        "save_json": False,
        "save_hybrid": False,
        "conf": None,
        "iou": 0.7,
        "max_det": 300,
        "half": False,
        "dnn": False,
        "plots": True,
        "source": None,
        "vid_stride": 1,
        "stream_buffer": False,
        "visualize": False,
        "augment": True,
        "agnostic_nms": False,
        "classes": None,
        "retina_masks": False,
        "embed": None,
        "show": False,
        "save_frames": False,
        "save_txt": False,
        "save_conf": False,
        "save_crop": False,
        "show_labels": True,
        "show_conf": True,
        "show_boxes": True,
        "line_width": None,
        "format": "torchscript",
        "keras": False,
        "optimize": False,
        "int8": False,
        "dynamic": False,
        "simplify": False,
        "opset": None,
        "workspace": 4,
        "nms": False,
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.001,
        "warmup_epochs": 5.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 6.0,
        "cls": 1.0,
        "dfl": 1.5,
        "pose": 12.0,
        "kobj": 1.0,
        "label_smoothing": 0.3,
        "nbs": 64,
        "hsv_h": 0.01,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 15.0,
        "translate": 0.1,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.001,
        "flipud": 0.1,
        "fliplr": 0.3,
        "bgr": 0.0,
        "mosaic": 0.1,
        "mixup": 0.1,
        "copy_paste": 0.1,
        "auto_augment": "randaugment",
        "erasing": 0.1,
        "crop_fraction": 1.0,
        "cfg": None,
        "tracker": "botsort.yaml",
        "save_dir": "runs/train/yolov8_BiFPN_CA_11"
    }
    # 启动训练
    results = model.train(**train_args)
    print(f"训练完成，最佳模型保存在: {results.save_dir}")

    # 保存最佳模型
    best_model_path = os.path.join(results.save_dir, 'weights/best.pt')
    if os.path.exists(best_model_path):
        fruit_vision_path = os.path.join(results.save_dir, 'weights/Fruit_allin.pt')
        os.rename(best_model_path, fruit_vision_path)
        print(f"最佳模型已保存为: {fruit_vision_path}")