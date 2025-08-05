# utils.py
import csv
import os
import time
from datetime import datetime
from typing import List, Dict
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class DetectionResultProcessor:
    """处理并格式化检测结果"""

    @staticmethod
    def format_yolo_results(results) -> List[Dict]:
        """将YOLO检测结果转换为标准字典格式"""
        formatted = []
        for box in results.boxes:
            data = {
                "class": results.names[box.cls[0].item()],
                "confidence": box.conf[0].item(),
                "xmin": int(box.xyxy[0][0].item()),
                "ymin": int(box.xyxy[0][1].item()),
                "xmax": int(box.xyxy[0][2].item()),
                "ymax": int(box.xyxy[0][3].item())
            }
            formatted.append(data)
        return formatted


class ImageProcessor:
    """图像处理工具类"""

    @staticmethod
    def annotate_frame(frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """在图像上绘制检测框和标签"""
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 加载中文字体
        try:
            font = ImageFont.truetype("simhei.ttf", 20)
        except:
            font = ImageFont.load_default()

        for obj in results:
            # 绘制边界框
            draw.rectangle([obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]],
                           outline=(255, 0, 0), width=2)

            # 准备标签文本
            label = f"{obj['class']} {obj['confidence']:.2f}"

            # 计算文本位置
            text_width, text_height = draw.textsize(label, font)
            text_bg = [obj["xmin"], obj["ymin"] - text_height,
                       obj["xmin"] + text_width, obj["ymin"]]

            # 绘制文本背景和文字
            draw.rectangle(text_bg, fill=(255, 0, 0))
            draw.text((obj["xmin"], obj["ymin"] - text_height), label,
                      fill=(255, 255, 255), font=font)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class FileWriter:
    """文件存储管理类"""

    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_results(self, results: List[Dict], timestamp: str):
        """保存检测结果到CSV文件"""
        filename = os.path.join(self.output_dir, f"results_{timestamp}.csv")
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    def save_image(self, image: np.ndarray, timestamp: str):
        """保存带标注的图像"""
        filename = os.path.join(self.output_dir, f"detection_{timestamp}.jpg")
        cv2.imwrite(filename, image)

    def export(self, path: str, fmt: str, data: List) -> bool:
        """导出历史数据"""
        try:
            if fmt == "CSV":
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(data)
            elif fmt == "TXT":
                with open(path, "w", encoding="utf-8") as f:
                    for entry in data:
                        f.write(str(entry) + "\n")
            return True
        except Exception as e:
            print(f"导出失败: {e}")
            return False

    def cleanup(self):
        """清理临时文件"""
        # 可根据需要添加清理逻辑
        pass


class PerformanceMonitor:
    """性能监控类"""

    def __init__(self):
        self.times = []
        self.start_time = time.time()

    def update(self, process_time: float):
        """更新处理时间"""
        self.times.append(process_time)
        # 保持最近100条记录
        if len(self.times) > 100:
            self.times.pop(0)

    @property
    def fps(self) -> float:
        """计算实时帧率"""
        if len(self.times) == 0:
            return 0.0
        avg_time = sum(self.times) / len(self.times)
        return 1.0 / avg_time if avg_time != 0 else 0.0

    @property
    def total_uptime(self) -> float:
        """获取总运行时间"""
        return time.time() - self.start_time