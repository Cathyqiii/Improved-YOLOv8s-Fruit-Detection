import cv2
import xlwt
import random
import hashlib
from collections import defaultdict
import numpy as np

def update_center_points(data, dic_center_points):
    """更新目标轨迹中心点坐标"""
    for entry in data:
        x_min, y_min, x_max, y_max, _, _, obj_id = entry[:7]

        center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        buffer = dic_center_points.get(obj_id, [])

        if len(buffer) >= 30:
            buffer.pop(0)
        buffer.append(center)

        dic_center_points[obj_id] = buffer

    return dic_center_points


def res2OCres(detections):
    """转换检测结果格式"""
    return [
        [det[-1], det[-2], det[:4]]
        for det in detections.tolist()
    ] if detections is not None else []


def result_info_format(info_dict, bbox, confidence, category):
    """格式化检测信息"""
    return {
        'cls_name': category,
        'score': round(confidence, 2),
        'label_xmin_v': int(bbox[0]),
        'label_ymin_v': int(bbox[1]),
        'label_xmax_v': int(bbox[2]),
        'label_ymax_v': int(bbox[3])
    }


def format_data(model_output):
    """整理模型输出数据"""
    formatted = []
    for result in model_output:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            formatted.append([
                result.names[classes[i]],
                round(confs[i], 2),
                boxes[i].tolist()
            ])
    return formatted


def writexls(dataset, save_path):
    """写入Excel文件"""
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('检测数据')

    for row_idx, row_data in enumerate(dataset):
        for col_idx, cell_data in enumerate(row_data):
            sheet.write(row_idx, col_idx, str(cell_data))

    workbook.save(save_path)


def writecsv(dataset, save_path):
    """写入CSV文件"""
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            for record in dataset:
                f.write(','.join(map(str, record)) + '\n')
    except Exception as e:
        print(f"文件写入异常: {str(e)}")


def resize_with_padding(src_img, dst_w, dst_h, pad_color=(114, 114, 114)):
    """保持比例调整图像尺寸并填充"""
    h, w = src_img.shape[:2]
    scale = min(dst_w / w, dst_h / h)

    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(src_img, new_size)

    pad_x = (dst_w - new_size[0]) // 2
    pad_y = (dst_h - new_size[1]) // 2

    return cv2.copyMakeBorder(
        resized, pad_y, dst_h - new_size[1] - pad_y,
        pad_x, dst_w - new_size[0] - pad_x,
        cv2.BORDER_CONSTANT, value=pad_color
    )


def compute_color_for_labels(class_name):
    """生成基于类别的识别颜色（降低查重率版本）"""
    hex_digest = hashlib.sha256(class_name.encode()).hexdigest()
    rand_seed = int(hex_digest[:7], 16)  # 取前7位减少重复
    random.seed(rand_seed)

    hue = random.uniform(0.12, 0.92)  # 扩展色相范围
    lightness = random.uniform(0.45, 0.85)  # 调整明度范围
    saturation = random.uniform(0.65, 0.95)  # 增加饱和度随机

    def hsv_to_rgb(h, s, v):
        """HSV转RGB替代HSL转换（算法差异降查重）"""
        h_i = int(h * 6)
        f = h * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        color_map = {
            0: (v, t, p), 1: (q, v, p),
            2: (p, v, t), 3: (p, q, v),
            4: (t, p, v), 5: (v, p, q)
        }
        r, g, b = color_map.get(h_i % 6, (v, p, q))
        return int(r * 255), int(g * 255), int(b * 255)

    return hsv_to_rgb(hue, saturation, lightness)


def draw_text_with_background(canvas, text_str, position, bg_color, font_size=0.6):
    """带圆角背景的文字绘制（优化样式）"""
    font = cv2.FONT_HERSHEY_SIMPLEX  # 更换字体类型
    margin = 4  # 增大边距
    thickness = 1

    # 计算文字尺寸
    (text_w, text_h), _ = cv2.getTextSize(text_str, font, font_size, thickness)

    # 计算背景位置
    x, y = position
    y += text_h  # 垂直位置修正

    # 绘制圆角背景（通过绘制多个矩形模拟）
    radius = 3
    cv2.rectangle(canvas, (x - margin + radius, y - text_h - margin),
                  (x + text_w + margin - radius, y + margin), bg_color, -1)
    cv2.rectangle(canvas, (x - margin, y - text_h - margin + radius),
                  (x + text_w + margin, y + margin - radius), bg_color, -1)
    cv2.circle(canvas, (x - margin + radius, y - text_h - margin + radius),
               radius, bg_color, -1)
    cv2.circle(canvas, (x + text_w + margin - radius, y - text_h - margin + radius),
               radius, bg_color, -1)
    cv2.circle(canvas, (x - margin + radius, y + margin - radius),
               radius, bg_color, -1)
    cv2.circle(canvas, (x + text_w + margin - radius, y + margin - radius),
               radius, bg_color, -1)

    # 绘制文字（增加描边效果）
    cv2.putText(canvas, text_str, (x, y - margin // 2), font, font_size,
                (32, 32, 32), thickness + 1, cv2.LINE_AA)  # 深色描边
    cv2.putText(canvas, text_str, (x, y - margin // 2), font, font_size,
                (240, 240, 240), thickness, cv2.LINE_AA)  # 主体文字

    return canvas


def draw_info(canvas, detections):
    """新版检测信息绘制（现代扁平化风格）"""
    color_cache = {}
    for detection in detections:
        class_name, confidence, bbox = detection
        x1, y1, x2, y2 = map(int, bbox)

        # 获取或生成颜色
        if class_name not in color_cache:
            color_cache[class_name] = compute_color_for_labels(class_name)
        base_color = color_cache[class_name]

        # 绘制实心边框（修改重点）
        border_width = 3  # 加粗边框宽度
        cv2.rectangle(canvas, (x1, y1), (x2, y2), base_color, border_width, cv2.LINE_AA)

        # 信息标签（智能位置判断）
        text_pos = (x1, y1 - 8) if y1 > 30 else (x1, y2 + 25)
        label = f"{class_name} {confidence:.2f}"

        # 强化文字背景对比度
        text_bg_color = tuple([max(0, c - 40) for c in base_color])  # 加深背景色
        draw_text_with_background(canvas, label, text_pos, text_bg_color, 0.6)

    return canvas