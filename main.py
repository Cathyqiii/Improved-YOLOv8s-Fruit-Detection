# -*- coding: utf-8 -*-
import traceback
from asyncio import Queue
from datetime import datetime
import time

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QMessageBox
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor

from Fruit_Detection import MyMainWindow

from PyQt5 import QtGui
from PyQt5.QtWidgets import QFileDialog

import cv2
import os
import numpy as np

from ultralytics import YOLO

from tool.parser import get_config
from tool.tools import draw_info, result_info_format, format_data, writexls, writecsv, resize_with_padding

import winsound

# ================== 多线程组件 ==================
class DetectionWorker(QThread):
    finished = pyqtSignal(object, object, float, str)  # 添加时间戳参数

    def __init__(self, model, img, img_path):
        super().__init__()
        self.model = model
        self.img = img
        self.image_file_path = img_path

    def run(self):
        try:
            start_time = time.time()
            detection_data = self.model.predict(self.img, imgsz=640, verbose=False)
            processed_detection_data = format_data(detection_data)
            self.finished.emit(processed_detection_data, self.img.copy(), start_time, self.img_path)
        except Exception as e:
            print(f"Detection error: {str(e)}")

class CaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                break
        cap.release()

    def stop(self):
        self.running = False

class DetectionMainWindow(QMainWindow, MyMainWindow):
    def __init__(self, cfg=None):
        super().__init__()

        # 初始化UI配置
        self.setupUi(self)
        self.setup_ui_from_config()

        # 硬件加速配置
        self.device = 'cuda' if cfg.MODEL.DEVICE == 'gpu' else 'cpu'
        self.batch_size = 4  # 批量推理大小

        # 多线程组件
        self.detection_threads = []
        self.capture_thread = None
        self.frame_queue = Queue(maxsize=10)  # 帧缓冲队列

        self.init_style_enhancement()

        self.frame_number = 0
        self.comboBox_index = None
        self.detection_data = []
        self.result_img_name = None
        self.setupUi(self)

        # 根据config配置文件更新界面配置
        self.setup_ui_from_config()
        self.start_type = None
        self.img = None
        self.img_path = None
        self.video = None
        self.video_path = None
        # 绘制了识别信息的frame
        self.img_show = None
        self.sign = True

        self.result_info = None

        self.chinese_name = chinese_name

        # 获取当前工程文件位置
        self.ProjectPath = os.getcwd()
        self.comboBox_text = '所有目标'
        run_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # 保存所有的输出文件
        self.output_dir = os.path.join(self.ProjectPath, 'output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        result_time_path = os.path.join(self.output_dir, run_time)
        os.mkdir(result_time_path)

        # 保存txt内容
        self.result_txt = os.path.join(result_time_path, 'result.txt')
        with open(self.result_txt, 'w') as result_file:
            result_file.write(
                str(['序号', '文件名称', '上传时间', '检测结果', '目标数量', '计算耗时', '存储位置'])[1:-1])
            result_file.write('\n')

        # 保存绘制好的图片结果
        self.result_img_path = os.path.join(result_time_path, 'img_result')
        os.mkdir(self.result_img_path)

        # 默认选择为所有目标
        self.comboBox_value = '所有目标'

        self.number = 1
        self.RowLength = 0
        self.consum_time = 0
        self.input_time = 0

        # 打开图片
        self.pushButton_img.clicked.connect(self.load_image)
        # 打开文件夹
        self.pushButton_dir.clicked.connect(self.load_image_dir)
        # 打开视频
        self.pushButton_video.clicked.connect(self.load_video)
        # 启用实时摄像头
        self.pushButton_camera.clicked.connect(self.toggle_camera)
        # 绑定启动检测按钮
        self.pushButton_start.clicked.connect(self.start)
        # 导出数据
        self.pushButton_export.clicked.connect(self.write_files)

        self.comboBox.activated.connect(self.handle_selection_change)
        self.comboBox.mousePressEvent = self.on_comboBox_click

        # 初始化UI配置
        self.setupUi(self)
        self.setup_ui_from_config()

        # 硬件加速配置
        self.device = 'cuda' if cfg.MODEL.DEVICE == 'gpu' else 'cpu'
        self.batch_size = 4  # 批量推理大小

        # 多线程组件
        self.detection_threads = []
        self.capture_thread = None
        self.frame_queue = Queue(maxsize=10)  # 帧缓冲队列

        self.init_style_enhancement()

        self.frame_number = 0
        self.comboBox_index = None
        self.detection_data = []
        self.result_img_name = None
        self.setupUi(self)

        # 根据config配置文件更新界面配置
        self.setup_ui_from_config()
        self.start_type = None
        self.img = None
        self.img_path = None
        self.video = None
        self.video_path = None
        # 绘制了识别信息的frame
        self.img_show = None
        self.sign = True

        self.result_info = None

        self.chinese_name = chinese_name

        # 获取当前工程文件位置
        self.ProjectPath = os.getcwd()
        self.comboBox_text = '所有目标'
        run_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # 保存所有的输出文件
        self.output_dir = os.path.join(self.ProjectPath, 'output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        result_time_path = os.path.join(self.output_dir, run_time)
        os.makedirs(result_time_path, exist_ok=True)  # 如果目录存在不会报错

        # 保存txt内容
        self.result_txt = os.path.join(result_time_path, 'result.txt')
        with open(self.result_txt, 'w') as result_file:
            result_file.write(
                str(['序号', '文件名称', '上传时间', '检测结果', '目标数量', '计算耗时', '存储位置'])[1:-1])
            result_file.write('\n')

        # 保存绘制好的图片结果
        self.result_img_path = os.path.join(result_time_path, 'img_result')
        # 使用makedirs并设置exist_ok=True
        os.makedirs(self.result_img_path, exist_ok=True)  # 目录存在时不会报错
        # 默认选择为所有目标
        self.comboBox_value = '所有目标'

        self.number = 1
        self.RowLength = 0
        self.consum_time = 0
        self.input_time = 0

        # 打开图片
        self.pushButton_img.clicked.connect(self.load_image)
        # 打开文件夹
        self.pushButton_dir.clicked.connect(self.load_image_dir)
        # 打开视频
        self.pushButton_video.clicked.connect(self.load_video)
        # 启用实时摄像头
        self.pushButton_camera.clicked.connect(self.toggle_camera)
        # 绑定启动检测按钮
        self.pushButton_start.clicked.connect(self.start)
        # 导出数据
        self.pushButton_export.clicked.connect(self.write_files)

        self.comboBox.activated.connect(self.handle_selection_change)
        self.comboBox.mousePressEvent = self.on_comboBox_click

        # 表格点击事件绑定
        self.tableWidget_info.cellClicked.connect(self.handle_cell_selection)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.image_files = []
        self.current_index = 0


        # 表格点击事件绑定
        self.tableWidget_info.cellClicked.connect(self.handle_cell_selection)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.image_files = []
        self.current_index = 0

    def setup_ui_from_config(self):
        """根据config.yaml配置更新界面组件"""
        # 基础界面配置
        self.setWindowTitle(title)
        self.label_title.setText(label_title)
        self.setStyleSheet(f"#centralwidget {{background-image: url('{background_img}')}}")
        self.label_img.setPixmap(QtGui.QPixmap(zhutu))

        # 信息标签配置
        self.label_info.setText(label_info_txt)
        self.label_info.setStyleSheet(f"color: rgb({label_info_color});")
        self.label_info.setHidden(True)

        # 按钮样式配置
        button_styles = [
            (self.pushButton_start, start_button_bg, start_button_font),
            (self.pushButton_export, export_button_bg, export_button_font)
        ]
        for btn, bg, fg in button_styles:
            btn.setStyleSheet(f"background-color: rgb({bg}); border-radius: 15px; color: rgb({fg});")

        # 区域背景配置
        area_backgrounds = [
            (self.label_control, label_control_color),
            (self.label_img, label_img_color)
        ]
        for widget, color in area_backgrounds:
            widget.setStyleSheet(f"background-color: rgba({color}); border-radius: 15px;")

        # 表格样式配置
        table_styles = {
            "header": """
                QHeaderView::section {{
                    background-color: rgb(0, 128, 0);
                    color: rgb(255, 255, 255);
                    border: none;
                    padding: 4px;
                }}""",
            "body": f"""
                QTableWidget {{
                    background-color: rgb({background_color});
                }}
                QTableView::item:hover{{
                    background-color:rgb({item_hover_background_color});
                }}"""
        }
        self.tableWidget_info.horizontalHeader().setStyleSheet(table_styles["header"])
        self.tableWidget_info.setStyleSheet(table_styles["body"])

        # 表格列宽配置
        for col, width in enumerate(column_widths):
            self.tableWidget_info.setColumnWidth(col, width)

    def handle_cell_selection(self, row, column):
        """处理表格单元格选择事件"""
        self.update_comboBox_default()

        # 检查有效数据行
        if not self.tableWidget_info.item(row, 1):
            return

        # 获取基础数据
        self.img_path = self.tableWidget_info.item(row, 1).text()
        self.detection_data = eval(self.tableWidget_info.item(row, 3).text())
        self.result_img_name = self.tableWidget_info.item(row, 6).text()
        self.img_show = cv2.imdecode(np.fromfile(self.result_img_name, dtype=np.uint8), -1)

        # 初始化默认值
        box = [0, 0, 0, 0]
        score = 0
        cls_name = '无目标'
        accuracy = 0.0

        # 处理检测结果
        if self.detection_data:
            # 计算平均置信度
            accuracy = sum(x[1] for x in self.detection_data) / len(self.detection_data)
            # 获取首个结果
            first_result = self.detection_data[0]
            box, score, cls_name = first_result[2], first_result[1], first_result[0]

        # 构建结果信息
        result_info = result_info_format({}, box, score, cls_name)
        result_info['accuracy'] = accuracy

        # 更新界面显示
        self.get_comboBox_value(self.detection_data)
        self.show_all(self.img_show, result_info)

    def on_comboBox_click(self, event):
        if event.button() == Qt.LeftButton:
            # 控制选择下拉列表时，是否暂停识别
            # self.sign = False
            # 清空列表
            self.comboBox.clear()
            if type(self.comboBox_value) == str:
                self.comboBox_value = [self.comboBox_value]
            self.comboBox.addItems(self.comboBox_value)
        QComboBox.mousePressEvent(self.comboBox, event)

    def handle_selection_change(self):
        """处理下拉列表选择变化事件 - 修正准确率显示问题"""
        self.sign = True
        comboBox_text = self.comboBox.currentText()
        self.comboBox_index = self.comboBox.currentIndex()
        result_info = {}

        if len(self.detection_data) == 0:
            print('图片中无目标！')
            QMessageBox.information(self, "信息", "图片中无目标", QMessageBox.Yes)
            return

        # 所有目标模式
        if comboBox_text == '所有目标':
            box = self.detection_data[0][2]
            score = self.detection_data[0][1]
            cls_name = self.detection_data[0][0]
            lst_info = self.detection_data
            # 计算所有目标的平均置信度
            accuracy = sum(x[1] for x in self.detection_data) / len(self.detection_data) if self.detection_data else 0.0
        else:  # 单个目标模式
            select_result = self.detection_data[self.comboBox_index - 1]
            box = select_result[2]
            cls_name = select_result[0]
            score = select_result[1]
            lst_info = [[cls_name, score, box]]
            # 单个目标的置信度直接作为准确率
            accuracy = score

        # 构建结果信息（确保包含accuracy字段）
        result_info = result_info_format(result_info, box, score, cls_name)
        result_info['accuracy'] = accuracy  # 关键修复：确保准确率字段被正确设置

        # 更新显示
        self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.img_show = draw_info(self.img, lst_info)
        self.show_all(self.img_show, result_info)

    def display_frame(self, img):
        self.update()
        if img is not None:
            # 填充颜色
            shrink = resize_with_padding(img, self.label_img.width(), self.label_img.height(),
                                         [padvalue[2], padvalue[1], padvalue[0]])
            shrink = cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB)
            QtImg = QtGui.QImage(shrink[:], shrink.shape[1], shrink.shape[0], shrink.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)
            self.label_img.setPixmap(QtGui.QPixmap.fromImage(QtImg))

    def load_image(self):
        try:
            # 更新下拉列表的状态
            self.update_comboBox_default()
            # 选择文件  ;;All Files (*)
            self.img_path, filetype = QFileDialog.getOpenFileName(None, "选择文件", self.ProjectPath,
                                                                  "JPEG Image (*.jpg);;PNG Image (*.png);;JFIF Image (*.jfif)")
            if self.img_path == "":  # 未选择文件
                self.start_type = None
                return

            self.img_name = os.path.basename(self.img_path)
            # 显示相对应的文字
            self.label_img_path.setText(" " + self.img_path)
            self.label_dir_path.setText(" 选择图片文件夹")
            self.label_video_path.setText(" 选择视频文件")
            self.label_camera_path.setText(" 启用实时摄像头")

            self.start_type = 'img'
            # 读取中文路径下图片
            self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            # 显示原图
            self.display_frame(self.img)
        except Exception as e:
            traceback.print_exc()

    def load_image_dir(self):
        try:
            # 更新下拉列表的状态
            self.update_comboBox_default()
            self.img_path_dir = QFileDialog.getExistingDirectory(None, "选择文件夹")
            if not self.img_path_dir:
                self.start_type = None
                return

            self.start_type = 'dir'
            # 显示相对应的文字
            self.label_dir_path.setText(" " + self.img_path_dir)
            self.label_img_path.setText(" 选择图片文件")
            self.label_video_path.setText(" 选择视频文件")
            self.label_camera_path.setText(" 启用实时摄像头")

            self.image_files = [file for file in os.listdir(self.img_path_dir) if file.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))]

            if not self.image_files:
                QMessageBox.information(self, "信息", "文件夹中没有符合条件的图片", QMessageBox.Yes)
                return

            self.current_index = 0
            self.img_path = os.path.join(self.img_path_dir, self.image_files[self.current_index])
            self.img_name = self.image_files[self.current_index]

            self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.display_frame(self.img)
        except Exception as e:
            traceback.print_exc()

    def load_video(self):
        try:
            # 更新下拉列表的状态
            self.update_comboBox_default()
            # 选择文件
            self.video_path, filetype = QFileDialog.getOpenFileName(None, "选择文件", self.ProjectPath,
                                                                    "mp4 Video (*.mp4);;avi Video (*.avi)")
            if not self.video_path:
                self.start_type = None
                return

            self.start_type = 'video'
            # 显示相对应的文字
            self.label_video_path.setText(" " + self.video_path)
            self.label_img_path.setText(" 选择图片文件")
            self.label_dir_path.setText(" 选择图片文件夹")
            self.label_camera_path.setText(" 启用实时摄像头")

            self.video_name = os.path.basename(self.video_path)
            self.video = cv2.VideoCapture(self.video_path)
            # 读取第一帧
            ret, self.img = self.video.read()
            self.display_frame(self.img)
        except Exception as e:
            traceback.print_exc()

    def toggle_camera(self):
        try:
            # 更新下拉列表的状态
            self.update_comboBox_default()
            if self.label_camera_path.text() == ' 启用实时摄像头' or self.label_camera_path.text() == ' 摄像头已关闭':
                self.start_type = 'camera'
                # 显示相对应的文字
                self.label_img_path.setText(" 选择图片文件")
                self.label_dir_path.setText(" 选择图片文件夹")
                self.label_video_path.setText(" 选择视频文件")
                self.label_camera_path.setText(" 摄像头已打开")

                self.video_name = camera_num
                self.video = cv2.VideoCapture(self.video_name)
                ret, self.img = self.video.read()
                self.display_frame(self.img)
            elif self.label_camera_path.text() == ' 摄像头已打开':

                self.pushButton_start.setText("启动检测 ▶")
                self.label_camera_path.setText(" 摄像头已关闭")

        except Exception as e:
            traceback.print_exc()

    def init_style_enhancement(self):
        """增强界面美化效果 - 修正后的版本"""
        style_data = {
            'btn_bg': '5cb85c',
            'btn_font': 'ffffff',
            'border_color': '4cae4c',
            'btn_hover': '449d44',
            'border_hover': '398439',
            'header_bg': '337ab7',
            'header_font': 'ffffff'
        }

        # 修正样式表字符串格式
        dynamic_style = """
          QPushButton {{
              background-color: #{btn_bg};
              color: #{btn_font};
              border-radius: 8px;
              padding: 8px 16px;
              border: 1px solid #{border_color};
              min-width: 80px;
          }}
          QPushButton:hover {{
              background-color: #{btn_hover};
              border: 2px solid #{border_hover};
          }}
          QTableWidget {{
              alternate-background-color: #f5f5f5;
              gridline-color: #e0e0e0;
          }}
          QHeaderView::section {{
              background-color: #{header_bg};
              color: #{header_font};
              padding: 4px;
          }}
          """.format(**style_data)

        self.setStyleSheet(dynamic_style)

    # ================== 核心方法重构 ==================
    def start_detection_pipeline(self):
        """启动多线程检测流水线"""
        if self.start_type in ['video', 'camera']:
            # 启动视频采集线程
            source = self.video_path if self.start_type == 'video' else camera_num
            self.capture_thread = CaptureThread(source)
            self.capture_thread.frame_ready.connect(self.process_frame)
            self.capture_thread.start()

            # 启动检测线程池
            for _ in range(4):  # 创建4个检测线程
                thread = DetectionWorker(self.model, None, None)
                thread.finished.connect(self.handle_detection_result)
                thread.start()
                self.detection_threads.append(thread)

    def process_frame(self, frame):
        """帧处理流水线"""
        if self.frame_queue.qsize() < 5:  # 控制队列深度
            self.frame_queue.put(frame)
            self.update_throughput_counter()

    def handle_detection_result(self, detection_data, img, start_time, img_path):
        """处理检测结果（线程安全）"""
        # 计算处理耗时
        process_time = time.time() - start_time
        fps = 1 / process_time if process_time > 0 else 0

        # 更新界面
        self.show_all(img, detection_data)
        self.update_status_bar(f"实时FPS: {fps:.2f} | 队列深度: {self.frame_queue.qsize()}")

        # 保存结果
        self.save_detection_result(detection_data, img, img_path, process_time)

    # ================== 性能优化方法 ==================
    def batch_predict(self, img_batch):
        """批量推理优化"""
        return self.model(img_batch, imgsz=640, batch=self.batch_size, device=self.device)

    def async_save_result(self, img, path):
        """异步保存结果"""
        saver_thread = QThread()
        saver_thread.run = lambda: cv2.imwrite(path, img)
        saver_thread.start()

    # ================== 界面更新优化 ==================
    def show_all(self, img, info):
        """优化显示性能"""
        # 使用QPixmap缓存优化
        if not hasattr(self, 'last_pixmap'):
            self.last_pixmap = QtGui.QPixmap()

        shrink = resize_with_padding(img, self.label_img.width(), self.label_img.height(),
                                     [padvalue[2], padvalue[1], padvalue[0]])
        shrink = cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB)

        if self.last_pixmap.isNull() or self.last_pixmap.size() != self.label_img.size():
            self.last_pixmap = QtGui.QPixmap(self.label_img.size())

        painter = QtGui.QPainter(self.last_pixmap)
        painter.drawImage(0, 0, QtGui.QImage(shrink.data, shrink.shape[1], shrink.shape[0],
                                             shrink.shape[1] * 3, QtGui.QImage.Format_RGB888))
        painter.end()

        self.label_img.setPixmap(self.last_pixmap)
        self.show_info(info)

    # ================== 其他优化改进 ==================
    def closeEvent(self, event):
        """安全关闭线程"""
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread.quit()
        for thread in self.detection_threads:
            thread.quit()
        event.accept()

    def start(self):
        self.update_comboBox_default()
        try:
            btn_text = self.pushButton_start.text()
            # 输入类型校验逻辑合并
            if self.start_type is None:
                QMessageBox.information(self, "信息", "请先选择输入类型！", QMessageBox.Yes)
                return

            # 图像处理分支前置
            if self.start_type == 'img':
                self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                _, result_data = self.predict_img(self.img)
                # 简化暴力检测判断
                if result_data.get('cls_name', '') == 'violence':
                    winsound.Beep(2500, 500)
                self.show_all(self.img_show, result_data)
                return  # 提前返回减少嵌套层次

            # 视频/目录/摄像头统一处理
            is_detecting = btn_text == '启动检测 ▶'
            state_changed = False

            if is_detecting:
                self.timer.start(20)
                state_changed = True
                if self.start_type == 'camera':
                    self.label_camera_path.setText(" 摄像头已打开")
            else:
                self.timer.stop()
                state_changed = True
                if self.start_type == 'camera':
                    self.label_camera_path.setText(" 摄像头已关闭")
                    if hasattr(self, 'video'):
                        self.video.release()

            if state_changed:
                new_text = "结束检测 ▶" if is_detecting else "启动检测 ▶"
                self.pushButton_start.setText(new_text)

        except Exception:
            traceback.print_exc()

    def update_frame(self):
        if self.start_type == 'dir':
            # 检查是否处理完文件夹中的所有图像
            if self.current_index >= len(self.image_files):
                self.pushButton_start.setText("启动检测 ▶")
                self.timer.stop()
                self.frame_number = 0  # 重置帧计数器
                if self.current_index >= len(self.image_files):
                    QMessageBox.information(self, "信息", "此文件夹已识别完", QMessageBox.Yes)
                return

            # 获取当前图像的名称和路径
            self.img_name = self.image_files[self.current_index]
            self.img_path = os.path.join(self.img_path_dir, self.img_name)
            print('正在处理第%d张图片：%s' % (self.current_index + 1, self.img_name))
            # 读取图像并解码
            self.img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), -1)
            # 更新索引以处理下一张图像
            self.current_index += 1

        elif self.start_type in ['video', 'camera']:
            if self.frame_number == 0 and self.start_type == 'video':
                # 对于视频，处理第一帧
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, self.img = self.video.read()
                if ret:
                    # 获取当前帧号
                    frame_number = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
                    # 设置图像名称
                    self.img_name = f"{self.video_name}_{frame_number}.jpg"
                    self.img_path = self.video_path
                    self.frame_number = 0  # 重置帧计数器
                else:
                    # 如果读取失败，停止计时器并释放视频资源
                    self.pushButton_start.setText("启动检测 ▶")
                    self.timer.stop()
                    self.video.release()
                    self.frame_number = 0  # 重置帧计数器
                    QMessageBox.information(self, "信息", "视频识别已完成", QMessageBox.Yes)
                    return
            else:
                # 读取下一帧
                ret, self.img = self.video.read()
                if not ret:
                    # 如果读取失败，停止计时器并释放视频资源
                    self.pushButton_start.setText("启动检测 ▶")
                    self.timer.stop()
                    self.video.release()
                    self.frame_number = 0  # 重置帧计数器
                    if self.start_type == 'video':
                        QMessageBox.information(self, "信息", "视频识别已完成", QMessageBox.Yes)
                    elif self.start_type == 'camera':
                        self.label_camera_path.setText(" 摄像头已关闭")
                        QMessageBox.information(self, "信息", "摄像头关闭", QMessageBox.Yes)
                    return
                if self.start_type == 'video':
                    # 获取当前帧号并设置图像名称
                    frame_number = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
                    self.img_name = f"{self.video_name}_{frame_number}.jpg"
                    self.img_path = self.video_path
                elif self.start_type == 'camera':
                    # 对于摄像头，增加帧号并设置图像名称
                    self.frame_number += 1
                    self.img_name = f"camera_{self.frame_number}.jpg"
                    self.img_path = 'camera'

        # 进行图像预测
        results, result_info = self.predict_img(self.img)
        # 显示识别结果
        self.show_all(self.img_show, result_info)
        # 遍历字典
        for key, value in result_info.items():
            # 判断 cls_name 值是否为 violence
            if key == 'cls_name':
                if value == 'violence':
                    # QMessageBox.information(self, "警告", "图片中存在暴力行为！", QMessageBox.Yes)
                    # 声音提醒
                    winsound.Beep(2500, 500)
        if self.start_type == 'video':
            # 对于视频，增加帧号以处理下一帧
            self.frame_number += 1

    def reset_detection_state(self):
        """统一的重置检测状态方法"""
        self.pushButton_start.setText("启动检测 ▶")
        self.timer.stop()
        self.current_index = 0
        self.frame_number = 0
        if self.start_type == 'camera':
            self.label_camera_path.setText(" 摄像头已关闭")
        if hasattr(self, 'video') and self.video.isOpened():
            self.video.release()


    def predict_img(self, img):
        # 初始化结果信息字典
        result_info = {}
        # 记录开始时间以计算处理时间
        t1 = time.time()
        # 设置结果图像的路径
        self.result_img_name = os.path.join(self.result_img_path, self.img_name)
        # 模型识别
        self.detection_data = yolo.predict(img, imgsz=imgsz, conf=conf_thres, device=device, classes=classes)
        # 整理格式
        self.detection_data = format_data(self.detection_data)
        # 计算并记录消耗时间
        self.consum_time = str(round(time.time() - t1, 2)) + 's'
        # 记录输入时间
        self.input_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 将结果写入文件
        with open(self.result_txt, 'a+') as result_file:
            result_file.write(
                str([self.number, self.img_path, self.input_time, self.detection_data, len(self.detection_data), self.consum_time,
                     self.result_img_name])[1:-1])
            result_file.write('\n')

        # 显示识别信息表格
        self.show_table()
        # 增加编号
        self.number += 1
        # 获取下拉列表的值
        self.get_comboBox_value(self.detection_data)

        if len(self.detection_data) > 0:
            # 如果有识别结果，获取第一个结果的信息
            box = self.detection_data[0][2]
            score = self.detection_data[0][1]
            cls_name = self.detection_data[0][0]
        else:
            # 如果无识别结果，设置默认值
            box = [0, 0, 0, 0]
            score = 0
            cls_name = '无目标'

        # 格式化结果信息
        result_info = result_info_format(result_info, box, score, cls_name)
        # 在图像上绘制识别结果
        self.img_show = draw_info(img, self.detection_data)
        # 保存结果图像
        cv2.imencode('.jpg', self.img_show)[1].tofile(self.result_img_name)

        # 准确率计算逻辑
        # 修改准确率计算和存储方式
        if len(self.detection_data) > 0:
            total_score = sum([x[1] for x in self.detection_data])
            accuracy = total_score / len(self.detection_data)
        else:
            accuracy = 0.0

        # 将准确率存储为浮点数（原代码使用字符串格式化）
        result_info['accuracy'] = accuracy
        if self.device == 'cuda':
            img = cv2.cuda_GpuMat(img)  # 使用GPU加速图像处理
            img = cv2.cuda.cvtColor(img, cv2.COLOR_BGR2RGB)

        return self.detection_data, result_info

    def get_comboBox_value(self, detection_data):
        '''
        获取当前所有的类别和ID，点击下拉列表时，使用
        '''
        # 默认第一个是 所有目标
        lst = ["所有目标"]
        for bbox in detection_data:
            cls_name = bbox[0]
            lst.append(str(cls_name))
        self.comboBox_value = lst

    def show_info(self, result):
        try:
            if len(result) == 0:
                print("未识别到目标")
                return
            cls_name = result['cls_name']
            if len(self.chinese_name) > 3:
                cls_name = self.chinese_name[cls_name]
            if len(cls_name) > 10:
                # 当字符串太长时，显示不完整
                lst_cls_name = cls_name.split('_')
                cls_name = lst_cls_name[0][:10] + '...'

            self.label_class.setText(str(cls_name))
            self.label_score.setText(str(result['score']))
            self.label_xmin_v.setText(str(result['label_xmin_v']))
            self.label_ymin_v.setText(str(result['label_ymin_v']))
            self.label_xmax_v.setText(str(result['label_xmax_v']))
            self.label_ymax_v.setText(str(result['label_ymax_v']))
            self.update()  # 刷新界面

            # 修改后的准确率显示逻辑（确保使用正确的字段）
            accuracy = result.get('accuracy', 0.0)  # 直接获取存储的准确率值
            self.label_accuracy_value.setText(f"{accuracy:.2f}")  # 始终显示两位小数
            self.label_accuracy_value.setStyleSheet("font: 11pt '黑体';")

        except Exception as e:
            traceback.print_exc()

    def update_comboBox_default(self):
        """
        将下拉列表更新为 所有目标 默认状态
        """
        # 清空内容
        self.comboBox.clear()
        # 添加更新内容
        self.comboBox.addItems([self.comboBox_text])


    def show_table(self):
        try:
            # 显示表格
            self.RowLength = self.RowLength + 1
            self.tableWidget_info.setRowCount(self.RowLength)
            for column, content in enumerate(
                    [self.number, self.img_path, self.input_time, self.detection_data, len(self.detection_data), self.consum_time,
                     self.result_img_name]):
                # self.tableWidget_info.setColumnWidth(3, 0)  # 将第二列的宽度设置为0，即不显示
                row = self.RowLength - 1
                item = QtWidgets.QTableWidgetItem(str(content))
                # 居中
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                # 设置字体颜色
                item.setForeground(QColor.fromRgb(column_color[0], column_color[1], column_color[2]))
                self.tableWidget_info.setItem(row, column, item)
            # 滚动到底部
            self.tableWidget_info.scrollToBottom()
        except Exception as e:
            traceback.print_exc()

    def write_files(self):
        """
        导出 excel、csv 数据
        """
        path, filetype = QFileDialog.getSaveFileName(None, "另存为", self.ProjectPath,
                                                     "Excel 工作簿(*.xls);;CSV (逗号分隔)(*.csv)")
        with open(self.result_txt, 'r') as f:
            lst_txt = f.readlines()
            data = [list(eval(x.replace('\n', ''))) for x in lst_txt]

        if path == "":  # 未选择
            return
        if filetype == 'Excel 工作簿(*.xls)':
            writexls(data, path)
        elif filetype == 'CSV (逗号分隔)(*.csv)':
            writecsv(data, path)
        QMessageBox.information(None, "成功", "数据已保存！", QMessageBox.Yes)

# UI.ui转UI.py
# pyuic5 -x UI.ui -o UI.py
if __name__ == "__main__":
    path_cfg = 'config/configs.yaml'
    config = get_config()
    config.merge_from_file(path_cfg)
    # 加载模型相关的参数配置
    cfg_model = config.MODEL
    weights = 'weights\BASC.pt'
    conf_thres = float(cfg_model. DETECTION_THRESHOLD)
    classes = eval(cfg_model.CLASSES)
    imgsz = int(cfg_model.IMGSIZE)
    device = cfg_model.DEVICE
    # 加载UI界面相关的配置
    cfg_UI = config.UI
    background_img = cfg_UI.background_img
    padvalue = cfg_UI.padding_value
    column_widths = cfg_UI.column_widths
    column_color = cfg_UI.column_color
    title = cfg_UI.title
    label_title = cfg_UI.label_title
    zhutu = cfg_UI.main_img
    label_info_txt = cfg_UI.label_info_txt
    label_info_color = cfg_UI.label_info_color
    start_button_bg = cfg_UI.start_button_bg
    start_button_font = cfg_UI.start_button_font
    export_button_bg = cfg_UI.export_button_bg
    export_button_font = cfg_UI.export_button_font
    label_control_color = cfg_UI.label_control_color
    label_img_color = cfg_UI.label_img_color
    header_background_color = cfg_UI.table_widget_info_styles.header_background_color
    header_color = cfg_UI.table_widget_info_styles.header_color
    background_color = cfg_UI.table_widget_info_styles.background_color
    item_hover_background_color = cfg_UI.table_widget_info_styles.item_hover_background_color
    # 加载通用配置
    camera_num = int(config.CONFIG.camera_num)
    chinese_name = config.CONFIG.chinese_name
    # 模型加载
    yolo = YOLO(weights)
    # 模型预热
    yolo.predict(np.zeros((300, 300, 3), dtype='uint8'), device=device)

    # 创建QApplication实例
    app = QApplication([])
    # 创建自定义的主窗口对象
    window = DetectionMainWindow(config)
    # 显示窗口
    window.show()
    # 运行应用程序
    app.exec_()
