# -*- coding: utf-8 -*-
import time
from PyQt5.QtWidgets import QApplication , QMainWindow, QFileDialog,QMessageBox,QWidget,QHeaderView,QTableWidgetItem, QAbstractItemView, QLabel, QPushButton
import sys
import os
from PIL import ImageFont
from ultralytics import YOLO
sys.path.append('UIProgram')
from UIProgram.UiMain import Ui_MainWindow
import sys
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal,QCoreApplication, QPoint, QRect
from PyQt5.QtGui import QPixmap, QImage
import detect_tools as tools
import cv2
import Config
from UIProgram.QssLoader import QSSLoader
from UIProgram.precess_bar import ProgressBar
import numpy as np
# import torch

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initMain()
        self.signalconnect()

        # 加载css渲染效果
        style_file = 'UIProgram/style.css'
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)

    def signalconnect(self):
        self.ui.PicBtn.clicked.connect(self.open_img)
        self.ui.comboBox.activated.connect(self.combox_change)
        self.ui.VideoBtn.clicked.connect(self.open_img)
        self.ui.CapBtn.clicked.connect(self.camera_show)
        self.ui.SaveBtn.clicked.connect(self.save_detect_video)
        self.ui.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.ui.FilesBtn.clicked.connect(self.detact_batch_imgs)

        # 重播按钮（始终可见）
        self.replayBtn = QPushButton(self.ui.groupBox_4)
        self.replayBtn.setGeometry(QRect(135, 25, 120, 40))
        self.replayBtn.setText('重播')
        self.replayBtn.setToolTip('重播当前视频')
        self.replayBtn.setStyleSheet("font-size:12px; border:1px solid #4CAF50; border-radius:4px; background:#e8f5e9; font-weight:bold; color:#2e7d32;")
        self.replayBtn.clicked.connect(self.replay_video)
        self.replayBtn.setVisible(True)

        # 记录最后打开的视频路径
        self.last_video_path = None

    def initMain(self):
        self.show_width = 770
        self.show_height = 480

        self.org_path = None

        self.is_camera_open = False
        self.cap = None

        # self.device = 0 if torch.cuda.is_available() else 'cpu'

        # 加载检测模型
        self.model = YOLO(Config.model_path, task='detect')
        self.model(np.zeros((48, 48, 3)))  #预先加载推理模型
        self.fontC = ImageFont.truetype("Font/platech.ttf", 25, 0)

        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()

        # 更新视频图像
        self.timer_camera = QTimer()

        # 更新检测信息表格
        # self.timer_info = QTimer()
        # 保存视频
        self.timer_save_video = QTimer()

        # 表格
        self.ui.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(40)
        self.ui.tableWidget.setColumnWidth(0, 80)  # 设置列宽
        self.ui.tableWidget.setColumnWidth(1, 200)
        self.ui.tableWidget.setColumnWidth(2, 150)
        self.ui.tableWidget.setColumnWidth(3, 90)
        self.ui.tableWidget.setColumnWidth(4, 230)
        # self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 表格铺满
        # self.ui.tableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        # self.ui.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 设置表格不可编辑
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置表格整行选中
        self.ui.tableWidget.verticalHeader().setVisible(False)  # 隐藏列标题
        self.ui.tableWidget.setAlternatingRowColors(True)  # 表格背景交替

        # 设置下拉框最小宽度，防止文字截断
        self.ui.comboBox.setMinimumWidth(260)
        self.ui.comboBox.view().setMinimumWidth(260)

        # 图片缩放功能（视窗裁剪方式）
        self.zoom_factor = 1.0
        self.zoom_min = 1.0
        self.zoom_max = 5.0
        self.viewport_x = 0.5  # 视窗中心X（0~1，相对于原图比例）
        self.viewport_y = 0.5  # 视窗中心Y
        self.drag_pos = None
        self.is_dragging = False
        self.draw_img = None
        # 在label_show上安装事件过滤器
        self.ui.label_show.installEventFilter(self)
        self.ui.label_show.setMouseTracking(True)

        # 设置主页背景图片border-image: url(:/icons/ui_imgs/icons/camera.png)
        # self.setStyleSheet("#MainWindow{background-image:url(:/bgs/ui_imgs/bg3.jpg)}")

    def eventFilter(self, obj, event):
        """事件过滤器：处理图片缩放和拖动（视窗裁剪方式）"""
        if obj == self.ui.label_show:
            if event.type() == event.Wheel:
                if self.draw_img is None:
                    return False
                delta = event.angleDelta().y()
                if delta > 0:
                    self.zoom_factor = min(self.zoom_factor * 1.05, self.zoom_max)
                else:
                    self.zoom_factor = max(self.zoom_factor / 1.05, self.zoom_min)
                self._render_viewport()
                return True

            elif event.type() == event.MouseButtonPress:
                if event.button() == Qt.LeftButton and self.zoom_factor > 1.0:
                    self.is_dragging = True
                    self.drag_pos = event.pos()
                    self.ui.label_show.setCursor(Qt.ClosedHandCursor)
                    return True

            elif event.type() == event.MouseMove:
                if self.is_dragging and self.drag_pos is not None and self.draw_img is not None:
                    delta = event.pos() - self.drag_pos
                    self.drag_pos = event.pos()
                    h, w = self.draw_img.shape[:2]
                    # 降低拖动灵敏度
                    sensitivity = 0.0005
                    self.viewport_x -= delta.x() * sensitivity
                    self.viewport_y -= delta.y() * sensitivity
                    # 限制范围
                    half_frac = 0.5 / self.zoom_factor
                    self.viewport_x = max(half_frac, min(1.0 - half_frac, self.viewport_x))
                    self.viewport_y = max(half_frac, min(1.0 - half_frac, self.viewport_y))
                    self._render_viewport()
                    return True

            elif event.type() == event.MouseButtonRelease:
                if event.button() == Qt.LeftButton:
                    self.is_dragging = False
                    self.drag_pos = None
                    self.ui.label_show.setCursor(Qt.ArrowCursor)
                    return True

        return super().eventFilter(obj, event)

    def _render_viewport(self):
        """根据zoom_factor和viewport_x/y裁剪并显示图片区域"""
        if self.draw_img is None:
            return
        h, w = self.draw_img.shape[:2]
        # 实时获取label尺寸
        lw = self.ui.label_show.width()
        lh = self.ui.label_show.height()
        if lw <= 0 or lh <= 0:
            return
        # 视窗大小（在原图中的可见区域）
        view_w = w / self.zoom_factor
        view_h = h / self.zoom_factor
        # 视窗左上角（原图坐标）
        x1 = int(self.viewport_x * w - view_w / 2)
        y1 = int(self.viewport_y * h - view_h / 2)
        x2 = int(x1 + view_w)
        y2 = int(y1 + view_h)
        # 边界保护
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        # 裁剪并缩放到label固定尺寸
        cropped = self.draw_img[y1:y2, x1:x2]
        if cropped.size == 0:
            return
        display = cv2.resize(cropped, (lw, lh))
        pix_img = tools.cvimg_to_qpiximg(display)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

    def open_img(self):
        if self.cap:
            self.video_stop()
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.cap = None

        # 统一打开文件：支持图片和视频
        file_path, _ = QFileDialog.getOpenFileName(None, '打开文件', './',
            "所有支持的文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.gif *.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.gif);;视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;所有文件 (*)")
        if not file_path:
            return

        ext = file_path.split('.')[-1].lower()
        vid_suffix = ['mp4','avi','mkv','mov','wmv','flv','webm']

        # 统一更新两个路径显示
        self.ui.PiclineEdit.setText(file_path)
        self.ui.VideolineEdit.setText(file_path)
        self.ui.PiclineEdit.setText(file_path)
        self.ui.VideolineEdit.setText(file_path)

        if ext in vid_suffix:
            # 视频模式
            self.org_path = file_path
            self.last_video_path = file_path
            self.replayBtn.setVisible(False)
            self.cap = cv2.VideoCapture(file_path)
            self.video_start()
            self.ui.comboBox.setDisabled(False)
            return

        # 图片模式
        self.replayBtn.setVisible(False)
        self.ui.comboBox.setDisabled(False)
        self.org_path = file_path
        self.org_img = tools.img_cvread(self.org_path)

        # 目标检测
        t1 = time.time()
        self.results = self.model(self.org_path)[0]
        t2 = time.time()
        take_time_str = '{:.3f} s'.format(t2 - t1)
        self.ui.time_lb.setText(take_time_str)

        location_list = self.results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = self.results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each*100) for each in self.conf_list]

        total_nums = len(location_list)
        cls_percents = []
        for i in range(6):
            res = self.cls_list.count(i) / total_nums
            cls_percents.append(res)
        self.set_percent(cls_percents)

        # now_img = self.cv_img.copy()
        # for loacation, type_id, conf in zip(self.location_list, self.cls_list, self.conf_list):
        #     type_id = int(type_id)
        #     color = self.colors(int(type_id), True)
        #     # cv2.rectangle(now_img, (int(x1), int(y1)), (int(x2), int(y2)), colors(int(type_id), True), 3)
        #     now_img = tools.drawRectBox(now_img, loacation, Config.CH_names[type_id], self.fontC, color)
        now_img = self.results.plot()
        self.draw_img = now_img
        self.zoom_factor = 1.0
        self.viewport_x = 0.5
        self.viewport_y = 0.5
        self._render_viewport()
        # 设置路径显示
        self.ui.PiclineEdit.setText(self.org_path)
        self.ui.VideolineEdit.setText(self.org_path)
        self.ui.PiclineEdit.setText(self.org_path)
        self.ui.VideolineEdit.setText(self.org_path)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))

        # 设置目标选择下拉框（按类别归类，最多6类）
        choose_list = ['全部']
        detected_classes = sorted(set(self.cls_list))
        for cls_id in detected_classes:
            count = self.cls_list.count(cls_id)
            choose_list.append(f'{Config.CH_names[cls_id]}({count}个)')

        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(choose_list)

        if target_nums >= 1:
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
        #   默认显示第一个目标框坐标
        #   设置坐标位置值
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.ui.type_lb.setText('')
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

        # # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list,path=self.org_path)


    def detact_batch_imgs(self):
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.cap = None

        # 支持选择文件夹、单张图片或视频
        file_path, _ = QFileDialog.getOpenFileName(
            None, '选择图片或视频', './',
            "所有支持的文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.gif *.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.gif);;视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;所有文件 (*)")

        if not file_path:
            # 用户取消文件选择，尝试选文件夹
            directory = QFileDialog.getExistingDirectory(self, "或选取文件夹", "./")
            if not directory:
                return
            # 文件夹模式
            self.org_path = directory
            img_suffix = ['jpg','jpeg','png','bmp','tiff','webp','gif']
            file_list = []
            for file_name in os.listdir(directory):
                full_path = os.path.join(directory, file_name)
                if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                    file_list.append(full_path)
            if not file_list:
                QMessageBox.about(self, '提示', '所选文件夹中没有找到图片文件！')
                return
            self._run_batch_detect(file_list)
            return

        # 单文件模式
        ext = file_path.split('.')[-1].lower()
        img_suffix = ['jpg','jpeg','png','bmp','tiff','webp','gif']
        vid_suffix = ['mp4','avi','mkv','mov','wmv','flv','webm']

        if ext in vid_suffix:
            # 视频模式：打开视频检测
            self.org_path = file_path
            self.last_video_path = file_path
            self.ui.VideolineEdit.setText(file_path)
            self.ui.PiclineEdit.setText(file_path)
            self.cap = cv2.VideoCapture(file_path)
            self.video_start()
            self.ui.comboBox.setDisabled(False)
            return
        elif ext in img_suffix:
            # 单张图片模式
            self._run_batch_detect([file_path])
            return
        else:
            QMessageBox.about(self, '提示', '不支持的文件格式！')
            return

    def _run_batch_detect(self, file_list):
        """批量检测图片列表"""
        self.org_path = file_list[0] if len(file_list) == 1 else os.path.dirname(file_list[0])
        for img_path in file_list:
                self.org_img = tools.img_cvread(img_path)
                # 目标检测
                t1 = time.time()
                self.results = self.model(img_path)[0]
                t2 = time.time()
                take_time_str = '{:.3f} s'.format(t2 - t1)
                self.ui.time_lb.setText(take_time_str)

                location_list = self.results.boxes.xyxy.tolist()
                self.location_list = [list(map(int, e)) for e in location_list]
                cls_list = self.results.boxes.cls.tolist()
                self.cls_list = [int(i) for i in cls_list]
                self.conf_list = self.results.boxes.conf.tolist()
                self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

                total_nums = len(location_list)
                cls_percents = []
                for i in range(6):
                    res = self.cls_list.count(i) / total_nums
                    cls_percents.append(res)
                self.set_percent(cls_percents)

                now_img = self.results.plot()

                self.draw_img = now_img
                # 获取缩放后的图片尺寸
                self.img_width, self.img_height = self.get_resize_size(now_img)
                resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
                pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                self.ui.label_show.setPixmap(pix_img)
                self.ui.label_show.setAlignment(Qt.AlignCenter)
                # 设置路径显示
                self.ui.PiclineEdit.setText(img_path)
                self.ui.VideolineEdit.setText(img_path)
                self.ui.PiclineEdit.setText(img_path)
                self.ui.VideolineEdit.setText(img_path)

                # 目标数目
                target_nums = len(self.cls_list)
                self.ui.label_nums.setText(str(target_nums))

                # 设置目标选择下拉框（按类别归类）
                choose_list = ['全部']
                detected_classes = sorted(set(self.cls_list))
                for cls_id in detected_classes:
                    count = self.cls_list.count(cls_id)
                    choose_list.append(f'{Config.CH_names[cls_id]}({count}个)')

                self.ui.comboBox.clear()
                self.ui.comboBox.addItems(choose_list)

                if target_nums >= 1:
                    self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
                    self.ui.label_conf.setText(str(self.conf_list[0]))
                    #   默认显示第一个目标框坐标
                    #   设置坐标位置值
                    self.ui.label_xmin.setText(str(self.location_list[0][0]))
                    self.ui.label_ymin.setText(str(self.location_list[0][1]))
                    self.ui.label_xmax.setText(str(self.location_list[0][2]))
                    self.ui.label_ymax.setText(str(self.location_list[0][3]))
                else:
                    self.ui.type_lb.setText('')
                    self.ui.label_conf.setText('')
                    self.ui.label_xmin.setText('')
                    self.ui.label_ymin.setText('')
                    self.ui.label_xmax.setText('')
                    self.ui.label_ymax.setText('')

                # # 删除表格所有行
                # self.ui.tableWidget.setRowCount(0)
                # self.ui.tableWidget.clearContents()
                self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=img_path)
                self.ui.tableWidget.scrollToBottom()
                QApplication.processEvents()  #刷新页面

    def draw_rect_and_tabel(self, results, img):
        now_img = img.copy()
        location_list = results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

        for loacation, type_id, conf in zip(self.location_list, self.cls_list, self.conf_list):
            type_id = int(type_id)
            color = self.colors(int(type_id), True)
            now_img = tools.drawRectBox(now_img, loacation, Config.CH_names[type_id], self.fontC, color)

        self.draw_img = now_img
        self.zoom_factor = 1.0
        self.viewport_x = 0.5
        self.viewport_y = 0.5
        self._render_viewport()
        # 设置路径显示
        self.ui.PiclineEdit.setText(self.org_path)
        self.ui.VideolineEdit.setText(self.org_path)
        self.ui.PiclineEdit.setText(self.org_path)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))
        if target_nums >= 1:
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.ui.type_lb.setText('')
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)
        return now_img

    def combox_change(self):
        com_text = self.ui.comboBox.currentText()
        if not hasattr(self, 'results') or self.results is None:
            return
        if com_text == '全部':
            cur_box = self.location_list
            cur_img = self.results.plot()
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(str(self.conf_list[0]))
            # 坐标显示第一个目标
            self.ui.label_xmin.setText(str(cur_box[0][0]))
            self.ui.label_ymin.setText(str(cur_box[0][1]))
            self.ui.label_xmax.setText(str(cur_box[0][2]))
            self.ui.label_ymax.setText(str(cur_box[0][3]))
        elif '(' in com_text and com_text.endswith(')') and not com_text.endswith('个)'):
            # 类别模式：如"举手(3)" 或 "举手(3个)"
            cls_name = com_text.split('(')[0]
            cls_id = Config.CH_names.index(cls_name)
            # 找到该类别的所有目标
            indices = [i for i, c in enumerate(self.cls_list) if c == cls_id]
            if indices:
                # 只画框不加文字，用该类别颜色标出
                cur_img = self.org_img.copy()
                color = self.colors(cls_id, True)
                for idx in indices:
                    x1, y1, x2, y2 = self.location_list[idx]
                    cv2.rectangle(cur_img, (x1, y1), (x2, y2), color, 3)
                cur_box = [self.location_list[indices[0]]]
                self.ui.type_lb.setText(cls_name)
                self.ui.label_conf.setText(str(self.conf_list[indices[0]]))
                self.ui.label_xmin.setText(str(cur_box[0][0]))
                self.ui.label_ymin.setText(str(cur_box[0][1]))
                self.ui.label_xmax.setText(str(cur_box[0][2]))
                self.ui.label_ymax.setText(str(cur_box[0][3]))
            else:
                return
        else:
            return

        self.draw_img = cur_img
        self.viewport_x = 0.5
        self.viewport_y = 0.5
        self._render_viewport()


    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;所有文件 (*)")
        if not file_path:
            return None
        self.org_path = file_path
        self.ui.VideolineEdit.setText(file_path)
        self.ui.PiclineEdit.setText(file_path)
        self.ui.VideolineEdit.setText(file_path)
        return file_path

    def video_start(self):
        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()

        # 初始化视频模式下拉框（固定6类+全部）
        self._video_combo_initialized = False

        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(1)
        self.timer_camera.timeout.connect(self.open_frame)

    def tabel_info_show(self, locations, clses, confs, path=None):
        path = path
        for location, cls, conf in zip(locations, clses, confs):
            row_count = self.ui.tableWidget.rowCount()  # 返回当前行数(尾部)
            self.ui.tableWidget.insertRow(row_count)  # 尾部插入一行
            item_id = QTableWidgetItem(str(row_count+1))  # 序号
            item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中
            item_path = QTableWidgetItem(str(path))  # 路径
            # item_path.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            item_cls = QTableWidgetItem(str(Config.CH_names[cls]))
            item_cls.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_conf = QTableWidgetItem(str(conf))
            item_conf.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_location = QTableWidgetItem(str(location)) # 目标框位置
            # item_location.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            self.ui.tableWidget.setItem(row_count, 0, item_id)
            self.ui.tableWidget.setItem(row_count, 1, item_path)
            self.ui.tableWidget.setItem(row_count, 2, item_cls)
            self.ui.tableWidget.setItem(row_count, 3, item_conf)
            self.ui.tableWidget.setItem(row_count, 4, item_location)
        self.ui.tableWidget.scrollToBottom()

    def video_stop(self):
        self.cap.release()
        self.timer_camera.stop()
        self.ui.label_show.setText('视频播放结束')

    def replay_video(self):
        """重播当前视频"""
        if not self.last_video_path:
            return
        if not os.path.exists(self.last_video_path):
            return
        # 保存当前分类选择
        saved_idx = self.ui.comboBox.currentIndex()
        # 先停止当前播放
        try:
            self.timer_camera.stop()
        except:
            pass
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except:
            pass
        # 重置状态
        self._video_combo_initialized = False
        self.ui.label_show.clear()
        self.org_path = self.last_video_path
        self.cap = cv2.VideoCapture(self.last_video_path)
        if not self.cap.isOpened():
            return
        self.video_start()
        self.ui.comboBox.setDisabled(False)
        # 恢复分类选择
        if saved_idx < self.ui.comboBox.count():
            self.ui.comboBox.setCurrentIndex(saved_idx)
        # self.timer_info.stop()

    def open_frame(self):
        ret, now_img = self.cap.read()
        if ret:
            # 目标检测
            t1 = time.time()
            results = self.model(now_img)[0]
            t2 = time.time()
            take_time_str = '{:.3f} s'.format(t2 - t1)
            self.ui.time_lb.setText(take_time_str)

            location_list = results.boxes.xyxy.tolist()
            self.location_list = [list(map(int, e)) for e in location_list]
            cls_list = results.boxes.cls.tolist()
            self.cls_list = [int(i) for i in cls_list]
            self.conf_list = results.boxes.conf.tolist()
            self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

            total_nums = len(location_list)
            cls_percents = []
            for i in range(6):
                if total_nums!= 0 :  #识别到的目标数量
                    res = self.cls_list.count(i) / total_nums
                else :
                    res=0
                #res = self.cls_list.count(i) / total_nums
                cls_percents.append(res)
            self.set_percent(cls_percents)

            now_img = results.plot()

            # 检查当前下拉框选择，应用类别过滤
            com_text = self.ui.comboBox.currentText()
            if com_text != '全部' and '(' in com_text:
                cls_name = com_text.split('(')[0]
                if cls_name in Config.CH_names:
                    cls_id = Config.CH_names.index(cls_name)
                    indices = [i for i, c in enumerate(self.cls_list) if c == cls_id]
                    if indices:
                        # 在原图上只画选中类别的框
                        now_img = now_img.copy()
                        # 先清除所有框：用原始帧重绘
                        raw_frame = results.orig_img.copy()
                        color = self.colors(cls_id, True)
                        for idx in indices:
                            x1, y1, x2, y2 = self.location_list[idx]
                            cv2.rectangle(raw_frame, (x1, y1), (x2, y2), color, 3)
                        now_img = raw_frame

            # 获取缩放后的图片尺寸
            self.img_width, self.img_height = self.get_resize_size(now_img)
            resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.label_show.setPixmap(pix_img)
            self.ui.label_show.setAlignment(Qt.AlignCenter)

            # 目标数目
            target_nums = len(self.cls_list)
            self.ui.label_nums.setText(str(target_nums))

            # 固定6类下拉框，只更新人数（不触发信号，不影响用户选择）
            if not hasattr(self, '_video_combo_initialized') or not self._video_combo_initialized:
                self._video_combo_initialized = True
                init_list = ['全部'] + [f'{name}(0)' for name in Config.CH_names]
                self.ui.comboBox.blockSignals(True)
                self.ui.comboBox.clear()
                self.ui.comboBox.addItems(init_list)
                self.ui.comboBox.setCurrentIndex(0)
                self.ui.comboBox.blockSignals(False)

            # 更新每类的人数（直接setText，不触发重建）
            saved_idx = self.ui.comboBox.currentIndex()
            self.ui.comboBox.setItemText(0, '全部')
            for i in range(6):
                count = self.cls_list.count(i)
                self.ui.comboBox.setItemText(i + 1, f'{Config.CH_names[i]}({count})')
            # 保持用户选中的项不变
            if saved_idx >= 0 and saved_idx < self.ui.comboBox.count():
                self.ui.comboBox.setCurrentIndex(saved_idx)

            if target_nums >= 1:
                self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
                self.ui.label_conf.setText(str(self.conf_list[0]))
                #   默认显示第一个目标框坐标
                #   设置坐标位置值
                self.ui.label_xmin.setText(str(self.location_list[0][0]))
                self.ui.label_ymin.setText(str(self.location_list[0][1]))
                self.ui.label_xmax.setText(str(self.location_list[0][2]))
                self.ui.label_ymax.setText(str(self.location_list[0][3]))
            else:
                self.ui.type_lb.setText('')
                self.ui.label_conf.setText('')
                self.ui.label_xmin.setText('')
                self.ui.label_ymin.setText('')
                self.ui.label_xmax.setText('')
                self.ui.label_ymax.setText('')


            # 删除表格所有行
            # self.ui.tableWidget.setRowCount(0)
            # self.ui.tableWidget.clearContents()
            self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)

        else:
            self.cap.release()
            self.timer_camera.stop()

    def vedio_show(self):
        if self.is_camera_open:
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')

        video_path = self.get_video_path()
        if not video_path:
            return None
        self.last_video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.video_start()
        self.ui.comboBox.setDisabled(False)

    def camera_show(self):
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.ui.CaplineEdit.setText('摄像头开启')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
            self.ui.comboBox.setDisabled(False)
        else:
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.ui.label_show.setText('')
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
            self.ui.label_show.clear()

    def get_resize_size(self, img):
        _img = img.copy()
        img_height, img_width , depth= _img.shape
        ratio = img_width / img_height
        if ratio >= self.show_width / self.show_height:
            self.img_width = self.show_width
            self.img_height = int(self.img_width / ratio)
        else:
            self.img_height = self.show_height
            self.img_width = int(self.img_height * ratio)
        return self.img_width, self.img_height

    def save_detect_video(self):
        if self.cap is None and not self.org_path:
            QMessageBox.about(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
            return

        if self.is_camera_open:
            QMessageBox.about(self, '提示', '摄像头视频无法保存!')
            return

        if self.cap:
            res = QMessageBox.information(self, '提示', '保存视频检测结果可能需要较长时间，请确认是否继续保存？',QMessageBox.Yes | QMessageBox.No ,  QMessageBox.Yes)
            if res == QMessageBox.Yes:
                self.video_stop()
                com_text = self.ui.comboBox.currentText()
                self.btn2Thread_object = btn2Thread(self.org_path, self.model, com_text)
                self.btn2Thread_object.start()
                self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
            else:
                return
        else:
            if os.path.isfile(self.org_path):
                fileName = os.path.basename(self.org_path)
                name , end_name= fileName.rsplit(".",1)
                save_name = name + '_detect_result.' + end_name
                save_img_path = os.path.join(Config.save_path, save_name)
                # 保存图片
                cv2.imwrite(save_img_path, self.draw_img)
                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(save_img_path))
            else:
                img_suffix = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'gif']
                for file_name in os.listdir(self.org_path):
                    full_path = os.path.join(self.org_path, file_name)
                    if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                        name, end_name = file_name.rsplit(".",1)
                        save_name = name + '_detect_result.' + end_name
                        save_img_path = os.path.join(Config.save_path, save_name)
                        results = self.model(full_path)[0]
                        now_img = results.plot()
                        # 保存图片
                        cv2.imwrite(save_img_path, now_img)

                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(Config.save_path))


    def update_process_bar(self,cur_num, total):
        if cur_num == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()
        if cur_num >= total:
            self.progress_bar.close()
            QMessageBox.about(self, '提示', '视频保存成功!\n文件在{}目录下'.format(Config.save_path))
            return
        if self.progress_bar.isVisible() is False:
            # 点击取消保存时，终止进程
            self.btn2Thread_object.stop()
            return
        value = int(cur_num / total *100)
        self.progress_bar.setValue(cur_num, total, value)
        QApplication.processEvents()

    def set_percent(self, probs):
        # 显示各表情概率值
        items = [self.ui.progressBar, self.ui.progressBar_2, self.ui.progressBar_3, self.ui.progressBar_4,
                 self.ui.progressBar_5, self.ui.progressBar_6]
        labels = [self.ui.label_20, self.ui.label_21, self.ui.label_22, self.ui.label_23,
                  self.ui.label_24, self.ui.label_25]
        prob_values = [round(each * 100) for each in probs]
        label_values = ['{:.1f}%'.format(each * 100) for each in probs]
        for i in range(len(probs)):
            items[i].setValue(prob_values[i])
            labels[i].setText(label_values[i])


class btn2Thread(QThread):
    """
    进行检测后的视频保存
    """
    # 声明一个信号
    update_ui_signal = pyqtSignal(int,int)

    def __init__(self, path, model, com_text):
        super(btn2Thread, self).__init__()
        self.org_path = path
        self.model = model
        self.com_text = com_text
        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()
        self.is_running = True  # 标志位，表示线程是否正在运行

    def run(self):
        # VideoCapture方法是cv2库提供的读取视频方法
        cap = cv2.VideoCapture(self.org_path)
        # 设置需要保存视频的格式“xvid”
        # 该参数是MPEG-4编码类型，文件名后缀为.avi
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 设置视频帧频
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 设置视频大小
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # VideoWriter方法是cv2库提供的保存视频方法
        # 按照设置的格式来out输出
        fileName = os.path.basename(self.org_path)
        name, end_name = fileName.split('.')
        save_name = name + '_detect_result.avi'
        save_video_path = os.path.join(Config.save_path, save_name)
        out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] 视频总帧数：{}".format(total))
        cur_num = 0

        # 确定视频打开并循环读取
        while (cap.isOpened() and self.is_running):
            cur_num += 1
            print('当前第{}帧，总帧数{}'.format(cur_num, total))
            # 逐帧读取，ret返回布尔值
            # 参数ret为True 或者False,代表有没有读取到图片
            # frame表示截取到一帧的图片
            ret, frame = cap.read()
            if ret == True:
                # 检测
                results = self.model(frame)[0]
                frame = results.plot()
                out.write(frame)
                self.update_ui_signal.emit(cur_num, total)
            else:
                break
        # 释放资源
        cap.release()
        out.release()

    def stop(self):
        self.is_running = False




if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
