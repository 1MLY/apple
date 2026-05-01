import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow,QDialog,QFileDialog,QMessageBox,QHeaderView,QTableWidgetItem
from PyQt5 import uic
from PyQt5.QtGui import QImage, qRgb ,QPixmap, QPalette, QBrush
from PyQt5 import QtCore
from PyQt5.QtCore import  Qt,QDateTime

import numpy as np
from ultralytics import YOLO

#import mysql.connector #数据库链接
import pymysql
from datetime import datetime


# QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 1、适应高DPI设备
# QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)  # 2、解决图片模糊
# QApplication.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

model = YOLO(r"best.pt");#装载网络模型

def cv_image_to_qt_image(cv_img):
    """Convert an OpenCV image to a Qt image (QImage)."""
    # Convert the color space from BGR to RGB
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # Convert the image data to a format that Qt can handle (QImage)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

    return q_img


def cv_image_to_qt_pixmap(cv_img):
    """Convert an OpenCV image to a Qt pixmap (QPixmap)."""
    q_img = cv_image_to_qt_image(cv_img)
    return QPixmap.fromImage(q_img)

def enable_high_dpi_scaling():
    """启用高DPI缩放支持"""
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, False)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, False)

#这个函数检测苹果
def detect_apple(frame, conf_threshold=0.5):
    """
    使用YOLO检测苹果
    Args:
        frame: 输入图像
        conf_threshold: 置信度阈值
    Returns:
        detections: [(x1, y1, x2, y2, confidence, class_id), ...]
    """
    # 执行检测
    results = model(frame, conf=conf_threshold, verbose=False)[0]
    
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            # 获取检测框信息
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            #计算成熟度
            ripeness, ripelevel = calculate_ripeness(frame, (x1, y1, x2, y2),True)
            #保存到列表内部
            #返回坐标位置 置信度 类别
            detections.append([x1, y1, x2, y2, conf, ripelevel, cls_id])
    
    return detections



def calculate_ripeness(image, rect,visualize):
    """
    在给定的矩形区域内去除树叶，基于颜色计算苹果成熟度

    参数:
        image: 原始图像 (BGR格式)
        rect: 矩形区域 (x1, y1, x2, y2)

    返回:
        ripeness: 成熟度分数 (0~1, 1表示完全成熟)
        result_img: 带标注的图像
    """
    x1, y1, x2, y2 = rect
    
    # 裁剪ROI
    roi = image[y1:y2, x1:x2].copy()
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ----- 1. 定义苹果的颜色范围（可根据实际成熟度调整）-----
    # 红色苹果（成熟）
    red_lower1 = np.array([0, 40, 40])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([155, 40, 40])
    red_upper2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(roi_hsv, red_lower1, red_upper1) + cv2.inRange(roi_hsv, red_lower2, red_upper2)

    # 绿色苹果（未成熟）
    green_lower = np.array([30, 40, 40])
    green_upper = np.array([65, 255, 255])
    mask_green = cv2.inRange(roi_hsv, green_lower, green_upper)

    # 合并红色和绿色区域（根据实际需要，可只保留红色）
    mask_apple = cv2.bitwise_or(mask_red, mask_green)

    # 也可以排除明显的绿色树叶（如果树叶为深绿且苹果偏红）
    # 树叶绿色范围更宽，可以反向操作：先提取苹果颜色，忽略纯绿色背景

    # ----- 2. 形态学清理 -----
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_apple = cv2.morphologyEx(mask_apple, cv2.MORPH_CLOSE, kernel, iterations=2)  # 填充孔洞
    mask_apple = cv2.morphologyEx(mask_apple, cv2.MORPH_OPEN, kernel, iterations=1)   # 去除小噪点

    # 可选：只保留最大的连通域（假设矩形内主要目标是一个苹果）
    contours, _ = cv2.findContours(mask_apple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask_apple = np.zeros_like(mask_apple)
        cv2.drawContours(mask_apple, [largest], -1, 255, -1)

    # ----- 3. 提取苹果区域（去除树叶）-----
    apple_only = cv2.bitwise_and(roi, roi, mask=mask_apple)

    # ----- 4. 计算成熟度 -----
    # 只统计掩膜内的像素
    valid_pixels = mask_apple > 0
    if np.sum(valid_pixels) == 0:
        return 0.0, image

    # 在HSV空间分析
    hsv_apple = cv2.cvtColor(apple_only, cv2.COLOR_BGR2HSV)
    hues = hsv_apple[:, :, 0][valid_pixels]        # 色调 0~180
    saturations = hsv_apple[:, :, 1][valid_pixels] # 饱和度 0~255

    # 定义红色色调范围（0-10 和 155-180）
    is_red = (hues <= 10) | (hues >= 155)
    # 定义绿色色调范围（35-65）
    is_green = (hues >= 30) & (hues <= 65)

    red_ratio = np.sum(is_red) / len(hues)
    green_ratio = np.sum(is_green) / len(hues)

    # 平均饱和度（归一化）
    avg_saturation = np.mean(saturations) / 255.0

    # 成熟度公式：红色占比越高越成熟，绿色占比抑制，饱和度辅助
    ripeness = red_ratio /(red_ratio + green_ratio) + 0.2 * avg_saturation
    ripeness = np.clip(ripeness, 0.0, 1.0)
    
    #计算等级
    if ripeness < 0.3:
        ripelevel = 0 #未成熟
    elif ripeness < 0.55:
        ripelevel = 1 #半成熟
    else:
        ripelevel = 2 #成熟
      
    # ----- 5. 可视化 -----
    if visualize:
        # 在原图上绘制矩形和成熟度文字
        result = image.copy()
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{ripeness:.2f}"
        cv2.putText(result, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 显示中间结果
        cv2.imshow("ROI", roi)
        cv2.imshow("Apple Mask", mask_apple)
        cv2.imshow("Apple Only", apple_only)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        result = image

    return ripeness, ripelevel



class MyMainWindow(QMainWindow):
    m_result = 0
    def __init__(self):
        super().__init__()


        self.init_ui()
        self.image = None #原始图像
        
        # 创建连接
        try:
            conn = pymysql.connect(
                host='localhost',
                user='root',
                password='265283llyLLY',
                database='myapple'
            )
            print("数据库连接成功")
        except mysql.connector.Error as e:
            print(f"连接失败: {e}")

        # 保存到self内
        self.conn = conn  # 数据库链接对象


    def init_ui(self):
        uic.loadUi('myui.ui',self)  # 动态加载UI文件
        self.setWindowTitle('智能苹果检测系统')

       

        #print(self.ui.__dict__)  # 查看ui文件中有哪些控件
        #self.ui.setWindowIcon(QIcon('img.png'))
        self.pushButtonload.clicked.connect(self.on_button_loadimg)  # 给选择图像按钮增加函数
        self.pushButtondetect.clicked.connect(self.on_button_detect)  # 给检测按钮增加函数

        self.pushButtonquit.clicked.connect(self.on_button_quit)  # 给退出按钮增加函数

        self.pushButtonlog.clicked.connect(self.on_button_log)  # 给检测记录按钮增加函数
        
        # self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.setRowCount(0)  # 设置行数
        self.tableWidget.setColumnCount(4)  # 设置列数
        
        #设置每一列的宽度
        self.tableWidget.setColumnWidth(0, 100)
        self.tableWidget.setColumnWidth(1, 100)
        self.tableWidget.setColumnWidth(2, 100)
        self.tableWidget.setColumnWidth(3, 200)
        # 设置表头
        self.tableWidget.setHorizontalHeaderLabels(["目标序号", "置信度", "成熟度", "坐标位置"])

        # ============== 关键设置：允许列宽拖动 ==============
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)  # 允许手动拖动调整列宽
        # header_item = QTableWidgetItem("自定义列名")
        # self.tableWidget.setHorizontalHeaderItem(0, header_item)  # 设置第0列的标题


        # 设置表格自适应宽度
       # self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    
    
    

    
    def on_button_quit(self):
        self.close()

    def on_button_loadimg(self):

        #print("按钮被点击了！")  # 定义槽函数，这里是按钮点击时的行为

        filename, _ = QFileDialog.getOpenFileName(None, "选择图片", "", "Images (*.jpg *.bmp *.png *.gif);;All Files (*.*)")

        if filename:
            print("选择的文件：", filename)

            #filename2 = filename.encode("utf-8")
            image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)

            #灰度图转三通道
            if len(image.shape) == 2:
                image = cv2.merge([image, image, image])

            if image.shape[2] == 4:
                # 如果事四通道图像，切片截取 BGR
                image = image[:, :, :3]

            q_img = cv_image_to_qt_pixmap(image)
            self.labelimg.setPixmap(q_img)
            self.labelimg.setScaledContents(True)  # 让图像自动缩放以适应标签大小

            self.strfilename = filename  # 原始图像
            self.image = image  # 原始图像


    def on_button_detect(self):

        if self.image is None:
            print('请载入图像！')
            QMessageBox.critical( self, "提示", "你还未载入图片！");
            return

        image = self.image
       
        self.tableWidget.setRowCount(0)  # 清空表格
        #检测苹果位置和成熟度
        detections = detect_apple(image,0.35);
        #检测的目标个数
        objnum = len(detections)
        
        self.tableWidget.setRowCount(objnum)  # 设置行数
        img = image.copy()
        #遍历每个检测目标
        ripenames = ['未成熟','半成熟','全成熟']
        index = 1
        for detection in detections:
            x1, y1, x2, y2, conf, ripelevel, cls_id = detection
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(img, f"{index:d}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)


            #目标序号
            strindex = str.format("%d" % (index))
            item = QTableWidgetItem(strindex)
            self.tableWidget.setItem(index-1, 0, item)
            
            # 置信度
            strconf = str.format("%.3f" % (conf))
            item = QTableWidgetItem(strconf)
            self.tableWidget.setItem(index - 1, 1, item)
            
            # 成熟度
            #不同的成熟度 显示不同的名称
            item = QTableWidgetItem(ripenames[ripelevel])
            self.tableWidget.setItem(index - 1, 2, item)

            
            # 坐标
            strpos = str.format("%d,%d,%d,%d" % (x1,y1,x2,y2))
            item = QTableWidgetItem(strpos)
            self.tableWidget.setItem(index - 1, 3, item)

            index = index+1

            
       

        strobjnum = str.format("%d" % (objnum))
        self.plainTextEdittotalnum.setPlainText(strobjnum)
        
        #存储日志记录
        self.addchecklog(self.strfilename,objnum)

        q_img = cv_image_to_qt_pixmap(img)
        self.labelimg.setPixmap(q_img)
        self.labelimg.setScaledContents(True)  # 让图像自动缩放以适应标签大小

    # 这个函数 插入一条检测日志
    #picpath 存储的图片路径 
    def addchecklog(self, picpath,objnum,formatted_now=None):
        if formatted_now is None:
            # 获取当前时间并格式化为字符串
            now = datetime.now()
            formatted_now = now.strftime('%Y-%m-%d %H:%M:%S')

        strsql = "insert loginfo (logtime,sourcepath,objnum) VALUES (%s, %s, %s)"
        val = (formatted_now,picpath,objnum)
        conn = self.conn
        try:
            with conn.cursor() as cursor:

                # 执行 SQL 语句
                cursor.execute(strsql, val)
                # 提交到数据库执行
                conn.commit()
                cursor.close()
        except Exception as e:
            # 如果发生错误则回滚
            print(f"发生错误：{e}")
            connection.rollback()

        # cursor = conn.cursor()
        #
        # # 执行SQL查询
        # cursor.execute(strsql, val)
        #
        # conn.commit()
        #cursor.close()
    
    #检测记录按钮对应函数
    def on_button_log(self):
        self.recordwindow = LogRecord(self, self.conn)  # 创建弹出窗口实例

        self.recordwindow.exec_()  # 显示弹出窗口（对于QDialog）或者show函数
        return

class LogRecord(QDialog):

    def __init__(self, parent=None, data=None):
        super().__init__()
        uic.loadUi("log.ui",self)

        self.setWindowTitle('检测记录查询')
        self.pushButtonquit.clicked.connect(self.on_button_quit)  # 给按钮增加函数
        self.pushButtonsearch.clicked.connect(self.on_button_search)  # 给按钮增加函数

        # self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.setRowCount(0)  # 设置行数
        self.tableWidget.setColumnCount(3)  # 设置列数
        
        #设置每一列的宽度
        self.tableWidget.setColumnWidth(0, 150)
        self.tableWidget.setColumnWidth(1, 250)
        self.tableWidget.setColumnWidth(2, 100)

        # 设置表头
        self.tableWidget.setHorizontalHeaderLabels(["检测时间", "检测图片路径","苹果个数"])

        # ============== 关键设置：允许列宽拖动 ==============
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)  # 允许手动拖动调整列宽
        # header_item = QTableWidgetItem("自定义列名")
        # self.tableWidget.setHorizontalHeaderItem(0, header_item)  # 设置第0列的标题
        
        self.dateTimeEdit.setDateTime(QDateTime.currentDateTime().addDays(-1))  # 默认1天前
        self.dateTimeEdit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.dateTimeEdit.setCalendarPopup(True)
        
        self.dateTimeEdit_2.setDateTime(QDateTime.currentDateTime())  # 默认当前时间
        self.dateTimeEdit_2.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.dateTimeEdit_2.setCalendarPopup(True)
        # 保存到self内
        self.conn = data  # 数据库链接对象,由主窗口传递来的

        
        #self.filltable()#装载表格

    def on_button_quit(self):

        self.close()

    def on_button_search(self):

        self.filltable()

    def filltable(self):
        conn = self.conn
        cursor = conn.cursor()
        
        #控件中获取选择的时间段
        start_time = self.dateTimeEdit.dateTime().toPyDateTime()
        end_time = self.dateTimeEdit_2.dateTime().toPyDateTime()
        
        #转为字符串
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        
        strsql = str.format("SELECT * FROM loginfo WHERE logtime BETWEEN '%s' AND '%s'" % (start_time_str,end_time_str))
        print(strsql)
        # 执行SQL查询
        cursor.execute(strsql)

        # 获取结果
        results = cursor.fetchall()
        rownum = len(results) #日志个数

        self.tableWidget.setRowCount(rownum)  # 设置行数

        index = 0
        for row in results:
            
            # 预警时间
            date_string = row[1].strftime("%Y-%m-%d %H:%M:%S")
            item = QTableWidgetItem(date_string)
            self.tableWidget.setItem(index, 0, item)

            # 源文件路径
            item = QTableWidgetItem(row[2])
            self.tableWidget.setItem(index, 1, item)

            # 苹果个数
            item = QTableWidgetItem(str(row[3]))
            self.tableWidget.setItem(index, 2, item)

           

            index = index+1
        return


    
if __name__ == "__main__":
    # 启用高DPI缩放
    enable_high_dpi_scaling()

    # 设置环境变量，解决高DPI问题
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
