import argparse
import datetime
import os
import threading
import time
from collections import Counter, deque

import cv2
import imutils
import numpy as np
import requests
import supervision as sv
from imutils.video import VideoStream
from ultralytics import YOLO


class detectHandler:
    def __init__(self, src, log_filename = "log.txt", IMG_FILENAME = "img", ObjectID = 0, model = "yolov8l.pt"):
        self.src = src  # 影像來源
        self.ObjectID = ObjectID    # 識別ID
        self.model_path = model     # 識別模組路徑
        self.model = None     # 識別模組
        self.outputFrame = None     # 識別後的圖像
        self.lock = threading.Lock()    # 執行續的鎖
        self.inSide = False         # 來源影像內是否有識別到指定物件
        self.counter = Counter()    # 初始化計時器
        self.detect_thread = None   # 辨識執行續
        self.stopped = False  # 結束旗標

        self.TRIGER_COUNTER = 5  # 觸發inside狀態變換的連續幀數
        self.IMG_LIMIT_IN_FILE = 5	# 圖片資料夾內的照片數量限制
        self.IMG_FILENAME = IMG_FILENAME	# 圖片資料夾名稱
        self.log_filename = log_filename    # LOG檔案名稱

        # 建立兩個列表以儲存每一種 inSize 狀態下的檔案名稱
        self.img_filename_deque_true = deque(maxlen=10)
        self.img_filename_deque_false = deque(maxlen=10)

        # 設定識別物件的標註參數
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )
    
    # 以新執行續執行 detect
    def start(self):
        self.vs = VideoStream(src=self.src).start()
        time.sleep(2.0)
        self.detect_thread = threading.Thread(target=self.detect)
        self.detect_thread.daemon = True
        self.detect_thread.start()
    
    # 終止程序
    def stop(self):
        if self.vs is not None:
            self.vs.stop()
            self.stopped = True
        if self.detect_thread is not None:
            self.detect_thread.join()

    # 寫入 log 檔
    def save_Log(self, current_time, detections, inSide):
        # 獲取當前時間
        current_time = current_time.replace("_", "/")
        with open(self.log_filename, 'a') as file:
            if len(detections) > 0:
                count = 0
                # 從 detection 獲取資料
                for _, confidence, class_id, _ in detections:
                    count += 1
                    class_name = self.model.model.names[class_id]
                    # 寫入 log.txt
                    file.write(f"{current_time}:{class_name} {confidence:0.2f} {inSide} -From number{count}\n")
            else:
                file.write(f"{current_time}:No detection {inSide}\n")

    # 維護 img 目錄中的照片數量
    def maintain_img_limit(self, IMG_FILENAME):
        directory = os.path.join(IMG_FILENAME)
        # 檢查文件夾是否存在，如果不存在則建立
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError:
                print(f"Error: Creating directory {directory}")
                return

        # 獲取文件夾中所有照片的列表
        try:
            file_list = os.listdir(directory)
        except OSError:
            print(f"Error: Reading directory {directory}")
            return

        file_list.sort()  # 對文件名稱進行排序，保證最舊的照片會首先被刪除

        # 刪除多餘的照片
        while len(file_list) > self.IMG_LIMIT_IN_FILE :
            try:
                os.remove(os.path.join(directory, file_list.pop(0)))
            except OSError:
                print(f"Error: Deleting file in {directory}")
                return

        return
    
    # 回傳識別後的 frame
    def get_outputframe(self):
        return self.outputFrame
    
    # 獲取全部的 log
    def get_Log(self):
        with open(self.log_filename, 'r') as file:
            log = file.read()
        return log
    
    # 獲取最後一項 log
    def get_final_log(self):
        logs = self.get_Log().split('\n')
        return logs[-2] if len(logs) > 1 else None
    
    # 獲取最新的一張識別到的照片
    def get_final_img(self, IMG_FILENAME):
        while True:
            # 等待鎖
            with self.lock:
                directory = os.path.join(IMG_FILENAME)
                # 檢查文件夾是否存在
                if not os.path.exists(directory):
                    print(f"Error: Finding directory {directory}")
                    return None

                # 獲取文件夾中所有照片的列表
                try:
                    file_list = os.listdir(directory)
                except OSError:
                    print(f"Error: Reading directory {directory}")
                    return None
                
                if not file_list:  # 如果列表為空，直接返回 None
                    return None

                # 找到最新的圖片（列表中的最後一個元素）
                filename = file_list[-1]
                img_path = os.path.join(directory, filename)

                frame = cv2.imread(img_path)
                # 以 JPEG 格式編碼
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                
            if flag:
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                    bytearray(encodedImage) + b'\r\n')
            else:
                return None

    # 回傳字元格式的識別圖像
    def get_detected_image_in_byte(self):
        # 循環輸出流中的幀
        while True:
            # 等待直到獲取鎖
            with self.lock:
                # 檢查輸出幀是否可用，否則跳過
                # 循環的迭代次數
                if self.get_outputframe() is None:
                    continue
                # 將幀編碼為 JPEG 格式
                (flag, encodedImage) = cv2.imencode(".jpg", self.get_outputframe())
                # 確保幀已成功編碼
                if not flag:
                    continue
            # 產生字節格式的輸出幀
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')

    # 分辨
    def detect(self):
        # 取得 model
        if self.model is None:
            self.model = YOLO(self.model_path)
            
        # 循環視頻流中的幀
        while True:
            frame = self.vs.read()
            # 用YOLO模型進行物件偵測
            result = self.model(frame, agnostic_nms=True,conf=0.5)[0]
            detections = sv.Detections.from_yolov8(result)
            detections = detections[detections.class_id == self.ObjectID]   # 只擷取指定類型的物件

            # 在畫面上顯示物件的邊界框和標籤
            labels = [
                f"{self.model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections
            ]

            frame = self.box_annotator.annotate(
                scene=frame, 
                detections=detections, 
                labels=labels
            )

            # 將當前時間標示在圖片上
            timestamp = datetime.datetime.now()
            cv2.putText(frame, timestamp.strftime(
                "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
            # 如果偵測到的物件包含指定物件，則更新計數器的數值
            object_detected = len(detections) > 0
            self.counter[object_detected] += 1  # 當前狀態的次數+1
            self.counter[not object_detected] = 0   # 初始化另一個狀態

            # 如果符合觸發條件（例如連續五次都有(或沒有)偵測到貓咪），則更改 inSide 的值，並儲存當前幀的影像
            if (not self.inSide and self.counter[True] >= self.TRIGER_COUNTER) or (self.inSide and self.counter[False] >= self.TRIGER_COUNTER):
                self.inSide = not self.inSide

                # 使用 datetime 模組來格式化當前的時間，並作為檔名的一部分
                current_time = datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')
                # 偵測到指定物件時，載圖到img資料夾
                if self.inSide :
                    filename = f'{self.inSide}_{current_time}.png'
                    cv2.imwrite(os.path.join(self.IMG_FILENAME, filename), self.get_outputframe())

                    # 維護 img 目錄中的照片數量
                    self.maintain_img_limit(self.IMG_FILENAME)

                # 寫入 log 檔
                self.save_Log(current_time, detections, self.inSide)

            # 鎖打開了才執行
            with self.lock:
                self.outputFrame = frame.copy()
            
            # 結束程式
            if self.stopped:
                break
