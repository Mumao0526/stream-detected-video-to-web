<<<<<<< HEAD
=======
import argparse
>>>>>>> 2dcc45b4796a15a76bc9b29b4e4028c2c91bf4ab
import datetime
import os
import threading
import time
<<<<<<< HEAD
import imutils
from collections import Counter

import cv2
import supervision as sv
from imutils.video import VideoStream
from ultralytics import YOLO
from Logger import Logger
from ImageHandler import ImageHandler
=======
from collections import Counter, deque

import cv2
import imutils
import numpy as np
import requests
import supervision as sv
from imutils.video import VideoStream
from ultralytics import YOLO
>>>>>>> 2dcc45b4796a15a76bc9b29b4e4028c2c91bf4ab


class DetectHandler:
    def __init__(self, src, log_filename = "log.txt", img_foldername = "img", ObjectID = 0, model = "yolov8m.pt"):
        self.src = src  # 影像來源
        self.ObjectID = ObjectID    # 識別ID
<<<<<<< HEAD
        self.model = YOLO(model)     # 識別模組
=======
        self.model_path = model     # 識別模組路徑
        self.model = None     # 識別模組
>>>>>>> 2dcc45b4796a15a76bc9b29b4e4028c2c91bf4ab
        self.outputFrame = None     # 識別後的圖像
        self.lock = threading.Lock()    # 執行續的鎖
        self.is_inSide = False         # 來源影像內是否有識別到指定物件
        self.counter = Counter()    # 初始化計時器
        self.detect_thread = None   # 辨識執行續
        self.stopped = False  # 結束旗標

        self.TRIGER_COUNTER = 5  # 觸發inside狀態變換的連續幀數
        self.logger = Logger(log_filename)  # 建立log處理器
        self.imageHandler = ImageHandler(img_foldername)  # 建立image檔案處理器
        self.img_foldername = img_foldername
        self.frameCount = 0     # 效能控制，執行辨識的間隔偵數

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
<<<<<<< HEAD
=======

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
>>>>>>> 2dcc45b4796a15a76bc9b29b4e4028c2c91bf4ab
    
    # 回傳識別後的 frame
    def get_outputframe(self):
        return self.outputFrame
<<<<<<< HEAD
=======
    
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
>>>>>>> 2dcc45b4796a15a76bc9b29b4e4028c2c91bf4ab

    # 回傳字元格式的識別圖像
    def get_detected_image_in_byte(self):
        # 循環輸出流中的幀
        while True:
<<<<<<< HEAD
            # 獲取一個執行緒鎖，並在程式碼塊結束時自動釋放這個鎖
=======
            # 等待直到獲取鎖
>>>>>>> 2dcc45b4796a15a76bc9b29b4e4028c2c91bf4ab
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
<<<<<<< HEAD
        # 累計幀數
        accumulated_frames = 0
        # 計算FPS用
        start_time = time.time()

        fps = self.vs.stream.get(cv2.CAP_PROP_FPS)
        print(fps)

        # 循環視頻流中的幀
        while True:
            # 獲得影像幀
            ret, frame = self.vs.stream.read()
            # 成功獲得影像幀
            if ret:
                # 將當前時間標示在圖片上
                timestamp = datetime.datetime.now()
                cv2.putText(frame, # 圖像內容
                            timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), # 文字
                            (10, frame.shape[0] - 10), # 文字座標(左下角)
                            cv2.FONT_HERSHEY_SIMPLEX, # 字型
                            0.35, # 字體大小
                            (0, 0, 255), # 文字顏色
                            1)  # 字體粗細
                
                # 每過 frameCount 幀後才偵測一次
                if accumulated_frames > self.frameCount:
                    # 初始化累計幀數
                    accumulated_frames = 0
                    # 用YOLO模型進行物件偵測
                    result = self.model(frame, agnostic_nms=True,conf=0.5)[0]
                    detections = sv.Detections.from_yolov8(result)
                    detections = detections[detections.class_id == self.ObjectID]   # 只擷取指定類型的物件
=======
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
>>>>>>> 2dcc45b4796a15a76bc9b29b4e4028c2c91bf4ab

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
                    
                    # 如果偵測到的物件包含指定物件，則更新計數器的數值
                    object_detected = len(detections) > 0
                    self.counter[object_detected] += 1  # 當前狀態的次數+1
                    self.counter[not object_detected] = 0   # 初始化另一個狀態

<<<<<<< HEAD
                    # 如果符合觸發條件（例如連續五次都有(或沒有)偵測到貓咪），則更改 is_inSide 的值，並儲存當前幀的影像
                    if (not self.is_inSide and self.counter[True] >= self.TRIGER_COUNTER) or (self.is_inSide and self.counter[False] >= self.TRIGER_COUNTER):
                        # 狀態切換
                        self.is_inSide = not self.is_inSide
=======
            # 將當前時間標示在圖片上
            timestamp = datetime.datetime.now()
            cv2.putText(frame, timestamp.strftime(
                "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
            # 如果偵測到的物件包含指定物件，則更新計數器的數值
            object_detected = len(detections) > 0
            self.counter[object_detected] += 1  # 當前狀態的次數+1
            self.counter[not object_detected] = 0   # 初始化另一個狀態
>>>>>>> 2dcc45b4796a15a76bc9b29b4e4028c2c91bf4ab

                        # 獲取一個執行緒鎖，並在程式碼塊結束時自動釋放這個鎖
                        with self.lock:
                            # 偵測到指定物件時，載圖到img資料夾
                            if self.is_inSide :
                                # 儲存當前幀
                                self.imageHandler.saveimage(frame)
                            
                            # 寫入 log 檔
                            self.logger.save_log(detections=detections, is_inSide=self.is_inSide)

                # 顯示FPS
                if (time.time() - start_time) != 0:  # 實時顯示幀數
                    cv2.putText(frame, 
                                "FPS {0}".format(float('%.1f' % (1 / (time.time() - start_time)))), # FPS = 1 / (curret time - previous time)
                                (10, frame.shape[0] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                2)
                    
                # 幀數累加
                accumulated_frames += 1
                # 計算FPS用，紀錄本幀處理的時間
                start_time = time.time()

<<<<<<< HEAD
            # 獲取一個執行緒鎖，並在程式碼塊結束時自動釋放這個鎖
            with self.lock:
                self.outputFrame = frame.copy()
                
            # 結束程式
            if self.stopped:
                break
=======
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
>>>>>>> 2dcc45b4796a15a76bc9b29b4e4028c2c91bf4ab
