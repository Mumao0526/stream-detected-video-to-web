import datetime
import os
import threading
import time
import imutils
from collections import Counter

import cv2
import supervision as sv
from imutils.video import VideoStream
from ultralytics import YOLO
from Logger import Logger
from ImageHandler import ImageHandler


class DetectHandler:
    def __init__(self, src, log_filename = "log.txt", img_foldername = "img", ObjectID = 0, model = "yolov8m.pt"):
        self.src = src  # 影像來源
        self.ObjectID = ObjectID    # 識別ID
        self.model = YOLO(model)     # 識別模組
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
    
    # 回傳識別後的 frame
    def get_outputframe(self):
        return self.outputFrame

    # 回傳字元格式的識別圖像
    def get_detected_image_in_byte(self):
        # 循環輸出流中的幀
        while True:
            # 獲取一個執行緒鎖，並在程式碼塊結束時自動釋放這個鎖
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

                    # 如果符合觸發條件（例如連續五次都有(或沒有)偵測到貓咪），則更改 is_inSide 的值，並儲存當前幀的影像
                    if (not self.is_inSide and self.counter[True] >= self.TRIGER_COUNTER) or (self.is_inSide and self.counter[False] >= self.TRIGER_COUNTER):
                        # 狀態切換
                        self.is_inSide = not self.is_inSide

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

            # 獲取一個執行緒鎖，並在程式碼塊結束時自動釋放這個鎖
            with self.lock:
                self.outputFrame = frame.copy()
                
            # 結束程式
            if self.stopped:
                break