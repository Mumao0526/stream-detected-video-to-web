import datetime
import threading
from ultralytics import YOLO

class Logger:
    def __init__(self, log_filename = "log.txt", model = "yolov8m.pt"):
        self.log_filename = log_filename    # LOG檔案名稱
        self.model = YOLO(model)     # 識別模組
        self.lock = threading.Lock()    # 執行續的鎖

        # 獲取全部的 log
    def get_Log(self):
        # 獲取一個執行緒鎖，並在程式碼塊結束時自動釋放這個鎖
        with self.lock:
            with open(self.log_filename, 'r') as file:
                log = file.read()
            return log
    
        # 獲取最後一項 log
    def get_final_log(self):
        logs = self.get_Log().split('\n')
        return logs[-2] if len(logs) > 1 else None
    
        # 寫入 log 檔
    def save_log(self, detections, is_inSide):
        # 使用 datetime 模組來格式化當前的時間，並作為檔名的一部分
        current_time = datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')
        # 獲取當前時間
        current_time = current_time.replace("_", "/")
        # 獲取一個執行緒鎖，並在程式碼塊結束時自動釋放這個鎖
        with self.lock:
            with open(self.log_filename, 'a') as file:
                if len(detections) > 0:
                    count = 0
                    # 從 detection 獲取資料
                    for _, confidence, class_id, _ in detections:
                        count += 1
                        # 獲取識別物件ID的名稱
                        class_name = self.model.model.names[class_id]
                        # 寫入 log.txt
                        file.write(f"{current_time}:{class_name} {confidence:0.2f} {is_inSide} -From number{count}\n")
                else:
                    file.write(f"{current_time}:No detection {is_inSide}\n")