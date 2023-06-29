
import datetime
import os
import threading
import cv2


class ImageHandler:
    def __init__(self, img_foldername = "img"):
        self.img_foldername = img_foldername
        self.lock = threading.Lock()    # 執行續的鎖
    
    def saveimage(self, frame):
        # 使用 datetime 模組來格式化當前的時間，並作為檔名的一部分
        current_time = datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')
        # 儲存成 png 格式
        filename = f'True_{current_time}.png'
        cv2.imwrite(os.path.join(self.img_foldername, filename), frame)
        # 維護 img 目錄中的照片數量
        self.maintain_img_limit()

    
    # 維護 img 目錄中的照片數量
    def maintain_img_limit(self, file_limit = 5):
        # 獲取一個執行緒鎖，並在程式碼塊結束時自動釋放這個鎖
        with self.lock:
            directory = os.path.join(self.img_foldername)
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
            while len(file_list) > file_limit :
                try:
                    os.remove(os.path.join(directory, file_list.pop(0)))
                except OSError:
                    print(f"Error: Deleting file in {directory}")
                    return

            return
    
    # 獲取最新的一張識別到的照片
    def get_final_img(self):
        while True:
            # 獲取一個執行緒鎖，並在程式碼塊結束時自動釋放這個鎖
            with self.lock:
                directory = os.path.join(self.img_foldername)
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