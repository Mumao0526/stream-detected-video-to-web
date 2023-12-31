import os
import cv2
import threading
import time
from DetectHandler import DetectHandler
from flask import Response
from flask import Flask
from flask import jsonify
from flask import render_template
from Logger import Logger
from ImageHandler import ImageHandler

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# initialize a flask object
app = Flask(__name__)

src = 0  # 使用第一個本機裝置的攝影機
src2 = "http://192.168.0.201:81/stream"  # 使用ESP32-CAM的攝影機
ObjectID = 0  # 檢測的物件ID為0
model_path = "yolov8m.pt"  # 使用 yolov8l.pt 模型

# 建立物件
detector = DetectHandler(src = src, ObjectID=ObjectID, model=model_path)
logger = Logger("log.txt")
imageHandler = ImageHandler("img")

# 設定網站來源
@app.route("/")
def index():
	return render_template("main.html")

# 設定video_feed資料
@app.route("/video_feed")
def video_feed():
    return Response(detector.get_detected_image_in_byte(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

# 傳送最後一張記錄照片
@app.route("/image_feed")
def image_feed():
    return Response(imageHandler.get_final_img(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

# 更新log
@app.route('/get_log')
def get_log():
    log_message = logger.get_final_log()
    return jsonify(log=log_message)



def main():
    # 開始偵測
    detector.start()
    app.run(debug=True,threaded=True, use_reloader=False)

    '''
    # TEST

    while True:
        # 獲取最新的影像
        frame = detector.get_outputframe()
        if frame is not None:
            # 在主視窗顯示影像
            cv2.imshow("Frame", frame)

        # 如果按下ESC鍵則離開迴圈
        key = cv2.waitKey(1)
        if key == 27 :
            break
        elif key == ord("s"):
            print_thread = threading.Thread(target=print_message)
            print_thread.start()
            print_thread.join()
        elif key == ord("i"):
            show_thread = threading.Thread(target=show_img)
            show_thread.start()
    '''
    # 停止偵測並關閉視窗
    detector.stop()
    cv2.destroyAllWindows()
    
def show_img():
    img = cv2.imdecode(imageHandler.get_final_img(), cv2.IMREAD_COLOR)
    cv2.imshow("img", img)

if __name__ == "__main__":
    main()