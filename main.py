import cv2
from detectMotion import detectMotion
from flask import Response
from flask import Flask
from flask import render_template

# initialize a flask object
app = Flask(__name__)

# 設定影片來源
src = 0  # 使用第一個本機裝置的攝影機
ObjectID = 0  # 檢測的物件ID為0
model_path = "yolov8l.pt"  # 使用 yolov8l.pt 模型

# 建立物件並開始偵測
detector = detectMotion(src, ObjectID, model_path)

# 設定網站來源
@app.route("/")
def index():
	return render_template("main.html")

# 設定video_feed資料
@app.route("/video_feed")
def video_feed():
    return Response(detector.get_detected_image_in_byte(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

def main():

    detector.start()

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
        if key == 27:
            break
    '''
    
    app.run(debug=True,
    threaded=True, use_reloader=False)

    # 停止偵測並關閉視窗
    detector.stop()
    cv2.destroyAllWindows()






if __name__ == "__main__":
    main()
