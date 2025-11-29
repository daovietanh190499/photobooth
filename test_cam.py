import os
import subprocess
import cv2
from threading import Thread

PIPE = "/tmp/cam_pipe.mjpg"

def run_gphoto2(pipe_path):
    gphoto_cmd = [
        "gphoto2",
        "--set-config", "viewfinder=1", 
        "--stdout",
        "--capture-movie"
    ]
    
    with open(pipe_path, "wb") as pipe:
        proc = subprocess.Popen(gphoto_cmd, stdout=pipe)
        proc.wait()

# Tạo pipe nếu chưa có
if not os.path.exists(PIPE):
    os.mkfifo(PIPE)

# Chạy gphoto2 trong thread riêng
gphoto_thread = Thread(target=run_gphoto2, args=(PIPE,))
gphoto_thread.daemon = True  # Thread sẽ tự tắt khi chương trình chính kết thúc
gphoto_thread.start()

# Đợi một chút để gphoto2 khởi động
import time
time.sleep(1)

# Mở pipe bằng OpenCV
cap = cv2.VideoCapture(PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame, đợi camera...")
        continue
        
    cv2.imshow("DSLR LiveView", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
