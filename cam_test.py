import cv2

# Mở camera mặc định (0), nếu bạn có nhiều camera, thử đổi sang 1 hoặc 2
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

while True:
    ret, frame = cap.read()
    print(frame.shape)
    if not ret:
        print("Không thể đọc frame")
        break

    # Hiển thị khung hình
    cv2.imshow('Camera Test', frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
