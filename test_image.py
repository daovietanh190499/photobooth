import cv2
import numpy as np

img = cv2.imread('/home/gacontamnang/photobooth/gacon_75013d1c77aa4f0aaf6345236a3fcd85_20251207225559.jpg')

print(img.shape)

print((img.shape[0]/(img.shape[0] - 350))*img.shape[0])

new_img = np.ones((img.shape[0] + 350, int((img.shape[1]/img.shape[0]) * (img.shape[0] + 350)), 3), dtype=np.uint8)*255

# img[-150, :, :] = 0

# img[200, :, :] = 0

dx = int((new_img.shape[1] - img.shape[1]) / 2)
dy = int((new_img.shape[0] - img.shape[0]) / 2)

print(dx, dy)

new_img[dy:dy+img.shape[0], dx:dx+img.shape[1], :] = img

cv2.imwrite('/home/gacontamnang/photobooth/gacon_75013d1c77aa4f0aaf6345236a3fcd85_20251207225559_test.jpg', new_img)