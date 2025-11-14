import os
import cv2
import numpy as np
from PIL import Image
import pilgram
import qrcode
import datetime

class Processor:
    def __init__(self, save_folder, processed_folder, frame_folder, collection_name, topic, domain):
        self.save_folder = save_folder
        self.processed_folder = processed_folder
        self.frame_folder = frame_folder
        self.collection_name = collection_name
        self.topic = topic
        self.domain = domain

    def _hex_to_rgb(self, hex_color):
        return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

    def _create_mask(self, image_rgb, hex_color, tolerance=30):
        """Trả về mask nhị phân các pixel gần màu hex_color"""
        rgb = np.array(self._hex_to_rgb(hex_color))
        lower = np.clip(rgb - tolerance, 0, 255)
        upper = np.clip(rgb + tolerance, 0, 255)
        mask = cv2.inRange(image_rgb, lower, upper)
        return mask

    def _apply_filter(self, image_path, filter_type):
        im = Image.open(os.path.join(self.save_folder, image_path)).convert("RGB")
        func = getattr(pilgram, filter_type, None)
        if func:
            im = func(im)
        return im

    def count_photo_areas(self, frame_id, color_code="#838600", tolerance=25, min_area=500):
        """
        Đếm số vùng ảnh trong frame dựa vào mã màu ảnh (#838600).
        Trả về số lượng vùng (slots) và danh sách bounding box.
        """
        frame_path = os.path.join(self.frame_folder, f"{frame_id}")
        frame = cv2.imread(frame_path)
        if frame is None:
            raise FileNotFoundError(f"Frame {frame_path} not found")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = self._create_mask(frame_rgb, color_code, tolerance=tolerance)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area]

        # Sắp xếp theo vị trí từ trên xuống dưới, trái sang phải
        boxes.sort(key=lambda b: (b[1], b[0]))

        # Giữ lại các vùng lớn nhất phía trên (tránh vùng nhỏ lẫn vùng QR)
        if boxes:
            # Lọc theo chiều cao top 60% của ảnh để lấy vùng trên
            height = frame.shape[0]
            top_limit = int(height * 0.75)
            top_boxes = [b for b in boxes if b[1] + b[3] / 2 < top_limit]
        else:
            top_boxes = []

        return len(top_boxes), top_boxes

    def process(self, frame_id, image_paths, filter_types):
        frame_path = os.path.join(self.frame_folder, f"{frame_id}")
        output_path = os.path.join(self.processed_folder, f"processed_{frame_id}.jpg")

        frame = cv2.imread(frame_path)
        if frame is None:
            raise FileNotFoundError(f"Frame {frame_path} not found")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mã màu vùng ảnh và QR
        photo_color = "#838600"
        qr_color = "#868686"

        # Mask các vùng tương ứng
        photo_mask = self._create_mask(frame_rgb, photo_color, tolerance=25)
        qr_mask = self._create_mask(frame_rgb, qr_color, tolerance=25)

        # Lấy bounding box từng vùng để resize ảnh phù hợp
        contours, _ = cv2.findContours(photo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        photo_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 500]
        photo_boxes.sort(key=lambda b: (b[1], b[0]))

        # Chèn ảnh theo thứ tự
        for i, box in enumerate(photo_boxes):
            if i >= len(image_paths):
                break
            x, y, w, h = box

            # Ảnh filter
            if len(filter_types) >= 1:
                filtered = self._apply_filter(image_paths[i], filter_types[i % len(filter_types)])
            else:
                filtered = Image.open(os.path.join(self.save_folder, image_paths[i])).convert("RGB")
            filtered = filtered.resize((w, h))
            filtered_np = np.array(filtered)

            # Vùng mask cục bộ
            local_mask = photo_mask[y:y+h, x:x+w]
            local_mask_bool = local_mask > 0

            # Chèn pixel tương ứng
            frame_rgb[y:y+h, x:x+w][local_mask_bool] = filtered_np[local_mask_bool]

        return frame_rgb

    def final_process(self, frame_id, image_paths, filter_types):
        now = datetime.datetime.now()
        image_index = now.strftime("%Y%m%d%H%M%S")
        frame_path = os.path.join(self.frame_folder, f"{frame_id}")
        output_path = os.path.join(self.processed_folder, f"{self.collection_name}_{frame_id.split(".")[0]}_{image_index}.jpg")

        frame = cv2.imread(frame_path)
        if frame is None:
            raise FileNotFoundError(f"Frame {frame_path} not found")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mã màu vùng ảnh và QR
        photo_color = "#838600"
        qr_color = "#868686"

        # Mask các vùng tương ứng
        photo_mask = self._create_mask(frame_rgb, photo_color, tolerance=25)
        qr_mask = self._create_mask(frame_rgb, qr_color, tolerance=25)

        # Lấy bounding box từng vùng để resize ảnh phù hợp
        contours, _ = cv2.findContours(photo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        photo_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 500]
        photo_boxes.sort(key=lambda b: (b[1], b[0]))

        # Chèn ảnh theo thứ tự
        for i, box in enumerate(photo_boxes):
            if i >= len(image_paths):
                break
            x, y, w, h = box

            # Ảnh filter
            if len(filter_types) >= 1:
                filtered = self._apply_filter(image_paths[i], filter_types[i % len(filter_types)])
            else:
                filtered = Image.open(os.path.join(self.save_folder, image_paths[i])).convert("RGB")
            filtered = filtered.resize((w, h))
            filtered_np = np.array(filtered)

            # Vùng mask cục bộ
            local_mask = photo_mask[y:y+h, x:x+w]
            local_mask_bool = local_mask > 0

            # Chèn pixel tương ứng
            frame_rgb[y:y+h, x:x+w][local_mask_bool] = filtered_np[local_mask_bool]

        # Tạo QR code
        qr_data = f"{self.domain}/{self.topic}/processed/{self.collection_name}_{frame_id.split(".")[0]}_{image_index}.jpg"
        qr_img = qrcode.make(qr_data).convert("RGB")
        qr_img = qr_img.resize((frame_rgb.shape[1], frame_rgb.shape[0]))
        qr_np = np.array(qr_img)

        # Chèn QR vào các vùng QR mask (trên toàn ảnh)
        frame_rgb[qr_mask > 0] = qr_np[qr_mask > 0]

        # Lưu ảnh
        os.makedirs(self.processed_folder, exist_ok=True)
        result_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        img = result_bgr

        h, w = img.shape[:2]
        aspect_wh = w / h
        aspect_hw = h / w

        # Standard aspect ratios
        ratio_4x6 = 6 / 4  # 1.5
        ratio_2x6 = 6 / 2  # 3.0

        def approx(a, b, tol=0.15):
            return abs(a - b) / b < tol

        print(f"Image size: {w}x{h} | w/h={aspect_wh:.2f}, h/w={aspect_hw:.2f}")

        # --- Case 1: Already ~4x6 or rotated 6x4 ---
        if approx(aspect_wh, ratio_4x6):
            print("≈ 4x6 ratio detected (portrait)")
            resized = cv2.resize(img, (int(h * ratio_4x6), h))

        elif approx(aspect_hw, ratio_4x6):
            print("≈ 4x6 ratio detected but rotated → rotating 90°")
            img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = img_rot.shape[:2]
            resized = cv2.resize(img_rot, (int(h * ratio_4x6), h))

        # --- Case 2: ~2x6 ratio or rotated 6x2 ---
        elif approx(aspect_wh, ratio_2x6):
            print("≈ 2x6 ratio detected → duplicating vertically")
            combined = np.concatenate([img, img], axis=0)
            h2, w2 = combined.shape[:2]
            resized = cv2.resize(combined, (int(h2 * ratio_4x6), h2))

        elif approx(aspect_hw, ratio_2x6):
            print("≈ 2x6 ratio detected but rotated → rotating 90° and duplicating")
            img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            combined = np.concatenate([img_rot, img_rot], axis=0)
            h2, w2 = combined.shape[:2]
            resized = cv2.resize(combined, (int(h2 * ratio_4x6), h2))

        # --- Fallback: force 4x6 ratio ---
        else:
            print("Unknown ratio → resizing to 4x6 by default")
            resized = cv2.resize(img, (int(h * ratio_4x6), h))

        cv2.imwrite(output_path, resized)

        return output_path,  f"{self.collection_name}_{frame_id.split(".")[0]}_{image_index}.jpg"
