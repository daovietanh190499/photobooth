import os
import cv2
import yaml
import io
from process import Processor
from datetime import datetime
from pathlib import Path

from utils import load_config, get_paths, list_wifi, connect_wifi

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import subprocess
from typing import List
import uuid

import numpy as np
import qrcode

from threading import Thread
import time

CONFIG_PATH = Path("config.yml")

# Create a FastAPI instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="./templates/static"), name="static")

# Define a root path operation
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# Define a root path operation
@app.get("/test", response_class=HTMLResponse)
async def read_root():
    with open("templates/test.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/collage", response_class=HTMLResponse)
async def read_collage():
    with open("templates/collage.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

class ConfigModel(BaseModel):
    operating_system: str
    domain: str
    topic: str
    collection_name: str
    folders: dict
    printer: dict
    camera: dict
    commands: dict
    wifi: dict

@app.get("/config")
def get_config():
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="Config file not found")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data

@app.post("/config")
def update_config(new_config: ConfigModel):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(new_config.dict(), f, allow_unicode=True, sort_keys=False)
    return {"message": "Configuration updated successfully"}

@app.get("/stat")
async def stat():
    config = load_config(CONFIG_PATH)
    stat_printer_command = config.get("commands").get("printer").get("stat_printer_command")
    stat_cam_command = config.get("commands").get("camera").get("stat_cam_command")
    printer_code = config.get("printer").get("code")
    camera_code = config.get("camera").get("code")
    
    printer_stat = False
    cam_stat = False

    result = subprocess.run(
        [command.replace("$printer_code", printer_code) for command in stat_printer_command],
        capture_output=True,
        text=True
    )

    printers = result.stdout.strip().splitlines()
    for line in printers:
        if printer_code.lower() in line.lower():
            printer_stat = True

    result = subprocess.run(
        [command.replace("$camera_code", camera_code) for command in stat_cam_command],
        capture_output=True,
        text=True
    )

    cameras = result.stdout.strip().splitlines()
    for line in cameras:
        if camera_code.lower() in line.lower():
            cam_stat = True

    return {
        "message": "success",
        "data": {
            "camera_stat": "OK" if cam_stat else "Not Connected",
            "printer_stat": "OK" if printer_stat else "Not Connected"
        }
    }


PIPE = "/tmp/cam_pipe.mjpg"

from queue import Queue
import signal
from threading import Thread, Lock

class CameraStream:
    def __init__(self):
        self.pipe_path = PIPE
        self.process = None
        self.cap = None
        self.is_running = False
        self.counting = 0

    def start_gphoto2(self):
        if not os.path.exists(self.pipe_path):
            os.mkfifo(self.pipe_path)
        
        # def run_gphoto2(pipe_path):
        gphoto_cmd = [
            "bash", "running-stream.sh"
        ]
        
        self.process = subprocess.Popen(gphoto_cmd)

        time.sleep(1)  # Wait for gphoto2 to start
        
        self.cap = cv2.VideoCapture(self.pipe_path)
        self.is_running = True

    def stop_gphoto2(self):
        kill_gphoto_cmd = ["pkill", "-f", "gphoto2"]
        subprocess.run(kill_gphoto_cmd, check=True)
        self.is_running = False
        if self.cap:
            self.cap.release()

    def generate_frames(self):
        while True:
            if self.is_running and not (self.cap is None):
                ret, frame = self.cap.read()
            else:
                ret = None
            if not ret:
                if self.is_running:
                    print("Không đọc được frame, đợi camera...")
                    # self.counting += 1
                    # if self.counting == 10:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.pipe_path)
                    # self.counting = 0
                else:
                    break
            try:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                # Yield MJPEG format
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                continue

camera = CameraStream()

@app.on_event("shutdown")
async def shutdown_event():
    camera.stop_gphoto2()

@app.get("/video-stream")
async def video_stream():
    if  not camera.is_running:
        camera.start_gphoto2()

    return StreamingResponse(
        camera.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/video-start")
async def video_start():
    if not camera.is_running:
        camera.start_gphoto2()
    return {"message": "Camera stream started"}    

@app.get("/video-stop")
async def video_stop():
    if camera.is_running:
        camera.stop_gphoto2()
    return {"message": "Camera stream stopped"}

@app.post("/capture")
async def capture(file: UploadFile = File(...)):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    
    config = load_config(CONFIG_PATH)
    control_camera_command = config.get("commands").get("camera").get("control_cam_command")
    topic = config.get("topic")
    collection_name = config.get("collection_name")
    _, _, save_folder, _, _ = get_paths(config)
    
    with open(f"{save_folder}{collection_name}_{timestamp}.jpg", "wb") as f:
        f.write(await file.read())
        
    if camera.is_running:
        camera.stop_gphoto2()

    try:
        subprocess.run(
            [
                "bash", "running-capture.sh",
                f"{config.get("folders").get("work_folder")}/data/{topic}/images/{collection_name}_{timestamp}.jpg"
            ],
            capture_output=True,
            timeout=5,
            text=True,
            check=True
        )
    except subprocess.TimeoutExpired:
        print("Capture Timeout, Fallback to webcam")

    return {"image_path": f"{collection_name}_{timestamp}.jpg"}

class ProcessInfo(BaseModel):
    images: List[str] | None
    filters: List[str] = []
    layout_id: str | None

@app.post("/process")
async def process(process_info: ProcessInfo):
    if not process_info.images:
        raise HTTPException(status_code=400, detail="No image paths provided.")

    if not process_info.layout_id:
        raise HTTPException(status_code=400, detail="No frame id provided.")

    config = load_config(CONFIG_PATH)
    
    topic = config.get("topic")
    collection_name = config.get("collection_name")
    domain = config.get("domain")
    
    root_folder, topic_folder, save_folder, processed_folder, frame_folder = get_paths(config)
    
    processor = Processor(
        save_folder=save_folder,
        processed_folder=processed_folder,
        frame_folder=frame_folder,
        collection_name=collection_name,
        topic=topic,
        domain=domain
    )

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")

    image_rgb = processor.process(
        process_info.layout_id,
        process_info.images,
        process_info.filters
    )

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    _, img_encoded = cv2.imencode('.jpg', image_bgr)

    return StreamingResponse(
        io.BytesIO(img_encoded.tobytes()),
        media_type="image/jpeg"
    )

@app.post("/final-process")
async def final_process(process_info: ProcessInfo):
    if not process_info.images:
        raise HTTPException(status_code=400, detail="No image paths provided.")

    if not process_info.layout_id:
        raise HTTPException(status_code=400, detail="No frame id provided.")

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")

    config = load_config(CONFIG_PATH)
    
    topic = config.get("topic")
    collection_name = config.get("collection_name")
    domain = config.get("domain")
    
    root_folder, topic_folder, save_folder, processed_folder, frame_folder = get_paths(config)

    processor = Processor(
        save_folder=save_folder,
        processed_folder=processed_folder,
        frame_folder=frame_folder,
        collection_name=collection_name,
        topic=topic,
        domain=domain
    )

    image_path, image_id = processor.final_process(
        process_info.layout_id,
        process_info.images,
        process_info.filters
    )

    return {"image_path": image_path, "image_id": image_id}

class PrintInfo(BaseModel):
    image_path: str | None

@app.post("/print")
async def print_image(print_info: PrintInfo):
    if not print_info.image_path:
        raise HTTPException(status_code=400, detail="No image path provided.")
    
    config = load_config(CONFIG_PATH)
    control_printer_command = config.get("commands").get("printer").get("control_printer_command")
    printer_name = config.get("printer").get("code")
    domain = config.get("domain")
    topic = config.get("topic")
    
    full_command = []
    
    image_path = config.get("folders").get("work_folder", "./") + print_info.image_path[2:]
    
    for command in control_printer_command:
        full_command.append(
            str(command).replace("$image_path", image_path).replace("$printer_code", printer_name)
        )

    # subprocess.run(" ".join(full_command))
    subprocess.run(full_command)
    
    qr_data = f"{domain}/{topic}/processed/{os.path.split(print_info.image_path)[-1]}"
    qr_img = qrcode.make(qr_data).convert("RGB")
    qr_img = np.array(qr_img)

    _, img_encoded = cv2.imencode('.jpg', qr_img)

    return StreamingResponse(
        io.BytesIO(img_encoded.tobytes()),
        media_type="image/jpeg"
    )

@app.get("/reboot")
async def reboot():
    
    config = load_config(CONFIG_PATH)
    reboot = config.get("commands").get("system").get("reboot_command")
    
    full_command = []
    for command in reboot:
        full_command.append(str(command))

    subprocess.run(full_command)

    return {"message": "Rebooting system..."}

@app.post("/frames/{topic_name}")
async def upload_frames(topic_name: str, file: UploadFile = File(...)):
    config = load_config(CONFIG_PATH)
    topic = config.get("topic")
    _, _, _, _, frame_folder = get_paths(config)
    
    if topic_name != topic:
        raise HTTPException(status_code=404, detail="Invalid topic")

    ext = os.path.splitext(file.filename)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(frame_folder, unique_name)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return JSONResponse({"status": "success", "filename": unique_name})

@app.get("/frames/{topic_name}")
async def list_frames(topic_name: str, query_param: str = None):   
    config = load_config(CONFIG_PATH)
    
    topic = config.get("topic")
    collection_name = config.get("collection_name")
    domain = config.get("domain")
    
    root_folder, topic_folder, save_folder, processed_folder, frame_folder = get_paths(config)
    
    processor = Processor(
        save_folder=save_folder,
        processed_folder=processed_folder,
        frame_folder=frame_folder,
        collection_name=collection_name,
        topic=topic,
        domain=domain
    )

    if topic_name != topic:
        raise HTTPException(status_code=404, detail="Page not found.")
    files = os.listdir(frame_folder)
    slots_list = []
    for file in files:
        slots, bbox, ratio = processor.count_photo_areas(file)
        slots_list.append((slots, ratio))
    return {"query_param": query_param, "files": [{"id": os.path.split(file)[-1], "slots": slots[0], "ratio": slots[1]} for file, slots in zip(files, slots_list)]}

@app.get("/frames/{topic_name}/{frame_id}")
async def retrieve_frames(topic_name: str, frame_id: str, query_param: str = None):
    config = load_config(CONFIG_PATH)
    topic = config.get("topic")
    _, _, _, _, frame_folder = get_paths(config)
    
    if topic_name != topic:
            raise HTTPException(status_code=404, detail="Page not found.")
    files = os.listdir(frame_folder)
    for file in files:
        if frame_id in file:
                return FileResponse(
                    path=f"{frame_folder}{file}",
                    media_type="image/jpeg"
                )
    return HTTPException(status_code=404, detail="Page not found.")

@app.delete("/frames/{topic_name}/{frame_id}")
async def delete_frame(topic_name: str, frame_id: str, query_param: str = None):
    config = load_config(CONFIG_PATH)
    topic = config.get("topic")
    _, _, _, _, frame_folder = get_paths(config)
    
    if topic_name != topic:
            raise HTTPException(status_code=404, detail="Page not found.")
    files = os.listdir(frame_folder)
    for file in files:
        if frame_id in file:
            os.remove(f"{frame_folder}{file}")
            return {"message": "Frame deleted successfully."}

    return HTTPException(status_code=404, detail="Page not found.")

@app.get("/images/{topic_name}")
async def list_image(topic_name: str, query_param: str = None):
    config = load_config(CONFIG_PATH)
    topic = config.get("topic")
    _, _, save_folder, _, _ = get_paths(config)

    if topic_name != topic:
        raise HTTPException(status_code=404, detail="Page not found.")
    files = os.listdir(save_folder)
    return {"query_param": query_param, "files": files}

@app.get("/images/{topic_name}/{image_id}")
async def read_image(topic_name: str, image_id: str, query_param: str = None):
    config = load_config(CONFIG_PATH)
    topic = config.get("topic")
    _, _, save_folder, _, _ = get_paths(config)
    
    if topic_name != topic:
            raise HTTPException(status_code=404, detail="Page not found.")
    files = os.listdir(save_folder)
    for file in files:
        if image_id in file:
            img = cv2.imread(os.path.join(save_folder, file))
            img = cv2.resize(img, None, fx=0.125, fy=0.125)
            _, img_encoded = cv2.imencode('.jpg', img)

            return StreamingResponse(
                io.BytesIO(img_encoded.tobytes()),
                media_type="image/jpeg"
            )

    return HTTPException(status_code=404, detail="Page not found.")

@app.get("/raw-images/{topic_name}/{image_id}")
async def raw_image(topic_name: str, image_id: str, query_param: str = None):
    config = load_config(CONFIG_PATH)
    topic = config.get("topic")
    _, _, save_folder, _, _ = get_paths(config)
    
    if topic_name != topic:
            raise HTTPException(status_code=404, detail="Page not found.")
    files = os.listdir(save_folder)
    for file in files:
        if image_id in file:
                return FileResponse(
                    path=f"{save_folder}{file}",
                    media_type="image/jpeg"
                )
    return HTTPException(status_code=404, detail="Page not found.")

@app.get("/processed-images/{topic_name}")
async def list_processed_image(topic_name: str, query_param: str = None):
    config = load_config(CONFIG_PATH)
    topic = config.get("topic")
    _, _, _, processed_folder, _ = get_paths(config)

    if topic_name != topic:
        raise HTTPException(status_code=404, detail="Page not found.")
    files = os.listdir(processed_folder)
    return {"query_param": query_param, "files": files}

@app.get("/processed-images/{topic_name}/{image_id}")
async def read_processed_image(topic_name: str, image_id: str, query_param: str = None):
    config = load_config(CONFIG_PATH)
    topic = config.get("topic")
    _, _, _, processed_folder, _ = get_paths(config)

    if topic_name != topic:
            raise HTTPException(status_code=404, detail="Page not found.")
    files = os.listdir(processed_folder)
    for file in files:
        if image_id in file:
                ext = file.split(".")[-1]
                if ext == "pdf":
                    return FileResponse(
                        path=f"{processed_folder}{file}",
                        media_type="application/pdf"
                    )
                else:
                    return FileResponse(
                        path=f"{processed_folder}{file}",
                        media_type="image/jpeg"
                    )
    return HTTPException(status_code=404, detail="Page not found.")

@app.get("/wifis")
async def list_wifi_api():   
    config = load_config(CONFIG_PATH)
    
    list_wifi_command = config.get("commands").get("system").get("list_wifi_command")
    
    wifis = list_wifi(cmd=list_wifi_command)

    return {"wifis": wifis}

@app.post("/connect-wifi")
async def connect_wifi_api():   
    config = load_config(CONFIG_PATH)
    
    connect_wifi_command = config.get("commands").get("system").get("connect_wifi_command")
    
    wifis = connect_wifi(cmd=connect_wifi_command,
                         ssid=config.get("wifi").get("ssid"),
                         password=config.get("wifi").get("password"),
                         iface=None)

    return {"message": "Connected to WiFi successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",  # or "127.0.0.1"
        port=8000,
    )
