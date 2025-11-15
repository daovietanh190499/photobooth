import os
from pathlib import Path
import yaml

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data

def get_paths(config):
    topic = config.get("topic")
    root_folder = config.get("folders").get("root_folder")
    topic_folder = os.path.join(root_folder, topic)
    save_folder = config.get("folders").get("save_folder").replace("$topic_folder", topic_folder)
    processed_folder = config.get("folders").get("processed_folder").replace("$topic_folder", topic_folder)
    frame_folder = config.get("folders").get("frame_folder").replace("$topic_folder", topic_folder)
    os.makedirs(topic_folder, exist_ok=True)
    os.makedirs(frame_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    return (root_folder, topic_folder, save_folder, processed_folder, frame_folder)