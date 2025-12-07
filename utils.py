import os
from pathlib import Path
import yaml
import json
import subprocess

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

def run_cmd(cmd):
    """Run shell command and return stdout"""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(result.stderr.strip())
    return result.stdout.strip()


def list_wifi(cmd: list = ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list"]):
    """
    Return list of available WiFi networks (SSID + signal + security)
    """
    output = run_cmd(" ".join(cmd))

    wifi_list = []
    for line in output.split("\n"):
        parts = line.split(":")
        if len(parts) >= 3:
            ssid, signal, security = parts[0], parts[1], parts[2]
            if ssid.strip():  # bỏ mạng ẩn
                wifi_list.append({
                    "ssid": ssid,
                    "signal": int(signal) if signal.isdigit() else 0,
                    "security": security
                })
    return wifi_list


def connect_wifi(cmd: list = ["nmcli", "device", "wifi", "connect", "$wifi_ssid", "password", "$wifi_password", "ifname", "$wifi_iface"],
                 ssid: str = "", 
                 password: str = "", 
                 iface: str = None):
    """
    Connect to WiFi using SSID and password.
    iface = tên interface (wlan0), nếu không truyền thì để nmcli tự chọn
    """
    ssid = ssid.replace('"', '\\"')  # escape nếu có ký tự "
    
    if iface:
        cmd = cmd
    else:
        cmd = cmd[:-2]
        
    full_command = []
    for c in cmd:
        if iface:
           c = c.replace("$wifi_iface", iface)
        c = c.replace("$wifi_ssid", ssid)
        c = c.replace("$wifi_password", password)
        full_command.append(c)
        
    print("Connecting to WiFi with command:", " ".join(full_command))

    return run_cmd(" ".join(full_command))

if __name__ == "__main__":
    wifis = list_wifi()
    print(wifis)