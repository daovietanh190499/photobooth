import subprocess
import json


def run_cmd(cmd):
    """Run shell command and return stdout"""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(result.stderr.strip())
    return result.stdout.strip()


def list_wifi():
    """
    Return list of available WiFi networks (SSID + signal + security)
    """
    cmd = "nmcli -t -f SSID,SIGNAL,SECURITY device wifi list"
    output = run_cmd(cmd)

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


def connect_wifi(ssid: str, password: str, iface: str = None):
    """
    Connect to WiFi using SSID and password.
    iface = tên interface (wlan0), nếu không truyền thì để nmcli tự chọn
    """
    ssid = ssid.replace('"', '\\"')  # escape nếu có ký tự "
    
    if iface:
        cmd = f'nmcli device wifi connect "{ssid}" password "{password}" ifname {iface}'
    else:
        cmd = f'nmcli device wifi connect "{ssid}" password "{password}"'

    return run_cmd(cmd)

wifis = list_wifi()
print(wifis)