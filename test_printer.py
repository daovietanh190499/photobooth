from utils import load_config
import subprocess

CONFIG_PATH = "config.yml"

config = load_config(CONFIG_PATH)
control_printer_command = config.get("commands").get("printer").get("control_printer_command")
printer_name = config.get("printer").get("code")
domain = config.get("domain")
topic = config.get("topic")

full_command = []
for command in control_printer_command:
    full_command.append(
        str(command).replace("$image_path", "/home/gacontamnang/photobooth/DSC_0002.JPG") \
        .replace("$printer_code", printer_name)
    )
    
print(" ".join(full_command))

subprocess.run(full_command)