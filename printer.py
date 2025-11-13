import subprocess

class Printer():
    def __init__(self,
        print_command = [
            "rundll32",
            "shimgvw.dll,ImageView_PrintTo",
            "/pt",
            "$image_path"
        ]
    ):
        self.window_command = print_command

    def window_print(self, image_path=""):
        full_command = []
        for command in self.window_command:
            full_command.append(command.replace("$image_path", image_path))
        print(full_command)
        subprocess.run(self.window_command, shell=True)
