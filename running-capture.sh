#!/bin/bash

pkill -f gphoto2
gphoto2 --capture-image-and-download --force-overwrite --filename "$1"
# bash /home/gacontamnang/photobooth/running-stream.sh