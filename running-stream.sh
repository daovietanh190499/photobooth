pkill -f gphoto2
gphoto2 --set-config viewfinder=1 --stdout --capture-movie > /tmp/cam_pipe.mjpg &