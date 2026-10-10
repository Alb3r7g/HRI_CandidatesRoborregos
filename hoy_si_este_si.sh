#!/bin/bash
set -e
cd "$(dirname "$0")"

FOXGLOVE_URL="https://app.foxglove.dev/?ds=foxglove-websocket&ds.url=ws%3A%2F%2Flocalhost%3A8765"
CAMERA_ARG=""
if [ -n "$1" ]; then
    CAMERA_ARG="camera_source:=http://$1:8080/video"
    echo "Cámara: http://$1:8080/video"
fi

docker compose up -d

if command -v ptyxis > /dev/null; then
    ptyxis --new-window -T "Conversación HRI" -x \
        "docker exec -it hri_dev bash -c 'source /ros2_ws/install/setup.bash && ros2 run hri_fusion conversation_view'" &
fi

(sleep 15 && xdg-open "$FOXGLOVE_URL" > /dev/null 2>&1) &

exec docker exec -it hri_dev bash -c \
    "source /ros2_ws/install/setup.bash && ros2 launch hri_bringup hoy_si_este_si.launch.py $CAMERA_ARG"
