#!/bin/bash
# This allows access to the host's camera devices
docker run -it --rm \
    --device=/dev/video0:/dev/video0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    robot-race-tracker