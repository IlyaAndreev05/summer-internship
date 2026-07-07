#!/bin/sh
if [ "$MODE" = "vk" ]; then
    exec gpss-helper --mode vk
else
    exec tail -f /dev/null
fi
