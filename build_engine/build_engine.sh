# convert the weights file
if [ ! -f /app/2022-07-15_512_512_600_s.wts ]; then
    python3 /repos/yolov5/gen_wts.py -w /app/2022-07-15_512_512_600_s.pt -o /app/2022-07-15_512_512_600_s.wts
fi

# generate engine
mkdir -p /app/build && cd /app/build && cmake /repos/tensorrtx/yolov5 && make
if [ ! -f /app/yolov5_512x512_s.engine ]; then
    /app/build/yolov5 -c /app/2022-07-15.yaml -w /app/2022-07-15_512_512_600_s.wts -o /app/yolov5_512x512_s.engine
fi
