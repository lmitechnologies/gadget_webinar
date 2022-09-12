printf '\nrunning without TensorRT\n'
python3 /repos/yolov5/benchmark_no_trt.py -w /app/2022-07-15_512_512_600_s.pt --imsz 512,512
printf '\nusing TensorRT\n'
python3 /repos/yolov5/benchmark_trt.py -e /app/yolov5_arm_512x512_s.engine -p /app/build
