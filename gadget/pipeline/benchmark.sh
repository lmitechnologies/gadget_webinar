# run the benchmark without TensorRT
python3 -m yolov5.benchmark_no_trt -w './build_engines/2022-07-15_512_512_600_s.pt'

# run the benchmark using TensorRT engines
python3 ./pipeline/pipeline_class.py