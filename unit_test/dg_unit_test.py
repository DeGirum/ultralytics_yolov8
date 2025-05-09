# Script to run benchmarks for DeGirum's Ultralytics repo; useful to test a local build of Ultralytics

import subprocess

if __name__ == '__main__':
    tests = [
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/coco8.yaml", "model=yolo11n.pt", "imgsz=160", "export_hw_optimized=True", "verbose=0.306", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/coco8-seg.yaml", "model=yolo11n-seg.pt", "imgsz=160", "export_hw_optimized=True", "verbose=0.247", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/imagenet10.yaml", "model=yolo11n-cls.pt", "imgsz=160", "export_hw_optimized=True", "verbose=0.166", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/coco8-pose.yaml", "model=yolo11n-pose.pt", "imgsz=160", "export_hw_optimized=True", "verbose=0.197", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/coco8.yaml", "model=yolo11n.pt", "imgsz=160", "verbose=0.308", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/imagenet10.yaml", "model=yolo11n-cls.pt", "imgsz=160", "verbose=0.249", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/coco8.yaml", "model=yolov8s-worldv2.pt", "imgsz=160", "verbose=0.337", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/coco8-seg.yaml", "model=yolo11n-seg.pt", "imgsz=160", "verbose=0.247", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/coco8-pose.yaml", "model=yolo11n-pose.pt", "imgsz=160", "verbose=0.197", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/dota8.yaml", "model=yolo11n-obb.pt", "imgsz=160", "verbose=0.597", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/coco8.yaml", "model=yolov10n.pt", "imgsz=160", "verbose=0.205", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/coco8.yaml", "model=yolo11n.pt", "imgsz=160", "separate_outputs=True", "verbose=0.306", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/coco8-seg.yaml", "model=yolo11n-seg.pt", "imgsz=160", "separate_outputs=True", "verbose=0.247", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/imagenet10.yaml", "model=yolo11n-cls.pt", "imgsz=160", "separate_outputs=True", "verbose=0.166", "device=cpu"],
        ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/coco8-pose.yaml", "model=yolo11n-pose.pt", "imgsz=160", "separate_outputs=True", "verbose=0.185", "device=cpu"],
        # ["yolo", "benchmark", "format=openvino", "data=ultralytics/cfg/datasets/imdb10-age.yaml", "model=../yolo_zoo_checkpoints/age_regression/8n_relu6/yolov8n_relu6_age.pt", "imgsz=256", "verbose=4.2", "device=cpu"],

        ["yolo", "benchmark", "int8=True", "format=openvino", "data=ultralytics/cfg/datasets/coco8.yaml", "model=yolo11n.pt", "imgsz=160", "export_hw_optimized=True", "verbose=0.301", "device=cpu"],
        ["yolo", "benchmark", "int8=True", "format=openvino", "data=ultralytics/cfg/datasets/coco8-seg.yaml", "model=yolo11n-seg.pt", "imgsz=160", "export_hw_optimized=True", "verbose=0.192", "device=cpu"],
        ["yolo", "benchmark", "int8=True", "format=openvino", "data=ultralytics/cfg/datasets/coco8-pose.yaml", "model=yolo11n-pose.pt", "imgsz=160", "export_hw_optimized=True", "verbose=0.203", "device=cpu"],
        ["yolo", "benchmark", "int8=True", "format=openvino", "data=ultralytics/cfg/datasets/coco8.yaml", "model=yolo11n.pt", "imgsz=160", "verbose=0.301", "device=cpu"],
        ["yolo", "benchmark", "int8=True", "format=openvino", "data=ultralytics/cfg/datasets/coco8.yaml", "model=yolov8s-worldv2.pt", "imgsz=160", "verbose=0.337", "device=cpu"],
        ["yolo", "benchmark", "int8=True", "format=openvino", "data=ultralytics/cfg/datasets/coco8-seg.yaml", "model=yolo11n-seg.pt", "imgsz=160", "verbose=0.184", "device=cpu"],
        ["yolo", "benchmark", "int8=True", "format=openvino", "data=ultralytics/cfg/datasets/coco8-pose.yaml", "model=yolo11n-pose.pt", "imgsz=160", "verbose=0.203", "device=cpu"],
        ["yolo", "benchmark", "int8=True", "format=openvino", "data=ultralytics/cfg/datasets/dota8.yaml", "model=yolo11n-obb.pt", "imgsz=160", "verbose=0.547", "device=cpu"],
        ["yolo", "benchmark", "int8=True", "format=openvino", "data=ultralytics/cfg/datasets/coco8.yaml", "model=yolov10n.pt", "imgsz=160", "verbose=0.202", "device=cpu"],
        ["yolo", "benchmark", "int8=True", "format=openvino", "data=ultralytics/cfg/datasets/coco8.yaml", "model=yolo11n.pt", "imgsz=160", "separate_outputs=True", "verbose=0.301", "device=cpu"],
        ["yolo", "benchmark", "int8=True", "format=openvino", "data=ultralytics/cfg/datasets/coco8-seg.yaml", "model=yolo11n-seg.pt", "imgsz=160", "separate_outputs=True", "verbose=0.184", "device=cpu"],
        ["yolo", "benchmark", "int8=True", "format=openvino", "data=ultralytics/cfg/datasets/coco8-pose.yaml", "model=yolo11n-pose.pt", "imgsz=160", "separate_outputs=True", "verbose=0.203", "device=cpu"],
    ]

    for test in tests:
        subprocess.run(test)