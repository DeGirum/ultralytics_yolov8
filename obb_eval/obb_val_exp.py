import os
import degirum as dg, degirum_tools as dg_tools

from obb_evaluator import OBBModelEvaluator

model = dg.load_model(
    "yolov8n_dota_obb--1024x1024_quant_hailort_multidevice_1",
    "@cloud",
    "https://hub.degirum.com/degirum/hailo/",
    dg_tools.get_token()
)

model.output_confidence_threshold =  0.1
model.output_nms_threshold = 0.7
model.output_max_detections = 300
model.output_max_detections_per_class = 300
model.input_letterbox_fill_color = (114, 114, 114)

evaluator = OBBModelEvaluator(model=model, show_progress=True)

base_path = __file__.rsplit('/', maxsplit=1)[0]

mAP_results = evaluator.evaluate(
    image_folder_path=base_path + os.sep + "example",
    ground_truth_annotations_path=os.sep.join([base_path, "example", "anno.json"])
)

print(f"[mAP50:95] {mAP_results[0][0]}")
print(f"[mAP50]    {mAP_results[0][1]}")
print(f"[mRecall]  {mAP_results[0][2]}")
print(f"[mPrecis]  {mAP_results[0][3]}")

# How annotations are expected to be formatted:
# [ {'img': "image_name.jpg", 'cls': [1, 1], 'bbox': [[x, y, w, h, a], [x, y, w, h, a]]} ]
# Annotations are a list of dictionaries, one dictionary per image.
#
# Each dictionary contains:
# - 'img': The image file name (string).
# - 'cls': A list of class IDs (integers) corresponding to each bounding box.
# - 'bbox': A list of bounding boxes, each defined by [center_x, center_y, width, height, rotation].
# 
# You can find an example annotation file in the "example" folder.
#
# If you have your annotations in x1, y1, x2, y2, x3, y3, x4, y4 format,
# you can convert them to the required format using a utility function in ultralytics.
#
# from ultralytics.utils.ops import xyxyxyxy2xywhr
# ann_to_convert = np.array([[x1, y1, x2, y2, x3, y3, x4, y4]])
# converted_ann = xyxyxyxy2xywhr(ann_to_convert)