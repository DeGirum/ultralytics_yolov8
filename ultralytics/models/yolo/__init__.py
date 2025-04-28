# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo import classify, multi_label_classify, detect, obb, pose, segment, world
from ultralytics.models.yolo import regress

from .model import YOLO, YOLOWorld

__all__ = "classify", "multi_label_classify","segment", "detect", "pose", "obb", "world", "YOLO", "YOLOWorld"
__all__ += "regress",
