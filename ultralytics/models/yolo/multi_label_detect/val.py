# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import itertools
import math

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.postprocess_utils import decode_bbox


class MultiLabelDetectionValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a detection model.

    This class implements validation functionality specific to multi-label object detection tasks,
    including metrics calculation, prediction processing, and visualization of results.

    Attributes:
        nt_per_class (np.ndarray): Number of targets per class.
        nt_per_image (np.ndarray): Number of targets per image.
        is_coco (bool): Whether the dataset is COCO.
        is_lvis (bool): Whether the dataset is LVIS.
        class_map (list): Mapping from model class indices to dataset class indices.
        metrics (DetMetrics): Object detection metrics calculator.
        iouv (torch.Tensor): IoU thresholds for mAP calculation.
        niou (int): Number of IoU thresholds.
        lb (list): List for storing ground truth labels for hybrid saving.
        jdict (list): List for storing JSON detection results.
        stats (dict): Dictionary for storing statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.multi_label_detect import MultiLabelDetectionValidator
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml")
        >>> validator = MultiLabelDetectionValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize multi-label detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (Any, optional): Progress bar for displaying progress.
            args (dict, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "multi_label_detect"
    
    def init_metrics(self, model):
        """
        Initialize evaluation metrics for YOLO multi-label detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        self.nc_per_label = model.nc_per_label if hasattr(model, "nc_per_label") else model.model.nc_per_label

    def postprocess(self, preds, img_shape):
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (List[torch.Tensor]): Processed predictions after NMS.
        """
        if self.separate_outputs:  # Quant friendly export with separated outputs
            preds = decode_bbox(preds, img_shape, self.device)
        
        preds = preds[0].permute(0, 2, 1)
        box, prob = preds.split((4, self.nc), dim=-1)
        joint_prob = torch.empty((*(prob.shape[:-1]), math.prod(self.nc_per_label))).to(box.device)

        def flat_index(indices, strides):
            return sum(i * s for i, s in zip(indices, strides))

        idx_strides = [1]
        idx_offsets = [0]
        for b in self.nc_per_label[:-1]:
            idx_strides.append(idx_strides[-1] * b)
            idx_offsets.append(idx_offsets[-1] + b)

        for combo in itertools.product(*[range(x) for x in reversed(self.nc_per_label)]):
            per_label_idx = tuple(reversed(combo))
            flat_idx = flat_index(per_label_idx, idx_strides)
            joint_prob[:, :, flat_idx:flat_idx + 1] = torch.prod(prob[:, :, [i + o for i, o in zip(per_label_idx, idx_offsets)]], dim=-1, keepdim=True)
        
        preds = torch.cat([box, joint_prob], dim=-1)

        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
        )
