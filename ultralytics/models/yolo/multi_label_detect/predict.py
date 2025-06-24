# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import itertools
import math

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops
from ultralytics.utils.postprocess_utils import decode_bbox


class MultiLabelDetectionPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a detection model.

    This predictor specializes in multi-label object detection tasks, processing model outputs into
    meaningful detection results with bounding boxes and class predictions.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The detection model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        postprocess: Process raw model predictions into detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import MultiLabelDetectionPredictor
        >>> args = dict(model="yolo11n.pt", source=ASSETS)
        >>> predictor = MultiLabelDetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = MultiLabelDetectionPredictor(overrides=dict(model="yolo11n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "save_feats", False)
        if self.separate_outputs:  # Quant friendly export with separated outputs
            preds = decode_bbox(preds, img.shape, self.device)

        nc_per_label = self.model.nc_per_label if hasattr(self.model, "nc_per_label") else self.model.model.nc_per_label
        nc = sum(nc_per_label)
        preds = preds[0].permute(0, 2, 1)
        box, prob = preds.split((4, nc), dim=-1)
        joint_prob = torch.empty((*(prob.shape[:-1]), math.prod(nc_per_label))).to(box.device)

        def flat_index(indices, strides):
            return sum(i * s for i, s in zip(indices, strides))

        idx_strides = [1]
        idx_offsets = [0]
        for b in nc_per_label[:-1]:
            idx_strides.append(idx_strides[-1] * b)
            idx_offsets.append(idx_offsets[-1] + b)

        for combo in itertools.product(*[range(x) for x in reversed(nc_per_label)]):
            per_label_idx = tuple(reversed(combo))
            flat_idx = flat_index(per_label_idx, idx_strides)
            joint_prob[:, :, flat_idx:flat_idx + 1] = torch.prod(prob[:, :, [i + o for i, o in zip(per_label_idx, idx_offsets)]], dim=-1, keepdim=True)
        
        preds = torch.cat([box, joint_prob], dim=-1)

        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results
