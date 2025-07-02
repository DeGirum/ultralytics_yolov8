# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops
from ultralytics.utils.postprocess_utils import decode_bbox, separate_outputs_decode


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
        nc_per_label = self.model.nc_per_label if hasattr(self.model, "nc_per_label") else self.model.model.nc_per_label

        if self.separate_outputs:  # Quant friendly export with separated outputs
            pred_order, mlb = separate_outputs_decode(preds, self.args.task, sum(nc_per_label))
            pred_decoded = decode_bbox(pred_order, img.shape, self.device)
            mlb = torch.permute(mlb, (0, 2, 1))
            mlb_softmax = torch.cat([m.sigmoid() if m.shape[1] == 1 else m.softmax(1) for m in mlb.split(nc_per_label, dim=1)])
            preds = torch.cat([pred_decoded, mlb_softmax], 1)

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
