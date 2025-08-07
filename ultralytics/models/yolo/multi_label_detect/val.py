# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import numpy as np

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.postprocess_utils import decode_bbox, separate_outputs_decode
from ultralytics.utils.metrics import MultiLabelDetectMetrics, box_iou


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
        self.topk = self.args.topk
        self.metrics = MultiLabelDetectMetrics(save_dir=self.save_dir)

    def preprocess(self, batch):
        """Preprocess batch by converting multi-label classification label data to float and moving it to the device."""
        batch = super().preprocess(batch)
        batch["mlb"] = batch["mlb"].to(self.device).float()
        return batch

    def get_desc(self):
        """Return description of evaluation metrics in string format."""
        self.topk = min(self.topk, min(self.dataloader.dataset.data.get("nc_per_label")))
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "MnAcc50-95",
            "SeqAcc50-95",
            "T1Acc50-95",
            f"T{self.topk}Acc50-95"
        )

    def postprocess(self, preds, img_shape):
        """Apply non-maximum suppression and return detections with high confidence scores."""
        if self.separate_outputs:  # Quant friendly export with separated outputs
            pred_order, mlb = separate_outputs_decode(preds, self.args.task, sum(self.nc_per_label))
            pred_decoded = decode_bbox(pred_order, img_shape, self.device)
            mlb = torch.permute(mlb, (0, 2, 1))
            mlb_softmax = torch.cat([m.sigmoid() if m.shape[1] == 1 else m.softmax(1) for m in mlb.split(self.nc_per_label, dim=1)])
            preds = torch.cat([pred_decoded, mlb_softmax], 1)
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.nc,
        )

    def init_metrics(self, model):
        """
        Initialize evaluation metrics for YOLO multi-label detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        self.metrics.topk = self.topk
        self.nc_per_label = model.nc_per_label if hasattr(model, "nc_per_label") else model.model.nc_per_label
        self.mlb_stats = dict(pred_mlb=[], target_mlb=[], mlb_matches=[])

    def get_stats(self):
        """
        Calculate and return metrics statistics.

        Returns:
            (dict): Dictionary containing metrics results.
        """
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        mlb_stats = {k: ([torch.cat([l[i] for l in v]).cpu() for i in range(len(v[0]))] if k == "mlb_matches" else torch.cat(v, 0).cpu()) for k, v in self.mlb_stats.items()}  # to numpy
        if len(stats):
            self.metrics.process(**stats, **mlb_stats, nc_per_label=self.nc_per_label, on_plot=self.on_plot)
        return self.metrics.results_dict

    def _prepare_batch(self, si, batch):
        """
        Prepare a batch for processing by converting keypoints to float and scaling to original dimensions.

        Args:
            si (int): Batch index.
            batch (dict): Dictionary containing batch data with keys like 'keypoints', 'batch_idx', etc.

        Returns:
            pbatch (dict): Prepared batch with keypoints scaled to original image dimensions.

        Notes:
            This method extends the parent class's _prepare_batch method by adding keypoint processing.
            Keypoints are scaled from normalized coordinates to original image dimensions.
        """
        pbatch = super()._prepare_batch(si, batch)
        pbatch["mlb"] = batch["mlb"][batch["batch_idx"] == si]
        return pbatch

    def update_metrics(self, preds, batch):
        """
        Update metrics with new predictions and ground truth data.

        This method processes each prediction, compares it with ground truth, and updates various statistics
        for performance evaluation.

        Args:
            preds (List[torch.Tensor]): List of prediction tensors from the model.
            batch (dict): Batch data containing images and ground truth annotations.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            mlb_stat = dict(
                mlb_matches=[torch.zeros(0, 2, dtype=int, device=self.device)]*len(self.iouv)
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            mlb_stat["target_mlb"] = pbatch.pop("mlb")
            
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]
            mlb_stat["pred_mlb"] = predn[:, 6:]

            # Evaluate
            if nl:
                stat["tp"], mlb_stat["mlb_matches"] = self._process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            for k in self.mlb_stats.keys():
                self.mlb_stats[k].append(mlb_stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            # if self.args.save_txt:
            #     self.save_one_txt(
            #         predn,
            #         self.args.save_conf,
            #         pbatch["ori_shape"],
            #         self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
            #     )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """
        Match predictions to ground truth objects using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (N,).
            true_classes (torch.Tensor): Target class indices of shape (M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct_det = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        # list for holding 10 tensors of correct match pairs, each tensor corresponding to one of 10 IoU thresholds
        correct_matches = []
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct_det[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct_det[matches[:, 1].astype(int), i] = True
                correct_matches.append(torch.tensor(matches, dtype=torch.int, device=pred_classes.device))
        return torch.tensor(correct_det, dtype=torch.bool, device=pred_classes.device), correct_matches

    # def plot_val_samples(self, batch, ni):
    #     """
    #     Plot and save validation set samples with ground truth bounding boxes and keypoints.

    #     Args:
    #         batch (dict): Dictionary containing batch data with keys:
    #             - img (torch.Tensor): Batch of images
    #             - batch_idx (torch.Tensor): Batch indices for each image
    #             - cls (torch.Tensor): Class labels
    #             - bboxes (torch.Tensor): Bounding box coordinates
    #             - keypoints (torch.Tensor): Keypoint coordinates
    #             - im_file (list): List of image file paths
    #         ni (int): Batch index used for naming the output file
    #     """
    #     plot_images(
    #         batch["img"],
    #         batch["batch_idx"],
    #         batch["cls"].squeeze(-1),
    #         batch["bboxes"],
    #         kpts=batch["keypoints"],
    #         paths=batch["im_file"],
    #         fname=self.save_dir / f"val_batch{ni}_labels.jpg",
    #         names=self.names,
    #         on_plot=self.on_plot,
    #     )

    # def plot_predictions(self, batch, preds, ni):
    #     """
    #     Plot and save model predictions with bounding boxes and keypoints.

    #     Args:
    #         batch (dict): Dictionary containing batch data including images, file paths, and other metadata.
    #         preds (List[torch.Tensor]): List of prediction tensors from the model, each containing bounding boxes,
    #             confidence scores, class predictions, and keypoints.
    #         ni (int): Batch index used for naming the output file.

    #     The function extracts keypoints from predictions, converts predictions to target format, and plots them
    #     on the input images. The resulting visualization is saved to the specified save directory.
    #     """
    #     pred_kpts = torch.cat([p[:, 6:].view(-1, *self.kpt_shape) for p in preds], 0)
    #     plot_images(
    #         batch["img"],
    #         *output_to_target(preds, max_det=self.args.max_det),
    #         kpts=pred_kpts,
    #         paths=batch["im_file"],
    #         fname=self.save_dir / f"val_batch{ni}_pred.jpg",
    #         names=self.names,
    #         on_plot=self.on_plot,
    #     )  # pred

    # def save_one_txt(self, predn, pred_kpts, save_conf, shape, file):
    #     """
    #     Save YOLO pose detections to a text file in normalized coordinates.

    #     Args:
    #         predn (torch.Tensor): Prediction boxes and scores with shape (N, 6) for (x1, y1, x2, y2, conf, cls).
    #         pred_kpts (torch.Tensor): Predicted keypoints with shape (N, K, D) where K is the number of keypoints
    #             and D is the dimension (typically 3 for x, y, visibility).
    #         save_conf (bool): Whether to save confidence scores.
    #         shape (tuple): Original image shape (height, width).
    #         file (Path): Output file path to save detections.

    #     Notes:
    #         The output format is: class_id x_center y_center width height confidence keypoints where keypoints are
    #         normalized (x, y, visibility) values for each point.
    #     """
    #     from ultralytics.engine.results import Results

    #     Results(
    #         np.zeros((shape[0], shape[1]), dtype=np.uint8),
    #         path=None,
    #         names=self.names,
    #         boxes=predn[:, :6],
    #         keypoints=pred_kpts,
    #     ).save_txt(file, save_conf=save_conf)

    # def pred_to_json(self, predn, filename):
    #     """
    #     Convert YOLO predictions to COCO JSON format.

    #     This method takes prediction tensors and a filename, converts the bounding boxes from YOLO format
    #     to COCO format, and appends the results to the internal JSON dictionary (self.jdict).

    #     Args:
    #         predn (torch.Tensor): Prediction tensor containing bounding boxes, confidence scores, class IDs,
    #             and keypoints, with shape (N, 6+K) where N is the number of predictions and K is the flattened
    #             keypoints dimension.
    #         filename (str | Path): Path to the image file for which predictions are being processed.

    #     Notes:
    #         The method extracts the image ID from the filename stem (either as an integer if numeric, or as a string),
    #         converts bounding boxes from xyxy to xywh format, and adjusts coordinates from center to top-left corner
    #         before saving to the JSON dictionary.
    #     """
    #     stem = Path(filename).stem
    #     image_id = (int(stem) if stem.isnumeric() else stem) if self.is_coco else os.path.basename(filename)
    #     box = ops.xyxy2xywh(predn[:, :4])  # xywh
    #     box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    #     for p, b in zip(predn.tolist(), box.tolist()):
    #         self.jdict.append(
    #             {
    #                 "image_id": image_id,
    #                 "category_id": self.class_map[int(p[5])],
    #                 "bbox": [round(x, 3) for x in b],
    #                 "keypoints": p[6:],
    #                 "score": round(p[4], 5),
    #             }
    #         )

    # def eval_json(self, stats):
    #     """Evaluates object detection model using COCO JSON format."""
    #     if self.args.save_json and (self.is_coco or self.args.anno_json) and len(self.jdict):
    #         anno_json = Path(self.args.anno_json) if self.args.anno_json else self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
    #         pred_json = self.save_dir / "predictions.json"  # predictions
    #         LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
    #         try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #             check_requirements("pycocotools>=2.0.6")
    #             from pycocotools.coco import COCO  # noqa
    #             from pycocotools.cocoeval import COCOeval  # noqa

    #             for x in anno_json, pred_json:
    #                 assert x.is_file(), f"{x} file not found"
    #             anno = COCO(str(anno_json))  # init annotations api
    #             pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
    #             for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "keypoints")]):
    #                 if self.is_coco:
    #                     eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
    #                 eval.evaluate()
    #                 eval.accumulate()
    #                 eval.summarize()
    #                 idx = i * 4 + 2
    #                 stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[
    #                     :2
    #                 ]  # update mAP50-95 and mAP50
    #         except Exception as e:
    #             LOGGER.warning(f"pycocotools unable to run: {e}")
    #     return stats
