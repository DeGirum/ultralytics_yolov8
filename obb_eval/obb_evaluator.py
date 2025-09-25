#
# obb_eval.py: obb models evaluator
#
# Copyright DeGirum Corporation 2024
# All rights reserved
#

import json, os, degirum as dg, numpy as np
from typing import List, Optional

from degirum_tools.math_support import xyxy2xywh
from degirum_tools.eval_support import ModelEvaluatorBase
from degirum_tools.ui_support import Progress

# from .math_support import xyxy2xywh
# from .eval_support import ModelEvaluatorBase
# from .ui_support import Progress

class _Metric:
    """
    Class for computing evaluation metrics for YOLOv8 model.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        mp(): Mean precision of all classes. Returns: Float.
        mr(): Mean recall of all classes. Returns: Float.
        map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
        map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
        map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
        mean_results(): Mean of results, returns mp, mr, map50, map.
        class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
        fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
        update(results): Update metric attributes with new evaluation results.
    """

    def __init__(self) -> None:
        """Initializes a Metric instance for computing evaluation metrics for the YOLOv8 model."""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @property
    def ap50(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        Returns the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        Returns the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map."""
        return [self.map, self.map50, self.mr, self.mp]

    def class_result(self, i):
        """Class-aware result, return p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """MAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        """Model fitness as a weighted combination of metrics."""
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.array(self.mean_results()) * w).sum()

    def update(self, results):
        """
        Updates the evaluation metrics of the model with a new set of results.

        Args:
            results (tuple): A tuple containing the following evaluation metrics:
                - p (list): Precision for each class. Shape: (nc,).
                - r (list): Recall for each class. Shape: (nc,).
                - f1 (list): F1 score for each class. Shape: (nc,).
                - all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
                - ap_class_index (list): Index of class for each AP score. Shape: (nc,).

        Side Effects:
            Updates the class attributes `self.p`, `self.r`, `self.f1`, `self.all_ap`, and `self.ap_class_index` based
            on the values provided in the `results` tuple.
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]
    
    
class OBBModelEvaluator(ModelEvaluatorBase):
    """
    This class evaluates the mAP for Object Detection models.
    """
    def __init__(self, model: dg.model.Model, **kwargs):
        """
        Constructor.

        Args:
            model (Detection model): PySDK detection model object
            kwargs (dict): arbitrary set of PySDK model parameters and the following evaluation parameters:
                show_progress (bool): show progress bar
                classmap (dict): dictionary which maps model category IDs to dataset category IDs
                pred_path (str): path to save the predictions as a JSON file of None if not required
        """

        # dictionary which maps model category IDs to dataset category IDs
        self.classmap: Optional[dict] = None
        # path to save the predictions as a JSON file
        self.pred_path: Optional[str] = None

        # TODO in future postprocesstype = OBB specific CPP version
        # will be separate from outputresulttype
        if model.output_postprocess_type not in [
            "DetectionYoloV8OBB",
            "None"
        ]:
            raise Exception("Model loaded for evaluation is not an OBB Model")
        
        # base constructor assigns kwargs to model or to self
        super().__init__(model, **kwargs)

    def evaluate(
        self,
        image_folder_path: str,
        ground_truth_annotations_path: str,
        max_images: int = 0,
    ) -> list:
        """
        Evaluation for the detection model.

        Args:
            image_folder_path (str): Path to images
            ground_truth_annotations_path (str): Path to the ground truth JSON annotations file (COCO format)
            max_images (int): max number of images used for evaluation. 0: all images in `image_folder_path` are used.

        Returns:
            the mAP statistics: [bbox_stats, kp_stats] for pose detection models and [bbox_stats] for non-pose models.
        """
        with open(ground_truth_annotations_path, 'r') as f:
            annos = json.load(f)
        
        for ann in annos:
            ann['img'] = os.path.join(image_folder_path, ann['img'])

        annos_exist = []
        for ann in annos:
            if os.path.exists(ann['img']):
                annos_exist.append(ann)

        if max_images > 0:
            annos_exist = annos_exist[0:max_images]

        # Remap classes
        if self.classmap:
            npclassmap = np.asarray(self.classmap)
            for ann in annos_exist:
                np_gt_cls = np.asarray(ann['cls'])
                ann['cls'] = npclassmap[np_gt_cls]

        all_ious = []
        all_cls = []
        all_confs = []

        pred_results = {}
        with self.model:
            if self.show_progress:
                progress = Progress(len(annos_exist))
            for image_number, predictions in enumerate(
                self.model.predict_batch([ann['img'] for ann in annos_exist])
            ):
                if self.show_progress:
                    progress.step()
                
                if self.pred_path:
                    pred_results[annos_exist[image_number]['img']] = predictions.results

                if len(predictions.results) > 0:
                    bbox_results = [r['bbox'] for r in predictions.results]
                    angle_results = [r['angle'] for r in predictions.results]
                    cls_results = [r['category_id'] for r in predictions.results]
                    conf_results = np.asarray([r['score'] for r in predictions.results])
                    bbox_results = np.asarray(bbox_results)
                    bbox_results = xyxy2xywh(bbox_results)
                    bbox_results = np.hstack((bbox_results, np.asarray(angle_results).reshape(-1, 1)))
                    cls_results = np.asarray(cls_results)
                else:
                    bbox_results = np.empty((0, 5), np.float32)
                    cls_results = np.empty((0), np.float32)
                    conf_results = np.empty((0), np.float32)

                gt_bbox = np.array(annos_exist[image_number]['bbox'])
                ious = OBBModelEvaluator._batch_probiou(gt_bbox, bbox_results)
                all_ious.append(ious)
                all_cls.append(cls_results)
                all_confs.append(conf_results)

        # save the predictions to a json file
        if self.pred_path:
            with open(self.pred_path, "w") as f:
                json.dump(pred_results, f, indent=4)

        nc = len(predictions._label_dictionary)
        all_gt_cls = [a['cls'] for a in annos]
        
        stats = OBBModelEvaluator._compute_map_results(all_ious, all_cls, all_confs, all_gt_cls, nc)

        return [stats]
    
    @staticmethod
    def _compute_map_results(all_ious, all_cls, all_confs, all_gt_cls, nc):
        match_results = []
        metrics = _Metric()
        metrics.nc = nc
        
        stats = dict(tp=[], conf=[], cls=[], gt_cls=[])
        iouv = np.linspace(0.5, 0.95, 10)

        for iou, cls, gt_cls, confs in zip(all_ious, all_cls, all_gt_cls, all_confs):
            if len(cls) == 0:
                if len(gt_cls):
                    empty_stat = dict(
                        tp=np.zeros((len(cls), iouv.size), dtype=bool),
                        conf=np.zeros(0),
                        cls=np.zeros(0),
                        gt_cls=gt_cls
                    )
                    for k in stats.keys():
                        stats[k].append(empty_stat[k])
                continue

            if len(gt_cls):
                match_results = OBBModelEvaluator._match_predictions(cls, np.array(gt_cls), iou, iouv)
                stats['tp'].append(match_results)
                stats['conf'].append(confs)
                stats['cls'].append(cls)
                stats['gt_cls'].append(gt_cls)

        stats = {k: np.concatenate(v, 0) for k, v in stats.items()}
        ap_results = OBBModelEvaluator._ap_per_class(**stats)[2:]
        metrics.update(ap_results)

        return metrics.mean_results()
    
    @staticmethod
    def _match_predictions(pred_classes, true_classes, iou, iouv):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes

        for i, threshold in enumerate(iouv.tolist()):
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return correct

    @staticmethod
    def _smooth(y, f=0.05):
        """Box filter of fraction f."""
        nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
        p = np.ones(nf // 2)  # ones padding
        yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
        return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed

    @staticmethod
    def _compute_ap(recall, precision):
        """
        Compute the average precision (AP) given the recall and precision curves.

        Args:
            recall (list): The recall curve.
            precision (list): The precision curve.

        Returns:
            (float): Average precision.
            (np.ndarray): Precision envelope curve.
            (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
        """
        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = "interp"  # methods: 'continuous', 'interp'
        if method == "interp":
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

    @staticmethod
    def _ap_per_class(
        tp, conf, cls, gt_cls, plot=False, on_plot=None, save_dir=None, names={}, eps=1e-16, prefix=""
    ):
        """
        Computes the average precision per class for object detection evaluation.

        Args:
            tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
            conf (np.ndarray): Array of confidence scores of the detections.
            pred_cls (np.ndarray): Array of predicted classes of the detections.
            target_cls (np.ndarray): Array of true classes of the detections.
            plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
            on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
            save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
            names (dict, optional): Dict of class names to plot PR curves. Defaults to an empty tuple.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
            prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

        Returns:
            tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.Shape: (nc,).
            fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class. Shape: (nc,).
            p (np.ndarray): Precision values at threshold given by max F1 metric for each class. Shape: (nc,).
            r (np.ndarray): Recall values at threshold given by max F1 metric for each class. Shape: (nc,).
            f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class. Shape: (nc,).
            ap (np.ndarray): Average precision for each class at different IoU thresholds. Shape: (nc, 10).
            unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
            p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
            r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
            f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
            x (np.ndarray): X-axis values for the curves. Shape: (1000,).
            prec_values (np.ndarray): Precision values at mAP@0.5 for each class. Shape: (nc, 1000).
        """
        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, cls = tp[i], conf[i], cls[i]

        # Find unique classes
        unique_classes, nt = np.unique(gt_cls, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        x, prec_values = np.linspace(0, 1, 1000), []

        # Average precision, precision and recall curves
        ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            i = cls == c
            n_l = nt[ci]  # number of labels
            n_p = i.sum()  # number of predictions
            if n_p == 0 or n_l == 0:
                continue

            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = OBBModelEvaluator._compute_ap(recall[:, j], precision[:, j])
                if j == 0:
                    prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

        prec_values = np.array(prec_values)  # (nc, 1000)

        # Compute F1 (harmonic mean of precision and recall)
        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
        names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
        names = dict(enumerate(names))  # to dict

        i = OBBModelEvaluator._smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
        p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
        tp = (r * nt).round()  # true positives
        fp = (tp / (p + eps) - tp).round()  # false positives
        return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values

    @staticmethod
    def _get_covariance_matrix(boxes):
        """
        Generating covariance matrix from obbs.

        Args:
            boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

        Returns:
            (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
        """
        # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
        gbbs = np.concatenate((np.power(boxes[:, 2:4], 2) / 12, boxes[:, 4:]), axis=-1)
        a, b, c = np.split(gbbs, 3, axis=-1)
        cos = np.cos(c)
        sin = np.sin(c)
        cos2 = np.power(cos, 2)
        sin2 = np.power(sin, 2)
        return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

    @staticmethod
    def _batch_probiou(obb1, obb2, eps=1e-7):
        """
        Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

        Args:
            obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
            obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
        """
        x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
        x2, y2 = (np.squeeze(x, -1)[None] for x in np.split(obb2[..., :2], 2, axis=-1))
        a1, b1, c1 = OBBModelEvaluator._get_covariance_matrix(obb1)
        a2, b2, c2 = (np.squeeze(x, -1)[None] for x in OBBModelEvaluator._get_covariance_matrix(obb2))
        sum_c_2 = np.power((c1 + c2), 2)

        t1 = (
            ((a1 + a2) * np.power((y1 - y2), 2) + (b1 + b2) * np.power((x1 - x2), 2)) / ((a1 + a2) * (b1 + b2) - sum_c_2 + eps)
        ) * 0.25
        t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - sum_c_2 + eps)) * 0.5
        t3 = np.log(
            ((a1 + a2) * (b1 + b2) - sum_c_2)
            / (4 * np.sqrt(np.clip((a1 * b1 - np.power(c1, 2)), a_min=0, a_max=None) * np.clip((a2 * b2 - np.power(c2, 2)), a_min=0, a_max=None)) + eps)
            + eps
        ) * 0.5
        bd = np.clip((t1 + t2 + t3), a_min=eps, a_max=100.0)
        hd = np.sqrt(1.0 - np.exp(-bd) + eps)
        return 1 - hd
