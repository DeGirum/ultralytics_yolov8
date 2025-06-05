# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.data import MultiLabelClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import MultiLabelClassifyMetrics
from ultralytics.utils.plotting import plot_images


class MultiLabelClassificationValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a multi label classification model.

    Notes:
        - Torchvision multi label classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.models.yolo.multi_label_classify import MultiLabelClassificationValidator

        args = dict(model="yolo11n-cls.pt", data="imagenet10")
        validator = MultiLabelClassificationValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, is_binary=True):
        """Initializes MultiLabelClassificationValidator instance with args, dataloader, save_dir, and progress bar."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "multi_label_classify"
        self.metrics = MultiLabelClassifyMetrics(is_binary=is_binary)
        self.is_binary = is_binary  # True for binary classification, False for multi-class classification

    # def get_desc(self):
    #     """Returns a formatted string summarizing multi label classification metrics."""
    #     return ("%22s" * 3) % (
    #         "labels",
    #         "mean_acc",
    #         "mean_f1_score"
    #     )

    def init_metrics(self, model):
        """Initialize class names, and mean accuracy."""
        self.labels = model.label_names if hasattr(model, "label_names") else model.model.label_names
        self.names = model.names
        self.nl = len(self.labels)
        self.nc = len(self.names)
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        """Preprocesses input batch and returns it."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        """Updates metrics with predictions and targets from the current batch."""
        if self.is_binary:
            preds = (preds > 0.5).int().cpu()
        else:
            # Expecting preds of shape [B, nl * nc] or [B, nl, nc]
            if preds.ndim == 2:
                preds = preds.view(-1, self.nl, self.nc)
            preds = preds.cpu()
            # preds = preds.argmax(dim=2).cpu()  # [B, nl]

        self.targets.append(batch["cls"].int().cpu())
        self.pred.append(preds)

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes metrics of the model such as speed."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def postprocess(self, preds, img_shape):
        """Preprocesses the multi label classification predictions."""
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def get_stats(self):
        """Returns a dictionary of metrics obtained by processing targets and predictions."""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def build_dataset(self, img_path):
        """Creates and returns a MultiLabelClassificationDataset instance using given image path and preprocessing parameters."""
        return MultiLabelClassificationDataset(img_path, args=self.args, augment=False, prefix=self.args.split)

    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader for multi label classification tasks with given parameters."""
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        """Prints evaluation metrics for multi-output multi-class classification model."""
        # Gather scalar metrics
        metric_values = [
            self.metrics.mean_acc,
            self.metrics.mean_f1_score,
            self.metrics.sequence_acc,
            self.metrics.top1_acc,
            self.metrics.topk_acc,
        ]

        # Print header row with metric keys
        header_fmt = "%22s" + "%22s" * len(self.metrics.keys)
        LOGGER.info(header_fmt % ("all", *self.metrics.keys))

        # Print actual metric values
        value_fmt = "%22s" + "%22.3g" * len(metric_values)
        LOGGER.info(value_fmt % ("all", *metric_values))

        # Optional: per-output (per-label) accuracy
        if getattr(self.args, "verbose", False) and not getattr(self, "training", False) and getattr(self, "nl", 1) > 1:
            value_fmt = "%22s%11.3f"
            label_fmt = "%22s%11s"
            LOGGER.info(label_fmt % ("label", "acc"))
            for i, acc in enumerate(self.metrics.label_acc.tolist()):
                label_name = self.labels[i] if hasattr(self, "labels") and i < len(self.labels) else f"Label {i}"
                LOGGER.info(value_fmt % (label_name, acc))

            LOGGER.info(label_fmt % ("class", "acc"))
            for i, acc in enumerate(self.metrics.per_class_acc.tolist()):
                class_name = self.names[i] if hasattr(self, "names") and i < len(self.names) else f"Class {i}"
                LOGGER.info(value_fmt % (class_name, acc))

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"],  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        if self.is_binary:
            thresholded_preds = (preds > 0.5).int()
        else:
            if preds.ndim == 2:
                preds = preds.view(-1, self.nl, self.nc)
            thresholded_preds = preds.argmax(dim=2)

        plot_images(
            batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=thresholded_preds,
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )
