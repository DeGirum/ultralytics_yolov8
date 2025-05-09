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

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initializes MultiLabelClassificationValidator instance with args, dataloader, save_dir, and progress bar."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = "multi_label_classify"
        self.metrics = MultiLabelClassifyMetrics()

    def get_desc(self):
        """Returns a formatted string summarizing multi label classification metrics."""
        return ("%22s" * 3) % (
            "classes",
            "mean_acc",
            "mean_f1_score"
        )

    def init_metrics(self, model):
        """Initialize class names, and mean accuracy."""
        self.names = model.names
        self.nc = len(model.names)
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        """Preprocesses input batch and returns it."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        """Updates running metrics with model predictions and batch targets."""
        binary_preds = (preds > 0.5).int().cpu()

        # Collect targets
        binary_targets = batch["cls"].int().cpu()

        # Append to lists for metric calculation later
        self.pred.append(binary_preds)
        self.targets.append(binary_targets)

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
        """Prints evaluation metrics for multi label classification model."""
        pf = "%22s" + "%22.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.metrics.mean_acc, self.metrics.mean_f1_score))
        if self.args.verbose and not self.training and self.nc > 1:
            for i, c in enumerate(self.metrics.per_label_acc):
                pf = "%22s%11.3f"
                LOGGER.info(pf % (self.names[i], c))
            
    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"].view(-1),  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=preds,
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
