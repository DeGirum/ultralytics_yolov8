# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import MultiLabelDetectionModel
from ultralytics.utils import RANK


class MultiLabelDetectionTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the yolo.detect.DetectionTrainer class for training based on a detection model.

    This trainer specializes in multi-label object detection tasks, handling the specific requirements for training YOLO models
    for multi-label object detection.

    Attributes:
        model (MultiLabelDetectionModel): The YOLO multi-label detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (Tuple[str]): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO multi-label detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_metrics: Plot metrics from a CSV file.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.multi_label_detect import MultiLabelDetectionTrainer
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = MultiLabelDetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return a YOLO multi-label detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (MultiLabelDetectionModel): YOLO multi-label detection model.
        """
        model = MultiLabelDetectionModel(cfg, nc=self.data["nc"], nc_per_label=self.data["nc_per_label"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        super().set_model_attributes()
        self.model.nc_per_label = self.data["nc_per_label"]  # attach number of classes per label to model
        self.model.label_class_names = self.data["label_class_names"]  # attach label class names to model

    def get_validator(self):
        """Return a MultiLabelDetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "mlb_loss"
        return yolo.multi_label_detect.MultiLabelDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
