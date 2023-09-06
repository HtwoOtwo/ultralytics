# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.addyolo.task import ADDYOLODetectionModel
from ultralytics.utils import RANK


class ADDYOLOTrainer(DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on an RT-DETR detection model.

    Notes:
        - F.grid_sample used in rt-detr does not support the `deterministic=True` argument.
        - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.

    Example:
        ```python
        from ultralytics.models.rtdetr.train import RTDETRTrainer

        args = dict(model='rtdetr-l.yaml', data='coco8.yaml', imgsz=640, epochs=3)
        trainer = RTDETRTrainer(overrides=args)
        trainer.train()
        ```
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = ADDYOLODetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

