# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
addyolo model interface
"""

from ultralytics.engine.model import Model

from ultralytics.models import yolo  # noqa
from ultralytics.models.addyolo.train import ADDYOLOTrainer
from ultralytics.models.addyolo.task import ADDYOLODetectionModel
    


class ADDYOLO(Model):
    """
    ADDYOLO model interface.
    """

    def __init__(self, model="yolov8.yaml") -> None:
        super().__init__(model=model, task='detect')

    @property
    def task_map(self):
        return {
            'detect': {
                'trainer': ADDYOLOTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': yolo.detect.DetectionPredictor,
                'model': ADDYOLODetectionModel}}


