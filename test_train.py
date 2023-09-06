from ultralytics.models.addyolo import ADDYOLO


if __name__ == "__main__":
    model = ADDYOLO("./ultralytics/cfg/models/v8/yolov8_add.yaml")
    model.train(data='coco128.yaml', epochs=100, imgsz=640)