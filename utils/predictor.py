import io
import time
import uuid
from typing import Any

from PIL import Image
from ultralytics import YOLO


class Predictor:
    model_path: str = ""
    model = None

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(self.model_path, task="detect")

    def load(self):
        self.model = YOLO(self.model_path, task="detect")

    def predict_bytes(self, image: Any, save_image: bool = False, conf=0.2, iou=0.5):
        results = self.model.predict(image, conf=conf, iou=iou)
        boxes = results[0].boxes
        image_stream = io.BytesIO()
        im_array = results[0].plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save(image_stream, format="JPEG")
        if save_image:
            file_str = f'assets/image_cache/{time.strftime("%Y-%m-%d-%H-%M-%S")}-{uuid.uuid4().hex[:8]}.jpg'
            im.save(file_str)

        return self.transform_data(boxes), image_stream.getvalue()

    @staticmethod
    def transform_data(boxes: Any) -> list[dict[str, int | Any]]:
        _id: int = 1
        xyxy_list: list = boxes.xyxy.tolist()
        conf_list: list = boxes.conf.tolist()
        cls_list: list = boxes.cls.tolist()
        merged_data: list[dict[str, int | Any]] = []
        for xyxy, conf, cls in zip(xyxy_list, conf_list, cls_list):
            merged_data.append({"id": _id, "xyxy": xyxy, "conf": conf, "cls": cls})
            _id += 1
        return merged_data
