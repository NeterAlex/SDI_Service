from pprint import pprint
from typing import Any

from PIL import Image
from ultralytics import YOLO

model_path = r"C:\Projects\soybean-model\models\pretrain-20231018-1602.pt"

model = YOLO(model_path)


def transform_data(boxes: Any) -> object:
    id: int = 1
    xyxy_list = boxes.xyxy.tolist()
    conf_list = boxes.conf.tolist()
    cls_list = boxes.cls.tolist()
    merged_data: list[dict[str, int | Any]] = []
    for xyxy, conf, cls in zip(xyxy_list, conf_list, cls_list):
        merged_data.append({
            'id': id,
            'xyxy': xyxy,
            'conf': conf,
            'cls': cls
        })
        id += 1
    return merged_data


if __name__ == '__main__':
    results = model.predict(r"C:\Projects\soybean-model\test_tiny\union1.png")
    boxes = results[0].boxes
    pprint(transform_data(boxes))
    results[0].plot()
    # model.export(format="onnx", opset=12, simplify=True)
