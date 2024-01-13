from typing import List

from model import MildewData


class Processor:
    enabled: bool

    def __init__(self):
        self.enabled = True

    @staticmethod
    def organize_detected_result(data: List[dict], image: str) -> dict:
        cls_dict = {}
        for item in data:
            item["xyxy"] = "omit"
            item["conf"] = "{:.2f}".format(float(item["conf"]))
            item["image"] = image
            cls_name = item["cls"]
            if cls_name in cls_dict:
                cls_dict[cls_name].append(item)
            else:
                cls_dict[cls_name] = [item]
        return cls_dict
