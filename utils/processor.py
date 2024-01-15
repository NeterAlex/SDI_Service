from typing import List

from model import MildewData


class Processor:
    enabled: bool

    def __init__(self):
        self.enabled = True

    @staticmethod
    def organize_detected_result(data: List[dict]) -> dict:
        """
        Organize the detected result via cls
        :param data: list of detected result
        :param image: image path of certain result
        :return: results classified by cls
        """
        cls_dict = {}
        for item in data:
            item["xyxy"] = "omit"
            item["conf"] = "{:.2f}".format(float(item["conf"]))
            cls_name = f"T{int(item['cls'])}"
            if cls_name in cls_dict:
                cls_dict[cls_name].append(item)
            else:
                cls_dict[cls_name] = [item]
        return cls_dict

    @staticmethod
    def organize_downy_detected_info(data: List[dict]) -> dict:
        cls_dict = {}
        count_dict = {}
        for item in data:
            cls_name = f"{int(item['cls'])}级"
            if cls_name in cls_dict:
                cls_dict[cls_name].append(item)
            else:
                cls_dict[cls_name] = [item]
        for cls_name, items in cls_dict.items():
            count_dict[cls_name] = len(items)
        return count_dict
