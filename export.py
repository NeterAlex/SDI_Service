import os.path
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils import Predictor, seg_tag


def distinguish_tier(lesion_ratio: float, lesion_count: int) -> int:
    if lesion_count >= 60 and lesion_ratio > 0.6:
        return 1
    elif lesion_count >= 40 and 0.4 <= lesion_ratio <= 0.5:
        return 3
    elif lesion_count >= 20 and 0.2 <= lesion_ratio < 0.4:
        return 5
    elif lesion_count >= 5 and 0.05 <= lesion_ratio < 0.2:
        return 7
    elif lesion_count < 5 and lesion_ratio < 0.05:
        return 9
    else:
        severity_score = (lesion_ratio * 10) + (lesion_count / 20)
        if severity_score >= 6.0:
            return 1
        elif severity_score >= 4.0:
            return 3
        elif severity_score >= 2.5:
            return 5
        elif severity_score >= 1.5:
            return 7
        else:
            return 9


# 输入图片文件夹路径
input_folder = (
    r"C:\Data\数据集\20241013大豆灰斑病数据\灰斑病整理照片20141013（1）（2）（3）\data"
)
output_excel = "leaf_analysis_results.xlsx"
frogeye_pd = Predictor(r"assets/models/frogeye_l_20241109.pt")

all_leaf_data = []

# 遍历文件夹中的每个图像文件
for img_file in Path(input_folder).glob("*.*"):
    if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:

        with open(img_file, "rb") as image_file:
            image_bytes = image_file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        data, _ = frogeye_pd.predict_bytes(image, False, conf=0.05, iou=0.05)
        data, image_bytes = seg_tag(data, image)
        save_path = os.path.join(input_folder, "result", img_file.name)
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        leaf_data_list = data
        for leaf_info in leaf_data_list:
            tier = distinguish_tier(
                leaf_info["lesion_ratio"], leaf_info["lesion_count"]
            )
            all_leaf_data.append(
                {
                    "数据": img_file.stem,
                    "叶片": leaf_info["leaf_index"],
                    "病害等级": tier,
                    "病斑面积比": leaf_info["lesion_ratio"],
                    "病斑数": leaf_info["lesion_count"],
                    "平均灰度": leaf_info["avg_gray_value"],
                }
            )

        print(f"已处理{image_file.name}")

df = pd.DataFrame(all_leaf_data)

df = df[
    [
        "数据",
        "叶片",
        "病害等级",
        "病斑面积比",
        "病斑数",
        "平均灰度",
    ]
]

df.to_excel(output_excel, index=False)
print(f"数据已保存到 {output_excel}")
