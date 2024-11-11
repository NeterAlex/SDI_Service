import cv2
import numpy as np


def sort_xyxy(boxes_data: list[dict], y_tolerance: int = 500) -> list[dict]:
    """
    按 xyxy 排序数据
    @param boxes_data: 包含 xyxy 字段的 list[dict]
    @param y_tolerance: 用于判定是否为一行内的 y 容忍度
    @return:
    """
    detections_sorted_by_y = sorted(boxes_data, key=lambda d: d["xyxy"][1])

    grouped_detections = []
    current_group = [detections_sorted_by_y[0]]

    for detection in detections_sorted_by_y[1:]:
        prev_y2 = current_group[-1]["xyxy"][3]
        current_y2 = detection["xyxy"][3]
        if abs(current_y2 - prev_y2) <= y_tolerance:
            current_group.append(detection)
        else:
            grouped_detections.append(sorted(current_group, key=lambda d: d["xyxy"][0]))
            current_group = [detection]

    if current_group:
        grouped_detections.append(sorted(current_group, key=lambda d: d["xyxy"][0]))

    sorted_boxes = [d for group in grouped_detections for d in group]
    return sorted_boxes


def seg_tag(boxes_data: list[dict], cv_image: object):
    # 读取原图
    if cv_image is None:
        raise ValueError("无法读取图片")
    image = cv_image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # boxes_data = sort_xyxy(boxes_data)

    leaf_boxes = [box for box in boxes_data if box["cls"] == 4]  # =4
    lesion_boxes = [box for box in boxes_data if 1 <= box["cls"] <= 3]  # <=3
    leaf_lesion_areas = []

    # 为每个叶片创建mask并计算病变率
    for index, leaf in enumerate(leaf_boxes):
        lesion_count = 0
        lx, ly, rx, ry = map(int, leaf["xyxy"])
        lw = rx - lx
        lh = ry - ly
        # 创建叶片掩码
        leaf_mask = np.zeros((lh, lw), dtype=np.uint8)
        center = (lw // 2, lh // 2)
        axes = (lw // 2 - 2, lh // 2 - 2)
        cv2.ellipse(leaf_mask, center, axes, 0, 0, 360, 255, -1)

        # 计算实际叶片面积（椭圆面积）
        leaf_area = cv2.countNonZero(leaf_mask)

        # 在原图上绘制半透明绿色叶片区域
        leaf_overlay = image.copy()
        leaf_roi = leaf_overlay[ly:ry, lx:rx]
        leaf_roi[leaf_mask > 0] = (0, 255, 0)  # 绿色

        # 将半透明叶片区域叠加到原图
        text_background_alpha = 0.1
        cv2.addWeighted(
            leaf_overlay,
            text_background_alpha,
            image,
            1 - text_background_alpha,
            0,
            image,
        )

        lesion_mask = np.zeros((lh, lw), dtype=np.uint8)
        overlay = image.copy()

        gray_values = []

        for lesion in lesion_boxes:
            x1, y1, x2, y2 = map(int, lesion["xyxy"])
            if x1 >= rx or x2 <= lx or y1 >= ry or y2 <= ly:
                continue

            lesion_count += 1
            rel_x1 = max(0, x1 - lx)
            rel_y1 = max(0, y1 - ly)
            rel_x2 = min(lw, x2 - lx)
            rel_y2 = min(lh, y2 - ly)

            lesion_gray_area = gray_image[y1:y2, x1:x2]  # 灰度区域
            actual_lesion_area = np.zeros(
                (rel_y2 - rel_y1, rel_x2 - rel_x1), dtype=np.uint8
            )
            center = ((rel_x2 - rel_x1) // 2, (rel_y2 - rel_y1) // 2)
            axes = ((rel_x2 - rel_x1) // 2, (rel_y2 - rel_y1) // 2)
            cv2.ellipse(actual_lesion_area, center, axes, 0, 0, 360, 255, -1)

            lesion_mask[rel_y1:rel_y2, rel_x1:rel_x2] = cv2.bitwise_or(
                lesion_mask[rel_y1:rel_y2, rel_x1:rel_x2], actual_lesion_area
            )
            # 计算灰度
            resized_mask = cv2.resize(
                actual_lesion_area,
                (lesion_gray_area.shape[1], lesion_gray_area.shape[0]),
            )
            gray_mask = resized_mask > 0
            gray_values.append(np.mean(lesion_gray_area[gray_mask]))

            overlay_roi = overlay[y1:y2, x1:x2]
            overlay_roi[resized_mask > 0] = (0, 0, 255)

        lesion_mask = cv2.bitwise_and(lesion_mask, leaf_mask)
        lesion_area = cv2.countNonZero(lesion_mask)
        lesion_ratio = lesion_area / leaf_area
        avg_gray_value = np.mean(gray_values) if gray_values else 0

        leaf_info = {
            "leaf_index": index,
            "leaf_area": leaf_area,
            "lesion_area": lesion_area,
            "lesion_ratio": round(lesion_ratio, 4),
            "lesion_count": lesion_count,
            "avg_gray_value": round(float(avg_gray_value), 4),
        }
        leaf_lesion_areas.append(leaf_info)

        # 绘制叶片框
        leaf_background_alpha = 0.3
        cv2.addWeighted(
            overlay, leaf_background_alpha, image, 1 - leaf_background_alpha, 0, image
        )
        cv2.rectangle(image, (lx, ly), (rx, ry), (104, 31, 17), 3)
        cv2.putText(
            image,
            f"{index}",
            (lx, ly - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (104, 31, 17),
            2,
        )

        # 在叶片上标注病变率
        text = f"Lesion ratio: {lesion_ratio:.2%}, Gray-scale: {avg_gray_value:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
        )
        top_left = (lx, ry - text_height - baseline - 13)
        bottom_right = (lx + 5 + text_width, ry + baseline - 13)
        overlay = image.copy()
        cv2.rectangle(
            overlay, top_left, bottom_right, (104, 31, 17), thickness=cv2.FILLED
        )
        text_background_alpha = 0.5
        cv2.addWeighted(
            overlay, text_background_alpha, image, 1 - text_background_alpha, 0, image
        )

        cv2.putText(
            image,
            text,
            (lx + 5, ry - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    _, image_bytes = cv2.imencode(".jpg", image)
    image_bytes = image_bytes.tobytes()
    return leaf_lesion_areas, image_bytes
