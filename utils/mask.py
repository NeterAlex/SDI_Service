import cv2
import numpy as np


def seg_tag(boxes_data, cv_image):
    # 读取原图
    if cv_image is None:
        raise ValueError("无法读取图片")
    image = cv_image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
        alpha = 0.1
        cv2.addWeighted(leaf_overlay, alpha, image, 1 - alpha, 0, image)

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

        alpha = 0.3
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.rectangle(image, (lx, ly), (rx, ry), (0, 230, 0), 3)
        cv2.putText(
            image,
            f"{index}",
            (lx, ly - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 230, 0),
            2,
        )
        # 在叶片上标注病变率
        text = f"Lesion ratio: {lesion_ratio:.2%}, count: {lesion_count}, avg_gray: {avg_gray_value:.2f}"
        cv2.putText(
            image, text, (lx, ry + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 230, 0), 2
        )

    _, image_bytes = cv2.imencode(".jpg", image)
    image_bytes = image_bytes.tobytes()
    return leaf_lesion_areas, image_bytes
