import os
import json
import hashlib
from pathlib import Path


def setup_mock_data():
    """设置mock数据的辅助脚本"""
    mock_dir = Path("mock_data")
    mock_dir.mkdir(exist_ok=True)

    # 扫描mock_data目录中的原始图片
    for img_file in mock_dir.glob("*.jpg"):
        if not img_file.name.startswith("original_"):
            continue

        # 读取图片并计算hash
        with open(img_file, 'rb') as f:
            file_bytes = f.read()

        file_hash = hashlib.md5(file_bytes).hexdigest()

        # 重命名图片文件（这将是mock的结果图片）
        new_img_path = mock_dir / f"{file_hash}.jpg"
        if not new_img_path.exists():
            img_file.rename(new_img_path)
            print(f"重命名图片: {img_file.name} -> {new_img_path.name}")

        # 创建对应的JSON数据文件
        json_path = mock_dir / f"{file_hash}.json"
        if not json_path.exists():
            # 这里填入你想要的mock预测数据
            mock_data = {
                "detection_count": 5,
                "confidence": 0.85,
                "areas": [
                    {"x": 100, "y": 150, "width": 50, "height": 40, "confidence": 0.9},
                    {"x": 200, "y": 300, "width": 60, "height": 45, "confidence": 0.8},
                    # 添加更多mock检测区域...
                ],
                "severity": "medium",
                "recommendations": ["建议使用杀菌剂", "加强通风"]
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(mock_data, f, ensure_ascii=False, indent=2)

            print(f"创建数据文件: {json_path.name}")
            print(f"对应的原始图片hash: {file_hash}")


def add_mock_image(original_image_path: str, result_image_path: str, mock_data: dict):
    """添加单个mock图片和数据"""
    mock_dir = Path("mock_data")
    mock_dir.mkdir(exist_ok=True)

    # 读取原始图片并计算hash
    with open(original_image_path, 'rb') as f:
        file_bytes = f.read()

    file_hash = hashlib.md5(file_bytes).hexdigest()

    # 复制结果图片
    result_path = mock_dir / f"{file_hash}.jpg"
    with open(result_image_path, 'rb') as src, open(result_path, 'wb') as dst:
        dst.write(src.read())

    # 保存mock数据
    json_path = mock_dir / f"{file_hash}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mock_data, f, ensure_ascii=False, indent=2)

    print(f"已添加mock数据:")
    print(f"  原始图片hash: {file_hash}")
    print(f"  结果图片: {result_path}")
    print(f"  数据文件: {json_path}")


if __name__ == "__main__":
    # 方式1: 批量设置（需要先将图片重命名为original_xxx.jpg放在mock_data目录）
    setup_mock_data()

    # 方式2: 手动添加单个mock数据
    # mock_data_example = {
    #     "detection_count": 3,
    #     "confidence": 0.92,
    #     "areas": [
    #         {"x": 120, "y": 180, "width": 40, "height": 35, "confidence": 0.95}
    #     ],
    #     "severity": "high"
    # }
    # add_mock_image("path/to/original.jpg", "path/to/result.jpg", mock_data_example)