import io
import os
import uuid
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from fastapi import UploadFile


def compress_img(img_bytes: bytes, *, f="JPEG", quality=25) -> bytes:
    compressed_image = Image.open(io.BytesIO(img_bytes))
    compressed_stream = io.BytesIO()
    compressed_image.save(compressed_stream, format=f, quality=quality, optimize=True)
    compressed_bytes = compressed_stream.getvalue()
    return compressed_bytes


def save_img_static(
    userid: int, img_type: str, img_bytes: bytes, *, time_format="%Y%m%d%H%M%S"
) -> str:
    current_time = datetime.now().strftime(time_format)
    save_relative_path = os.path.join(
        "static",
        "results",
        img_type,
        f"ID{userid}-{current_time}-{uuid.uuid4().hex[:8]}.jpg",
    )
    save_obvious_path = os.path.join(os.getcwd(), save_relative_path)
    with open(save_obvious_path, "wb") as f:
        f.write(img_bytes)
    return save_relative_path


async def read_img(file: UploadFile) -> object:
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    await file.close()
    return image
