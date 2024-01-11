import pprint
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File

from utils.predictor import Predictor

# Model Configurations
powdery_model_path: str = r"models/powdery_m_20240106.pt"
downy_model_path: str = r"models/downy_m_20231108.pt"

calculator_router = APIRouter()

# Init predictors
downy_pd = Predictor(downy_model_path)
powdery_pd = Predictor(powdery_model_path)


@calculator_router.post("/calc/downy")
async def downy_mildew_detect(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        data, result_stream = downy_pd.predict_bytes(image, True)
        await file.close()
        result = {
            "is_success": True,
            "message": "识别成功",
            "created_at": datetime.now().isoformat(timespec="seconds") + 'Z',
            "data": data,
        }
        return result
    except Exception as e:
        return {
            "is_success": False,
            "message": "识别失败, 由于" + e.__str__(),
            "created_at": datetime.now().isoformat(timespec="seconds") + 'Z',
            "data": [],
        }


@calculator_router.post("/calc/powdery")
async def powdery_mildew_detect(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        data, result_stream = powdery_pd.predict_bytes(image, True)
        await file.close()
        return data
    except Exception as e:
        return {
            "is_success": False,
            "message": "识别失败, 由于" + e.__str__(),
            "created_at": datetime.now().isoformat(timespec="seconds") + 'Z',
            "data": [],
        }
