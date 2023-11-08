import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Response

from model.predictor import Predictor

app = FastAPI()
# Swagger URL: http://127.0.0.1:8000/docs
# V8N model_path = r"C:\Projects\soybean-model\models\pretrain-20231018-1602.pt"
# V8M model_path = r"C:\Projects\soybean-disease-Identifier\service\models\pretrain-v8m-20231108-1.pt"
model_path = r"C:\Projects\soybean-disease-Identifier\service\models\pretrain-v8m-20231108-2.pt"
pd = Predictor(model_path)
Predictor.load(pd)


@app.get("/ping")
async def root():
    return {"message": "pong"}


@app.post("/calc")
async def model_calc(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        data, result_stream = pd.predict_bytes(image, True)
        await file.close()
        # return data
        return data
    except Exception as e:
        return {
            "status": "failed",
            "info": e.__dict__
        }
