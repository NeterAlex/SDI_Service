from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy import create_engine
from sqlmodel import SQLModel, Session
from model import User, MildewData
from utils import Predictor

# Initialize server
app = FastAPI()

# Initialize db
sqlite_url: str = "sqlite:///database.db"
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=True, connect_args=connect_args)


def get_session():
    with Session(engine) as session:
        yield session


# Model Configurations
powdery_model_path: str = r"assets/models/powdery_m_20240106.pt"
downy_model_path: str = r"assets/models/downy_m_20231108.pt"

# Initialize predictors
downy_pd = Predictor(downy_model_path)
powdery_pd = Predictor(powdery_model_path)


# On Startup
@app.on_event("startup")
def on_startup() -> None:
    SQLModel.metadata.create_all(engine)


# Controllers
@app.get("/ping")
async def root():
    return {"message": "pong"}


@app.post("/calc/downy")
async def downy_mildew_detect_controller(*, session: Session = Depends(get_session),
                                         file: UploadFile = File(...), user_id: int) -> object:
    """
    Downy Mildew Detect Controller
    :param session: Session
    :param file: Image file
    :return: Result with data and other info
    """
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
        # operate db
        user = session.get(User, user_id)
        if not user:
            return {"is_success": False, "message": "用户不存在"}
        data = MildewData(user=user, type="downy", data=result)
        session.add(data)
        session.commit()
        return result
    except Exception as e:
        return {
            "is_success": False,
            "message": "识别失败, 由于" + e.__str__(),
            "created_at": datetime.now().isoformat(timespec="seconds") + 'Z',
            "data": [],
        }


@app.post("/calc/powdery")
async def powdery_mildew_detect_controller(file: UploadFile = File(...)) -> object:
    """
    Powdery Mildew Detect Controller
    :param file: Image file
    :return: Result with data and other info
    """
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


@app.post("/user/register")
async def register_user(*, session: Session = Depends(get_session), username: str, password: str) -> object:
    try:
        user = User(username=username, password=password)
        session.add(user)
        session.commit()
        session.refresh(user)
        return {
            "is_success": True,
            "message": "注册成功",
        }
    except Exception as e:
        return {
            "is_success": False,
            "message": "注册失败, 由于" + e.__str__(),
        }
