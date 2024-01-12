import os
import uuid
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy import create_engine
from sqlmodel import SQLModel, Session, select
from starlette.staticfiles import StaticFiles

from model import User, MildewData
from utils import Predictor, hash_password, generate_jwt_token, verify_password

# Initialize server
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize db
sqlite_url: str = "sqlite:///database.db"
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=False, connect_args=connect_args)


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
    :param user_id: User ID
    :param session: Session
    :param file: Image file
    :return: Result with data and other info
    """
    try:
        image_bytes = await file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        data, result_stream = downy_pd.predict_bytes(image, False)
        await file.close()
        # save image
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        save_relative_path = os.path.join("static", "results", "downy-mildew",
                                          f"ID{user_id}-{current_time}-{uuid.uuid4().hex[:8]}.jpg")
        save_obvious_path = os.path.join(os.getcwd(), save_relative_path)
        with open(save_obvious_path, "wb") as f:
            f.write(result_stream)
        # operate db
        user = session.get(User, user_id)
        if not user:
            return {"is_success": False, "message": "用户不存在"}
        data = MildewData(user=user, type="downy", data=data.__str__(), image=save_relative_path)
        session.add(data)
        session.commit()
        # make result
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


@app.post("/calc/powdery")
async def powdery_mildew_detect_controller(*, session: Session = Depends(get_session),
                                           file: UploadFile = File(...), user_id: int) -> object:
    """
    Powdery Mildew Detect Controller
    :param user_id: User ID
    :param session: Session
    :param file: Image file
    :return: Result with data and other info
    """
    try:
        image_bytes = await file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        data, result_stream = powdery_pd.predict_bytes(image, False)
        await file.close()
        # save image
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        save_relative_path = os.path.join("static", "results", "powdery-mildew",
                                          f"ID{user_id}-{current_time}-{uuid.uuid4().hex[:8]}.jpg")
        save_obvious_path = os.path.join(os.getcwd(), save_relative_path)
        with open(save_obvious_path, "wb") as f:
            f.write(result_stream)
        # operate db
        user = session.get(User, user_id)
        if not user:
            return {"is_success": False, "message": "用户不存在"}
        data = MildewData(user=user, type="powdery", data=data.__str__(), image=save_relative_path)
        session.add(data)
        session.commit()
        # make result
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


@app.post("/user/register")
async def register_user(*, session: Session = Depends(get_session), username: str, password: str,
                        nickname: str) -> object:
    try:
        user = User(username=username, password=hash_password(password), nickname=nickname)
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


@app.post("/user/login")
async def login_user(*, session: Session = Depends(get_session), username: str, password: str) -> object:
    try:
        statement = select(User).where(User.username == username)
        user = session.exec(statement).first()
        if not user:
            return {"is_success": False, "message": "用户不存在"}
        if not verify_password(password, user.password):
            print(password, user.password, verify_password(password, user.password))
            return {"is_success": False, "message": "密码不匹配"}
        return {
            "is_success": True,
            "message": "登录成功",
            "created_at": datetime.now().isoformat(timespec="seconds") + 'Z',
            "data": {"jwt_token": generate_jwt_token(user)}
        }
    except Exception as e:
        return {
            "is_success": False,
            "message": "登录失败, 由于" + e.__str__(),
        }
