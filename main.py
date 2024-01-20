import json
import logging
import os
import uuid
from datetime import datetime
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends, Request, HTTPException, status, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Session, select, col, create_engine
from starlette.staticfiles import StaticFiles

from model import User, MildewData
from utils import Predictor, hash_password, generate_jwt_token, verify_password, decode_jwt_token, Processor
import os

# Initialize server
os.environ['TZ'] = 'Asia/Shanghai'
print(datetime.now().isoformat(timespec="seconds") + 'Z')
app = FastAPI(docs_url=None, redoc_url=None)
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


# Global error handler
@app.exception_handler(Exception)
async def exception_handler(request, exc):
    logging.exception(f"[Error] {exc}")
    return {
        "success": False,
        "message": "处理失败，由于" + str(exc),
        "time": datetime.now().isoformat(timespec="seconds") + 'Z'
    }


# Middleware
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def verify_token(request: Request, call_next):
    """
    Middleware to verify that the token is valid
    """
    # Get request path
    path: str = request.get('path')
    # Exclude /login & /docs & /static
    if path.startswith('/ping') | path.startswith('/static') | path.startswith('/user/login') | path.startswith(
            '/user/register'):
        response = await call_next(request)
        return response
    else:
        try:
            # Get token
            authorization: str = request.headers.get('Authorization')
            if authorization is None:
                return Response(status_code=401)
            token = authorization.split(' ')[1]
            # Verify token
            user_claim = decode_jwt_token(token)
            if user_claim.get('user_id') is not None:
                response = await call_next(request)
                return response
        except Exception:
            return Response(status_code=401)


# Controllers
@app.get("/ping")
async def root():
    return {"message": "pong"}


@app.post("/calc/downy")
async def downy_mildew_detect(*, session: Session = Depends(get_session),
                              file: UploadFile = File(...), user_id: Annotated[int, Form()]) -> object:
    """
    Downy Mildew Detect Controller
    :param user_id: User ID
    :param session: Session
    :param file: Image file
    :return: Result with data and other info
    """
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
        return {"success": False, "message": "用户不存在"}
    data = MildewData(user=user, type="downy", data=data.__str__(), image=save_relative_path)
    session.add(data)
    session.commit()
    # make result
    result = {
        "success": True,
        "message": "识别成功",
        "time": datetime.now().isoformat(timespec="seconds") + 'Z',
        "data": data,
    }
    return result


@app.post("/calc/powdery")
async def powdery_mildew_detect(*, session: Session = Depends(get_session),
                                file: UploadFile = File(...), user_id: Annotated[int, Form()]) -> object:
    """
    Powdery Mildew Detect Controller
    :param user_id: User ID
    :param session: Session
    :param file: Image file
    :return: Result with data and other info
    """
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
        return {"success": False, "message": "用户不存在"}
    data = MildewData(user=user, type="powdery", data=data.__str__(), image=save_relative_path)
    session.add(data)
    session.commit()
    # make result
    result = {
        "success": True,
        "message": "识别成功",
        "time": datetime.now().isoformat(timespec="seconds") + 'Z',
        "data": data,
    }
    return result


@app.post("/user/register")
async def register_user(*, session: Session = Depends(get_session), username: Annotated[str, Form()],
                        password: Annotated[str, Form()],
                        nickname: Annotated[str, Form()]) -> object:
    """
    Register new user
    :param session: Session
    :param username: Username
    :param password: Password
    :param nickname: Nickname
    :return: Result with if register is successful
    """
    user = User(username=username, password=hash_password(password), nickname=nickname)
    session.add(user)
    session.commit()
    session.refresh(user)
    return {
        "success": True,
        "message": "注册成功",
    }


@app.post("/user/login")
async def login_user(*, session: Session = Depends(get_session), username: Annotated[str, Form()],
                     password: Annotated[str, Form()]) -> object:
    """
    User login with username and password
    :param session: Session
    :param username: Username
    :param password: Password
    :return: Result with jwt data
    """
    statement = select(User).where(User.username == username)
    user = session.exec(statement).first()
    if not user:
        return {"success": False, "message": "用户不存在"}
    if not verify_password(password, user.password):
        return {"success": False, "message": "密码不匹配"}
    return {
        "success": True,
        "message": "登录成功",
        "time": datetime.now().isoformat(timespec="seconds") + 'Z',
        "data": {
            "id": user.id,
            "nickname": user.nickname,
            "jwt_token": generate_jwt_token(user)
        }
    }


@app.get("/data/list")
async def get_data_list(*, session: Session = Depends(get_session), user_id: int) -> object:
    """
    Get data list owned by certain user
    :param session: Session
    :param user_id: owner id
    :return: Result with data list classified by cls
    """
    user = session.get(User, user_id)
    if not user:
        return {"success": False, "message": "用户不存在"}
    statement = select(MildewData).where(MildewData.user == user)
    data = session.exec(statement).all()
    result = []
    for item in data:
        result.append({
            "id": item.id,
            "type": item.type,
            "image": item.image,
            "time": item.created_at,
            "count": Processor.organize_downy_detected_info(json.loads(item.data.replace("\'", "\""))),
            "data": Processor.organize_detected_result(json.loads(item.data.replace("\'", "\""))),
        })
    return {
        "success": True,
        "message": "数据获取成功",
        "time": datetime.now().isoformat(timespec="seconds") + 'Z',
        "data": result
    }


@app.get("/data/recent")
async def get_recent_data(*, session: Session = Depends(get_session), user_id: int, count: int = 3) -> object:
    user = session.get(User, user_id)
    if not user:
        return {"success": False, "message": "用户不存在"}
    statement = select(MildewData).where(MildewData.user == user).order_by(col(MildewData.created_at).desc()).limit(
        count)
    data = session.exec(statement).all()
    result = []
    for item in data:
        result.append({
            "id": item.id,
            "type": item.type,
            "image": item.image,
            "time": item.created_at,
            "count": Processor.organize_downy_detected_info(json.loads(item.data.replace("\'", "\""))),
            "data": Processor.organize_detected_result(json.loads(item.data.replace("\'", "\""))),
        })
    return {
        "success": True,
        "message": "数据获取成功",
        "time": datetime.now().isoformat(timespec="seconds") + 'Z',
        "data": result
    }
