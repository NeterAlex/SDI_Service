import io
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Annotated

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Depends, Request, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Session, select, col, create_engine
from starlette.staticfiles import StaticFiles

from model import User, MildewData
from utils import (
    Predictor,
    hash_password,
    generate_jwt_token,
    verify_password,
    decode_jwt_token,
    Processor,
)

# Initialize server
os.environ["TZ"] = "Asia/Shanghai"

# Initialize db
sqlite_url: str = "sqlite:///database.db"
connect_args = {"check_same_thread": False}
engine = None


def get_session():
    with Session(engine) as session:
        yield session


# Model Configurations
powdery_model_path: str = r"assets/models/powdery_m_20240106.pt"
downy_model_path: str = r"assets/models/downy_m_20231108.pt"
frogeye_model_path: str = r"assets/models/frogeye_l_20241109.pt"


downy_pd: Predictor
powdery_pd: Predictor
frogeye_pd: Predictor


# On Startup
@asynccontextmanager
async def lifespan(_app: FastAPI):
    global downy_pd, powdery_pd, frogeye_pd, engine
    print("Starting Server")
    print(datetime.now().isoformat(timespec="seconds") + "Z")
    # Initialize SQLite
    print("Initializing Database")
    engine = create_engine(sqlite_url, echo=False, connect_args=connect_args)
    SQLModel.metadata.create_all(engine)
    # Initialize predictors
    print("Initializing Models")
    downy_pd = Predictor(downy_model_path)
    powdery_pd = Predictor(powdery_model_path)
    frogeye_pd = Predictor(frogeye_model_path)
    yield
    print("Shutting down Service")


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Global error handler
@app.exception_handler(Exception)
async def exception_handler(_request, exc):
    logging.exception(f"[Error] {exc}")
    return {
        "success": False,
        "message": "处理失败，由于" + str(exc),
        "time": datetime.now().isoformat(timespec="seconds") + "Z",
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
    path: str = request.get("path")
    # Exclude /login & /docs & /static
    if (
        path.startswith("/ping")
        | path.startswith("/static")
        | path.startswith("/user/login")
        | path.startswith("/user/register")
    ):
        response = await call_next(request)
        return response
    else:
        try:
            # Get token
            authorization: str = request.headers.get("Authorization")
            if authorization is None:
                return Response(status_code=401)
            token = authorization.split(" ")[1]
            # Verify token
            user_claim = decode_jwt_token(token)
            if user_claim.get("user_id") is not None:
                response = await call_next(request)
                return response
        except Exception:
            return Response(status_code=401)


# Controllers
@app.get("/ping")
async def root():
    return {"message": "pong"}


@app.post("/calc/downy")
async def downy_mildew_detect(
    *,
    session: Session = Depends(get_session),
    file: UploadFile = File(...),
    user_id: Annotated[int, Form()],
) -> object:
    """
    Downy Mildew Detect Controller
    :param user_id: User ID
    :param session: Session
    :param file: Image file
    :return: Result with data and other info
    """
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    data, result_bytes = downy_pd.predict_bytes(image, False)
    await file.close()
    # compress image
    compressed_image = Image.open(io.BytesIO(result_bytes))
    compressed_stream = io.BytesIO()
    compressed_image.save(compressed_stream, format="JPEG", quality=25, optimize=True)
    compressed_bytes = compressed_stream.getvalue()
    # save image
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    save_relative_path = os.path.join(
        "static",
        "results",
        "downy-mildew",
        f"ID{user_id}-{current_time}-{uuid.uuid4().hex[:8]}.jpg",
    )
    save_obvious_path = os.path.join(os.getcwd(), save_relative_path)
    with open(save_obvious_path, "wb") as f:
        f.write(compressed_bytes)
    # operate db
    user = session.get(User, user_id)
    if not user:
        return {"success": False, "message": "用户不存在"}
    data = MildewData(
        user=user, type="downy", data=data.__str__(), image=save_relative_path
    )
    session.add(data)
    session.commit()
    # make result
    result = {
        "success": True,
        "message": "识别成功",
        "time": datetime.now().isoformat(timespec="seconds") + "Z",
        "data": data,
    }
    return result


@app.post("/calc/powdery")
async def powdery_mildew_detect(
    *,
    session: Session = Depends(get_session),
    file: UploadFile = File(...),
    user_id: Annotated[int, Form()],
) -> object:
    """
    Powdery Mildew Detect Controller
    :param user_id: User ID
    :param session: Session
    :param file: Image file
    :return: Result with data and other info
    """
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    data, result_bytes = powdery_pd.predict_bytes(image, False)
    await file.close()
    # compress image
    compressed_image = Image.open(io.BytesIO(result_bytes))
    compressed_stream = io.BytesIO()
    compressed_image.save(compressed_stream, format="JPEG", quality=25, optimize=True)
    compressed_bytes = compressed_stream.getvalue()
    # save image
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    save_relative_path = os.path.join(
        "static",
        "results",
        "powdery-mildew",
        f"ID{user_id}-{current_time}-{uuid.uuid4().hex[:8]}.jpg",
    )
    save_obvious_path = os.path.join(os.getcwd(), save_relative_path)
    with open(save_obvious_path, "wb") as f:
        f.write(compressed_bytes)
    # operate db
    user = session.get(User, user_id)
    if not user:
        return {"success": False, "message": "用户不存在"}
    data = MildewData(
        user=user, type="powdery", data=data.__str__(), image=save_relative_path
    )
    session.add(data)
    session.commit()
    # make result
    result = {
        "success": True,
        "message": "识别成功",
        "time": datetime.now().isoformat(timespec="seconds") + "Z",
        "data": data,
    }
    return result


@app.post("/calc/frogeye")
async def frogeye_detect(
    *,
    session: Session = Depends(get_session),
    file: UploadFile = File(...),
    user_id: Annotated[int, Form()],
) -> None:
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    data, result_bytes = powdery_pd.predict_bytes(image, False)
    await file.close()
    pass


@app.post("/user/register")
async def register_user(
    *,
    session: Session = Depends(get_session),
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    nickname: Annotated[str, Form()],
) -> object:
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
async def login_user(
    *,
    session: Session = Depends(get_session),
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
) -> object:
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
        "time": datetime.now().isoformat(timespec="seconds") + "Z",
        "data": {
            "id": user.id,
            "nickname": user.nickname,
            "jwt_token": generate_jwt_token(user),
        },
    }


@app.get("/data/list")
async def get_data_list(
    *, session: Session = Depends(get_session), user_id: int
) -> object:
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
        result.append(
            {
                "id": item.id,
                "type": item.type,
                "image": item.image,
                "time": item.created_at + timedelta(hours=8),
                "count": Processor.organize_downy_detected_info(
                    json.loads(item.data.replace("'", '"'))
                ),
                "data": Processor.organize_detected_result(
                    json.loads(item.data.replace("'", '"'))
                ),
            }
        )
    return {
        "success": True,
        "message": "数据获取成功",
        "time": datetime.now().isoformat(timespec="seconds") + "Z",
        "data": result,
    }


@app.get("/data/recent")
async def get_recent_data(
    *, session: Session = Depends(get_session), user_id: int, count: int = 3
) -> object:
    user = session.get(User, user_id)
    if not user:
        return {"success": False, "message": "用户不存在"}
    statement = (
        select(MildewData)
        .where(MildewData.user == user)
        .order_by(col(MildewData.created_at).desc())
        .limit(count)
    )
    data = session.exec(statement).all()
    result = []
    for item in data:
        result.append(
            {
                "id": item.id,
                "type": item.type,
                "image": item.image,
                "time": item.created_at + timedelta(hours=8),
                "count": Processor.organize_downy_detected_info(
                    json.loads(item.data.replace("'", '"'))
                ),
                "data": Processor.organize_detected_result(
                    json.loads(item.data.replace("'", '"'))
                ),
            }
        )
    return {
        "success": True,
        "message": "数据获取成功",
        "time": datetime.now().isoformat(timespec="seconds") + "Z",
        "data": result,
    }


@app.delete("/data")
async def delete_data(
    *, session: Session = Depends(get_session), user_id: int, data_id: int
) -> object:
    try:
        user = session.get(User, user_id)
        if not user:
            return {"success": False, "message": "用户不存在"}
        statement = select(MildewData).where(MildewData.id == data_id)
        data = session.exec(statement).one()
        if not data:
            return {"success": False, "message": "数据不存在"}
        if data.user_id != user.id:
            return {"success": False, "message": "不可删除不属于自己的数据"}
        session.delete(data)
        session.commit()
        path = os.path.join(os.getcwd(), data.image)
        if os.path.exists(path):
            os.remove(path)
        return {
            "success": True,
            "message": "数据删除成功",
            "time": datetime.now().isoformat(timespec="seconds") + "Z",
            "data": "",
        }
    except Exception as e:
        raise e
