from datetime import datetime, timedelta

import jwt
from dotenv import dotenv_values
from passlib.context import CryptContext

from model import User

pwd_context = CryptContext(schemes=["bcrypt"])

config = dotenv_values(".env")
JWT_SECRET_KEY = config.get("JWT_SECRET_KEY")
JWT_TOKEN_EXPIRE_HOUR = config.get("JWT_TOKEN_EXPIRE_HOUR")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def generate_jwt_token(user: User) -> str:
    expires = datetime.utcnow() + timedelta(hours=float(JWT_TOKEN_EXPIRE_HOUR))
    payload = {
        'exp': expires,
        'iat': datetime.utcnow(),
        'user_id': user.id,
        'nickname': user.nickname
    }
    token = jwt.encode(payload=payload, key=JWT_SECRET_KEY, algorithm='HS256')
    return token


def decode_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        if datetime.utcnow() > payload['exp']:
            raise jwt.ExpiredSignatureError("Token已过期")
        return payload
    except jwt.ExpiredSignatureError:
        raise jwt.ExpiredSignatureError("Token已过期")
    except jwt.InvalidTokenError:
        raise jwt.InvalidTokenError("无效的Token")
