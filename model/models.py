from datetime import datetime
from typing import Optional, List

from sqlmodel import SQLModel, Field, Relationship


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password: str
    nickname: str
    datasets: List["MildewData"] = Relationship(back_populates="user")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MildewData(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    type: str = Field(index=True, default='mildew')
    data: str = Field(default="{}")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    user_id: Optional[int] = Field(default=None, foreign_key='user.id')
    user: Optional["User"] = Relationship(back_populates='datasets')
