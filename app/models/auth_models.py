from sqlalchemy import Column, Integer, String
from pydantic import BaseModel
from app.db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)


class UserCreate(BaseModel):
    name: str
    email: str
    password: str
