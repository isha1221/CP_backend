from sqlalchemy import Column, Integer, String, Enum
from pydantic import BaseModel
from app.db import Base
import enum


class PlanType(enum.Enum):
    RP_BASIC = "RP_BASIC"
    RP_MOD = "RP_MOD"
    RP_PRO = "RP_PRO"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    plan_type = Column(Enum(PlanType), nullable=False,
                       default=PlanType.RP_BASIC)
    resume_count = Column(Integer, nullable=False, default=0)


class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    plan_type: PlanType | None = PlanType.RP_BASIC
