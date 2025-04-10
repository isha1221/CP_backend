# app/schemas.py
from typing import Optional
from pydantic import BaseModel

class ResumeBase(BaseModel):
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    skills: Optional[str]
    experience: Optional[str]

class ResumeCreate(ResumeBase):
    pass

class ResumeResponse(ResumeBase):
    id: int

    class Config:
        orm_mode = True  